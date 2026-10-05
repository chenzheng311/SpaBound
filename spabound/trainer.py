# Reorganized from spatialVAE/SpatialGlue_MMVAE.py; see the repository license and attribution notices.
import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import SpaBoundModel, SpaBoundPrivateReconModel
from .losses import KLAnnealer, EdgePreservingLoss


class SpaBound:
    """Train on aligned AnnData features and precomputed spatial/feature graphs.

    ``data`` contains ``adata_omics1`` and ``adata_omics2`` with dense
    ``.obsm['feat']`` arrays in the same spot order, plus the four adjacency
    tensors returned by a graph constructor. ``train()`` returns NumPy arrays;
    ``output['SpaBound']`` is the normalized shared posterior-mean embedding.
    """

    def __init__(
        self,
        data,
        datatype='SPOTS',
        device=torch.device('cpu'),
        random_seed=2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        z_dim=64,
        n_latent_shared=32,
        n_latent_private=16,
        mmvae_weight=0.1,
        reconstruction_weight=0.1,
        kl_weight=0.01,
        use_appnp=True,
        appnp_hidden=128,
        appnp_layers=3,
        appnp_alpha=0.1,
        appnp_residual=True,
        appnp_dropout=0.0,  #  Disable dropout by default
        #  Boundary-aware graph construction
        use_boundary_aware_graph=False,
        gradient_penalty=0.5,
        sigma_spatial=1.0,
        sigma_feature=1.0,
        #  Edge-preserving regularization
        use_edge_preserving_loss=False,
        edge_preserving_weight=0.01,
        edge_preserving_type='charbonnier',
        # KL annealing
        kl_annealing=True,
        kl_annealing_start=0,
        kl_annealing_end=100,
        use_private_reconstruction=False
    ):
        '''
        Train SpaBound on two aligned modalities and four precomputed graphs.

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input. Default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? Default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. Default is 2022.
        learning_rate : float, optional
            Learning rate for representation learning. Default is 0.0001.
        weight_decay : float, optional
            Weight decay. Default is 0.00.
        epochs : int, optional
            Epoch for model training. Default is 600.
        dim_input : int, optional
            Dimension of input feature. Default is 3000.
        dim_output : int, optional
            Dimension of output representation. Default is 64.
        z_dim : int, optional
            Compatibility argument. Effective width is shared + 2 * private.
        n_latent_shared : int, optional
            Dimension of shared latent space. Default is 32.
        n_latent_private : int, optional
            Dimension of private latent space per modality. Default is 16.
        mmvae_weight : float, optional
            Weight for MMVAE fusion losses. Default is 0.1.
        reconstruction_weight : float, optional
            Weight for cross-modal reconstruction losses. Default is 0.1.
        kl_weight : float, optional
            Weight for KL divergence losses. Default is 0.01.
        use_appnp : bool, optional
            Whether to use APPNP encoders instead of standard GCN. Default is True.
        appnp_hidden : int, optional
            Hidden dimension for APPNP encoders. Default is 128.
        appnp_layers : int, optional
            Number of APPNP propagation layers. Default is 3.
        appnp_alpha : float, optional
            Teleport probability for APPNP. Default is 0.1.
        appnp_residual : bool, optional
            Whether to use residual connections in APPNP. Default is True.
        appnp_dropout : float, optional
            Dropout probability for APPNP encoders. Default is 0.0 (set to 0 for reproducibility).
        use_boundary_aware_graph : bool, optional
            Metadata flag. Graphs are supplied in data and are not rebuilt here.
        gradient_penalty : float, optional
            Penalty for boundary edges in graph construction (0-1). Default is 0.5.
        sigma_spatial : float, optional
            Bandwidth for spatial RBF kernel. Default is 1.0.
        sigma_feature : float, optional
            Bandwidth for feature similarity kernel. Default is 1.0.
        use_edge_preserving_loss : bool, optional
            Whether to use edge-preserving smoothness loss. Default is False.
        edge_preserving_weight : float, optional
            Weight for edge-preserving loss. Default is 0.01.
        edge_preserving_type : str, optional
            Type of robust penalty ('charbonnier', 'l1', 'huber'). Default is 'charbonnier'.

        Returns
        -------
        train() returns a dictionary; the shared embedding is stored under 'SpaBound'.
        '''

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed

        #  Initialize random number generators
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        import numpy as np
        np.random.seed(self.random_seed)
        import random
        random.seed(self.random_seed)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.z_dim = z_dim
        self.n_latent_shared = n_latent_shared
        self.n_latent_private = n_latent_private
        self.mmvae_weight = mmvae_weight
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.use_appnp = use_appnp
        self.appnp_hidden = appnp_hidden
        self.appnp_layers = appnp_layers
        self.appnp_alpha = appnp_alpha
        self.appnp_residual = appnp_residual
        self.appnp_dropout = appnp_dropout  #  Store dropout parameter

        # Boundary-aware parameters
        self.use_boundary_aware_graph = use_boundary_aware_graph
        self.gradient_penalty = gradient_penalty
        self.sigma_spatial = sigma_spatial
        self.sigma_feature = sigma_feature
        self.use_edge_preserving_loss = use_edge_preserving_loss
        self.edge_preserving_weight = edge_preserving_weight
        self.edge_preserving_type = edge_preserving_type

        # KL annealing
        self.kl_annealing = kl_annealing
        self.kl_annealer = KLAnnealer(
            start_epoch=kl_annealing_start,
            end_epoch=kl_annealing_end,
            min_weight=0.0,
            max_weight=1.0
        ) if kl_annealing else None
        self.use_private_reconstruction = use_private_reconstruction


        # Initialize adjacency matrices
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']

        #  Use dense adjacency to retain the original training implementation
        # GPU sparse operations may be nondeterministic
        print("Converting adjacency matrices to dense tensors...")

        adj_spatial_omics1_tensor = self.data['adj_spatial_omics1']
        adj_spatial_omics2_tensor = self.data['adj_spatial_omics2']
        adj_feature_omics1_tensor = self.data['adj_feature_omics1']
        adj_feature_omics2_tensor = self.data['adj_feature_omics2']

        # Convert to dense tensors
        if adj_spatial_omics1_tensor.is_sparse:
            self.adj_spatial_omics1 = adj_spatial_omics1_tensor.to_dense().to(self.device)
        else:
            self.adj_spatial_omics1 = adj_spatial_omics1_tensor.to(self.device)

        if adj_spatial_omics2_tensor.is_sparse:
            self.adj_spatial_omics2 = adj_spatial_omics2_tensor.to_dense().to(self.device)
        else:
            self.adj_spatial_omics2 = adj_spatial_omics2_tensor.to(self.device)

        if adj_feature_omics1_tensor.is_sparse:
            self.adj_feature_omics1 = adj_feature_omics1_tensor.to_dense().to(self.device)
        else:
            self.adj_feature_omics1 = adj_feature_omics1_tensor.to(self.device)

        if adj_feature_omics2_tensor.is_sparse:
            self.adj_feature_omics2 = adj_feature_omics2_tensor.to_dense().to(self.device)
        else:
            self.adj_feature_omics2 = adj_feature_omics2_tensor.to(self.device)

        print(" Adjacency matrices converted to dense tensors")

        # Initialize features
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        # Get dimensions
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]

        # Initialize edge-preserving loss module if enabled
        if self.use_edge_preserving_loss:
            self.edge_preserving_loss = EdgePreservingLoss(
                loss_type=self.edge_preserving_type,
                epsilon=0.001
            ).to(self.device)
            print(f"Edge-preserving loss enabled: {self.edge_preserving_type} (weight={self.edge_preserving_weight})")

    def train(self):
        """Fit a new model and return shared, auxiliary and modality embeddings."""
        # Initialize enhanced model with variational consensus fusion
        # Choose between APPNP and standard GCN encoders
        model_cls = (
            SpaBoundPrivateReconModel
            if self.use_private_reconstruction
            else SpaBoundModel
        )
        if self.use_appnp:
            print(f"Initializing SpaBound with APPNP encoders...")
            if self.use_private_reconstruction:
                print("  Private reconstruction: enabled")
            print(f"  APPNP hidden dim: {self.appnp_hidden}")
            print(f"  APPNP layers: {self.appnp_layers}")
            print(f"  APPNP alpha: {self.appnp_alpha}")
            print(f"  APPNP residual: {self.appnp_residual}")
            print(f"  APPNP dropout: {self.appnp_dropout}")

            self.model = model_cls(
                dim_in_feat_omics1=self.dim_input1,
                dim_out_feat_omics1=self.dim_output,
                dim_in_feat_omics2=self.dim_input2,
                dim_out_feat_omics2=self.dim_output,
                z_dim=self.z_dim,
                n_latent_shared=self.n_latent_shared,
                n_latent_private=self.n_latent_private,
                use_appnp=True,
                appnp_hidden=self.appnp_hidden,
                appnp_layers=self.appnp_layers,
                appnp_alpha=self.appnp_alpha,
                appnp_residual=self.appnp_residual,
                dropout=self.appnp_dropout
            ).to(self.device)
        else:
            print(f"Initializing SpaBound with standard GCN encoders...")
            if self.use_private_reconstruction:
                print("  Private reconstruction: enabled")
            self.model = model_cls(
                dim_in_feat_omics1=self.dim_input1,
                dim_out_feat_omics1=self.dim_output,
                dim_in_feat_omics2=self.dim_input2,
                dim_out_feat_omics2=self.dim_output,
                z_dim=self.z_dim,
                n_latent_shared=self.n_latent_shared,
                n_latent_private=self.n_latent_private,
                use_appnp=False
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        print(f"\nTraining SpaBound with {self.epochs} epochs...")
        print(f"Encoder type: {'APPNP' if self.use_appnp else 'Standard GCN'}")
        print(f"MMVAE latent dims: shared={self.n_latent_shared}, private={self.n_latent_private}")
        print(f"Private reconstruction: {self.use_private_reconstruction}")

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2
            )

            # Reconstruction losses (MSE)
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # self.loss_corr_omics1 = F.mse_loss(
            #     results['emb_latent_omics1'],
            #     results['emb_latent_omics1_across_recon']
            # )
            # self.loss_corr_omics2 = F.mse_loss(
            #     results['emb_latent_omics2'],
            #     results['emb_latent_omics2_across_recon']
            # )

            # variational consensus additional losses
            mmvae_results = results['mmvae_results']

            # Cross-modal reconstruction losses
            # Compare original encoded representations with cross-modal reconstructions
            loss_cross_recon_1 = F.mse_loss(
                results['emb_latent_omics1'],
                mmvae_results['recon_1_from_2']
            )
            loss_cross_recon_2 = F.mse_loss(
                results['emb_latent_omics2'],
                mmvae_results['recon_2_from_1']
            )

            # KL divergence losses with optional annealing
            kl_loss_1 = self.compute_kl_loss(mmvae_results['mu1'], mmvae_results['logvar1'])
            kl_loss_2 = self.compute_kl_loss(mmvae_results['mu2'], mmvae_results['logvar2'])

            #  KL annealing
            kl_weight_current = self.kl_weight
            if self.kl_annealer is not None:
                kl_weight_current = self.kl_weight * self.kl_annealer.get_weight(epoch)

            # Total loss
            loss_reconstruction = (
                self.reconstruction_weight * self.loss_recon_omics1 +
                self.reconstruction_weight * self.loss_recon_omics2
            )

            loss_mmvae = (
                #self.reconstruction_weight * (loss_cross_recon_1 + loss_cross_recon_2) +
                kl_weight_current * (kl_loss_1 + kl_loss_2)  #  Apply the annealed KL weight
            )

            #  Edge-preserving smoothness loss (optional)
            loss_edge_preserving = torch.tensor(0.0, device=self.device)
            if self.use_edge_preserving_loss and hasattr(self, 'edge_preserving_loss'):
                # Apply to the fused shared representation
                z_shared = mmvae_results['shared_representation']

                # Edge-preserving loss on spatial graph
                loss_edge_spatial_1 = self.edge_preserving_loss(
                    z_shared,
                    self.adj_spatial_omics1
                )
                loss_edge_spatial_2 = self.edge_preserving_loss(
                    z_shared,
                    self.adj_spatial_omics2
                )

                loss_edge_preserving = (loss_edge_spatial_1 + loss_edge_spatial_2) / 2.0

            total_loss = (
                loss_reconstruction +
                self.mmvae_weight * loss_mmvae +
                self.edge_preserving_weight * loss_edge_preserving
            )

            #total_loss = self.mmvae_weight * loss_mmvae
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Log progress
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"  Recon Loss (omics1): {self.loss_recon_omics1.item():.4f}")
                print(f"  Recon Loss (omics2): {self.loss_recon_omics2.item():.4f}")
                print(f"  KL Loss: {(kl_loss_1 + kl_loss_2).item():.4f}")
                if self.kl_annealer is not None:
                    print(f"  KL Weight (annealed): {kl_weight_current:.4f}")
                print(f"  MMVAE Loss: {loss_mmvae.item():.4f}")
                if self.use_edge_preserving_loss:
                    print(f"  Edge-Preserving Loss: {loss_edge_preserving.item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")

        print("Model training finished!\n")

        # Extract final representations
        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1,
                self.features_omics2,
                self.adj_spatial_omics1,
                self.adj_feature_omics1,
                self.adj_spatial_omics2,
                self.adj_feature_omics2
            )

        # Normalize representations
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        # Extract MMVAE-specific information
        mmvae_results = results['mmvae_results']
        z_shared = F.normalize(mmvae_results['shared_representation'], p=2, eps=1e-12, dim=1)
        z1_private = F.normalize(
            mmvae_results['z1'][:, self.n_latent_shared:self.n_latent_shared + self.n_latent_private],
            p=2, eps=1e-12, dim=1
        )
        z2_private = F.normalize(
            mmvae_results['z2'][:, self.n_latent_shared + self.n_latent_private:],
            p=2, eps=1e-12, dim=1
        )

        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'SpaBound': emb_combined.detach().cpu().numpy(),
            'MMVAE_Shared': z_shared.detach().cpu().numpy(),
            'MMVAE_Private_1': z1_private.detach().cpu().numpy(),
            'MMVAE_Private_2': z2_private.detach().cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
            'alpha_mmvae': results['alpha'].detach().cpu().numpy(),
            'mmvae_mu1': mmvae_results['mu1'].detach().cpu().numpy(),
            'mmvae_mu2': mmvae_results['mu2'].detach().cpu().numpy(),
            'mmvae_logvar1': mmvae_results['logvar1'].detach().cpu().numpy(),
            'mmvae_logvar2': mmvae_results['logvar2'].detach().cpu().numpy()
        }

        return output

    def compute_kl_loss(self, mu, logvar):
        """Compute KL divergence loss"""
        # KL divergence between N(mu, sigma) and N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def get_model_info(self):
        """Get information about the model architecture"""
        info = {
            'model_type': 'SpaBoundModel' if self.use_appnp else 'SpaBound',
            'encoder_type': 'APPNP' if self.use_appnp else 'Standard GCN',
            'input_dims': [self.dim_input1, self.dim_input2],
            'output_dims': [self.dim_output, self.dim_output],
            'mmvae_config': {
                'z_dim': self.z_dim,
                'n_latent_shared': self.n_latent_shared,
                'n_latent_private': self.n_latent_private
            },
            'appnp_config': {
                'use_appnp': self.use_appnp,
                'hidden_dim': self.appnp_hidden if self.use_appnp else None,
                'num_layers': self.appnp_layers if self.use_appnp else None,
                'alpha': self.appnp_alpha if self.use_appnp else None,
                'residual': self.appnp_residual if self.use_appnp else None
            },
            'boundary_aware_config': {
                'use_boundary_aware_graph': self.use_boundary_aware_graph,
                'gradient_penalty': self.gradient_penalty if self.use_boundary_aware_graph else None,
                'sigma_spatial': self.sigma_spatial if self.use_boundary_aware_graph else None,
                'sigma_feature': self.sigma_feature if self.use_boundary_aware_graph else None,
                'use_edge_preserving_loss': self.use_edge_preserving_loss,
                'edge_preserving_weight': self.edge_preserving_weight if self.use_edge_preserving_loss else None,
                'edge_preserving_type': self.edge_preserving_type if self.use_edge_preserving_loss else None,
                'use_private_reconstruction': self.use_private_reconstruction
            },
            'training_params': {
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            },
            'loss_weights': {
                'mmvae_weight': self.mmvae_weight,
                'reconstruction_weight': self.reconstruction_weight,
                'kl_weight': self.kl_weight
            }
        }
        return info

    def extract_latent_components(self, data):
        """
        Extract individual latent components (shared and private) for analysis

        Parameters
        ----------
        data : dict
            Output dictionary from train() method

        Returns
        -------
        dict
            Dictionary containing separated latent components
        """
        components = {
            'shared_representation': data.get('MMVAE_Shared'),
            'private_omics1': data.get('MMVAE_Private_1'),
            'private_omics2': data.get('MMVAE_Private_2'),
            'combined_fusion': data.get('SpaBound'),
            'original_omics1': data.get('emb_latent_omics1'),
            'original_omics2': data.get('emb_latent_omics2')
        }
        return components
