# Reorganized from spatialVAE/model_mmvae_appnp.py; see the repository license and attribution notices.
"""
SpaBound with optional APPNP encoders

This module implements SpaBound with APPNP (Approximate Personalized
Propagation of Neural Predictions) encoders as an alternative to standard GCN encoders.

Usage:
    model = SpaBoundModel(
        ...,
        use_appnp=True,  # Enable APPNP encoders
        appnp_hidden=128,
        appnp_layers=3,
        appnp_alpha=0.1
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np




class APPNPEncoder(Module):
    """
    APPNP (Approximate Personalized Propagation of Neural Predictions) Encoder

    Combines feature transformation with personalized PageRank for better
    long-range dependency capture while avoiding over-smoothing.

    Parameters
    ----------
    in_feat : int
        Dimension of input features
    hidden_feat : int
        Dimension of hidden layers
    out_feat : int
        Dimension of output latent representation
    num_layers : int
        Number of propagation layers (default: 2)
    alpha : float
        Teleport probability, controls interpolation between input and propagated features
        Higher values = more reliance on original features (default: 0.1)
    dropout : float
        Dropout probability (default: 0.1)
    residual : bool
        Whether to use residual connections (default: True)

    Reference
    ---------
    Klicpera et al., "Predict then Propagate: Graph Neural Networks meet Personalized PageRank"
    ICLR 2020
    """

    def __init__(self, in_feat, hidden_feat, out_feat, num_layers=2,
                 alpha=0.1, dropout=0.1, residual=True):
        super(APPNPEncoder, self).__init__()

        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat
        self.num_layers = num_layers
        self.alpha = alpha
        self.dropout = dropout
        self.residual = residual

        # Feature transformation layers (MLP)
        self.fc1 = nn.Linear(in_feat, hidden_feat)
        self.fc2 = nn.Linear(hidden_feat, out_feat)

        # Residual projection if dimensions don't match
        if residual and in_feat != out_feat:
            self.residual_proj = nn.Linear(in_feat, out_feat)
        else:
            self.residual_proj = nn.Identity()

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_feat)
        self.bn2 = nn.BatchNorm1d(out_feat)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat, adj):
        """
        Forward pass of APPNP encoder

        Parameters
        ----------
        feat : torch.Tensor
            Node features [n_nodes, in_feat]
        adj : torch.Tensor
            Sparse adjacency matrix [n_nodes, n_nodes]

        Returns
        -------
        torch.Tensor
            Encoded node representations [n_nodes, out_feat]
        """
        # Initial feature transformation
        x = self.fc1(feat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        x = self.fc2(x)
        x = self.bn2(x)

        # APPNP propagation
        # h^(0) = MLP(features)
        h = x.clone()

        # Iterative propagation: h^(k) = (1 - alpha) * A * h^(k-1) + alpha * h^(0)
        for k in range(self.num_layers):
            #  Support dense and sparse adjacency
            # Neighborhood aggregation
            if adj.is_sparse:
                h_propagated = torch.spmm(adj, h)
            else:
                h_propagated = adj @ h

            # Teleport to initial features (prevents over-smoothing)
            h = (1 - self.alpha) * h_propagated + self.alpha * x

        # Optional residual connection
        if self.residual:
            residual = self.residual_proj(feat)
            h = h + residual

        return h
# GCN components
class Encoder(Module):
    """Standard GCN encoder"""
    def __init__(self, in_feat, out_feat):
        super(Encoder, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat @ self.weight
        #  Support dense and sparse adjacency
        if adj.is_sparse:
            x = torch.spmm(adj, x)
        else:
            x = adj @ x
        return x


class AttentionLayer(Module):
    """Two-view attention layer"""
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.w_omega.size(1))
        self.w_omega.data.uniform_(-stdv, stdv)
        stdv = 1. / np.sqrt(self.u_omega.size(0))
        self.u_omega.data.uniform_(-stdv, stdv)

    def forward(self, emb1, emb2):
        # emb1, emb2: [n_nodes, in_feat]
        v = torch.tanh(torch.matmul(torch.cat([emb1, emb2], dim=0), self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu, dim=0)
        emb_combined = alpha[0] * emb1 + alpha[1] * emb2
        return emb_combined, alpha


class Decoder(Module):
    """Graph decoder"""
    def __init__(self, in_feat, out_feat):
        super(Decoder, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat @ self.weight
        #  Support dense and sparse adjacency
        if adj.is_sparse:
            x = torch.spmm(adj, x)
        else:
            x = adj @ x
        return x


class MMVAEFusionModule(Module):
    """variational consensus fusion module"""
    def __init__(self, input_dim_1, input_dim_2, z_dim, n_shared, n_private):
        super(MMVAEFusionModule, self).__init__()

        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.z_dim = z_dim
        self.n_shared = n_shared
        self.n_private = n_private

        # Encoders for each modality
        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim_1, z_dim * 2)
        )
        self.encoder_2 = nn.Sequential(
            nn.Linear(input_dim_2, z_dim * 2)
        )

        # Decoders for cross-modal reconstruction
        self.decoder_1 = nn.Linear(z_dim, input_dim_1)
        self.decoder_2 = nn.Linear(z_dim, input_dim_2)

        # Sparsity mask for shared/private decomposition
        mask = torch.zeros(2, z_dim)
        mask[0, :n_shared] = 1.0
        mask[0, n_shared:n_shared+n_private] = 1.0
        mask[1, :n_shared] = 1.0
        mask[1, n_shared+n_private:] = 1.0
        self.register_buffer('sparsity_mask', mask)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, emb1, emb2):
        # Encode
        h1 = self.encoder_1(emb1)
        mu1, logvar1 = h1.chunk(2, dim=-1)

        h2 = self.encoder_2(emb2)
        mu2, logvar2 = h2.chunk(2, dim=-1)

        # Sample from distributions
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        # Apply sparsity mask
        z1_masked = z1 * self.sparsity_mask[0]
        z2_masked = z2 * self.sparsity_mask[1]

        # Extract shared representation (average the shared components)
        z1_shared = z1_masked[:, :self.n_shared]
        z2_shared = z2_masked[:, :self.n_shared]
        z_shared = (z1_shared + z2_shared) / 2

        # Cross-modal reconstruction
        recon_1_from_2 = self.decoder_1(z2_masked)
        recon_2_from_1 = self.decoder_2(z1_masked)

        # Compute attention weights (based on shared representation norm)
        norm1 = z1_shared.norm(dim=1, keepdim=True) + 1e-8
        norm2 = z2_shared.norm(dim=1, keepdim=True) + 1e-8
        attention_weights = torch.stack([norm1, norm2], dim=1)
        attention_weights = F.softmax(attention_weights, dim=1)

        return {
            'shared_representation': z_shared,
            'z1': z1,
            'z2': z2,
            'z1_private': z1_masked[:, self.n_shared:self.n_shared + self.n_private],
            'z2_private': z2_masked[:, self.n_shared + self.n_private:],
            'mu1': mu1,
            'logvar1': logvar1,
            'mu2': mu2,
            'logvar2': logvar2,
            'recon_1_from_2': recon_1_from_2,
            'recon_2_from_1': recon_2_from_1,
            'attention_weights': attention_weights.squeeze(-1)
        }


class SpaBoundModel(Module):
    """
    SpaBound with optional APPNP encoders

    This model allows users to choose between:
    - Standard GCN encoders (use_appnp=False)
    - APPNP encoders for better long-range dependency capture (use_appnp=True)

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Input feature dimension for omics1
    dim_out_feat_omics1 : int
        Output latent dimension for omics1
    dim_in_feat_omics2 : int
        Input feature dimension for omics2
    dim_out_feat_omics2 : int
        Output latent dimension for omics2
    z_dim : int
        Retained compatibility argument; effective width is shared + 2 * private.
    n_latent_shared : int
        Dimension of shared latent space
    n_latent_private : int
        Dimension of private latent space per modality
    use_appnp : bool
        Whether to use APPNP encoders (default: True)
    appnp_hidden : int
        Hidden dimension for APPNP (default: 128)
    appnp_layers : int
        Number of APPNP propagation layers (default: 3)
    appnp_alpha : float
        Teleport probability for APPNP (default: 0.1)
    appnp_residual : bool
        Whether to use residual connections in APPNP (default: True)
    dropout : float
        Dropout rate (default: 0.0)
    likelihood : str
        Likelihood type: 'Gaussian' (MSE), 'NB' (Negative Binomial), or 'ZINB' (Zero-Inflated NB)
    """

    def __init__(
        self,
        dim_in_feat_omics1,
        dim_out_feat_omics1,
        dim_in_feat_omics2,
        dim_out_feat_omics2,
        z_dim=64,
        n_latent_shared=32,
        n_latent_private=16,
        use_appnp=True,
        appnp_hidden=128,
        appnp_layers=3,
        appnp_alpha=0.1,
        appnp_residual=True,
        dropout=0.0,
        act=F.relu,
        cross_view_modification=True,
        likelihood="Gaussian",
        use_private_reconstruction=False
    ):
        super(SpaBoundModel, self).__init__()

        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.z_dim = z_dim
        self.n_latent_shared = n_latent_shared
        self.n_latent_private = n_latent_private
        self.total_z_dim = n_latent_shared + 2 * n_latent_private
        self.dropout = dropout
        self.act = act
        self.use_appnp = use_appnp
        self.likelihood = likelihood
        self.use_private_reconstruction = use_private_reconstruction

        # NB/ZINB parameters (only used when likelihood is NB or ZINB)
        if likelihood in ['NB', 'ZINB']:
            # Theta (inverse dispersion) - learned per gene
            self.theta_omics1 = Parameter(torch.ones(1, dim_in_feat_omics1))
            self.theta_omics2 = Parameter(torch.ones(1, dim_in_feat_omics2))

            if likelihood == 'ZINB':
                # Pi (dropout probability) - learned via neural network
                self.pi_net_omics1 = nn.Sequential(
                    nn.Linear(dim_out_feat_omics1, dim_out_feat_omics1 // 2),
                    nn.ReLU(),
                    nn.Linear(dim_out_feat_omics1 // 2, dim_in_feat_omics1)
                )
                self.pi_net_omics2 = nn.Sequential(
                    nn.Linear(dim_out_feat_omics2, dim_out_feat_omics2 // 2),
                    nn.ReLU(),
                    nn.Linear(dim_out_feat_omics2 // 2, dim_in_feat_omics2)
                )

        if use_appnp:
            # Use APPNP encoders
            print("Using APPNP encoders for enhanced long-range dependency capture")
            self.encoder_omics1_spatial = APPNPEncoder(
                dim_in_feat_omics1, appnp_hidden, dim_out_feat_omics1,
                num_layers=appnp_layers, alpha=appnp_alpha, dropout=dropout,
                residual=appnp_residual
            )
            self.encoder_omics1_feature = APPNPEncoder(
                dim_in_feat_omics1, appnp_hidden, dim_out_feat_omics1,
                num_layers=appnp_layers, alpha=appnp_alpha, dropout=dropout,
                residual=appnp_residual
            )
            self.encoder_omics2_spatial = APPNPEncoder(
                dim_in_feat_omics2, appnp_hidden, dim_out_feat_omics2,
                num_layers=appnp_layers, alpha=appnp_alpha, dropout=dropout,
                residual=appnp_residual
            )
            self.encoder_omics2_feature = APPNPEncoder(
                dim_in_feat_omics2, appnp_hidden, dim_out_feat_omics2,
                num_layers=appnp_layers, alpha=appnp_alpha, dropout=dropout,
                residual=appnp_residual
            )
        else:
            # Use standard GCN encoders
            print("Using standard GCN encoders")
            self.encoder_omics1 = Encoder(dim_in_feat_omics1, dim_out_feat_omics1)
            self.encoder_omics2 = Encoder(dim_in_feat_omics2, dim_out_feat_omics2)

        # Within-modality attention layers (keep original)
        self.atten_omics1 = AttentionLayer(dim_out_feat_omics1, dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(dim_out_feat_omics2, dim_out_feat_omics2)

        # variational consensus fusion module
        self.mmvae_fusion = MMVAEFusionModule(
            input_dim_1=dim_out_feat_omics1,
            input_dim_2=dim_out_feat_omics2,
            z_dim=self.total_z_dim,
            n_shared=n_latent_shared,
            n_private=n_latent_private
        )

        # Decoders (modified to work with MMVAE fused representation)
        self.decoder_omics1 = Decoder(dim_out_feat_omics1, dim_in_feat_omics1)
        self.decoder_omics2 = Decoder(dim_out_feat_omics2, dim_in_feat_omics2)

        # Final projection layers to map MMVAE output to decoder input space
        decoder_input_dim = self.n_latent_shared
        if self.use_private_reconstruction:
            decoder_input_dim += self.n_latent_private
        self.proj_to_omics1 = nn.Linear(decoder_input_dim, dim_out_feat_omics1)
        self.proj_to_omics2 = nn.Linear(decoder_input_dim, dim_out_feat_omics2)

        # Cross-modal encoding consistency networks
        self.cross_encode_omics2 = Encoder(dim_in_feat_omics2, dim_out_feat_omics1)
        self.cross_encode_omics1 = Encoder(dim_in_feat_omics1, dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2,
                adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2):

        if self.use_appnp:
            # Use APPNP encoders
            emb_latent_spatial_omics1 = self.encoder_omics1_spatial(features_omics1, adj_spatial_omics1)
            emb_latent_feature_omics1 = self.encoder_omics1_feature(features_omics1, adj_feature_omics1)
            emb_latent_spatial_omics2 = self.encoder_omics2_spatial(features_omics2, adj_spatial_omics2)
            emb_latent_feature_omics2 = self.encoder_omics2_feature(features_omics2, adj_feature_omics2)
        else:
            # Use standard GCN encoders
            emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
            emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
            emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
            emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # Within-modality attention aggregation
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)

        # variational consensus multi-modal fusion
        mmvae_results = self.mmvae_fusion(emb_latent_omics1, emb_latent_omics2)

        # Extract the fused shared representation
        emb_latent_combined = mmvae_results['shared_representation']
        alpha_mmvae = mmvae_results['attention_weights']

        if self.use_private_reconstruction:
            recon_input_omics1 = torch.cat(
                [emb_latent_combined, mmvae_results['z1_private']],
                dim=1
            )
            recon_input_omics2 = torch.cat(
                [emb_latent_combined, mmvae_results['z2_private']],
                dim=1
            )
        else:
            recon_input_omics1 = emb_latent_combined
            recon_input_omics2 = emb_latent_combined

        # Project fused representation to decoder input spaces
        emb_combined_omics1 = self.proj_to_omics1(recon_input_omics1)
        emb_combined_omics2 = self.proj_to_omics2(recon_input_omics2)

        # Decode back to original expression spaces
        emb_recon_omics1 = self.decoder_omics1(emb_combined_omics1, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_combined_omics2, adj_spatial_omics2)

        # Cross-modal consistency encoding
        emb_recon_omics1_for_cross = self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2)
        emb_latent_omics1_across_recon = self.cross_encode_omics2(emb_recon_omics1_for_cross, adj_spatial_omics2)

        emb_recon_omics2_for_cross = self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1)
        emb_latent_omics2_across_recon = self.cross_encode_omics1(emb_recon_omics2_for_cross, adj_spatial_omics1)

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'alpha_omics1': alpha_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha_mmvae,
            'mmvae_results': mmvae_results
        }

        # Add NB/ZINB parameters if needed
        if self.likelihood in ['NB', 'ZINB']:
            results['theta_omics1'] = self.theta_omics1
            results['theta_omics2'] = self.theta_omics2

            if self.likelihood == 'ZINB':
                results['pi_omics1'] = self.pi_net_omics1(emb_combined_omics1)
                results['pi_omics2'] = self.pi_net_omics2(emb_combined_omics2)

        return results


class SpaBoundPrivateReconModel(SpaBoundModel):
    """SpaBound variant whose modality decoders use shared + private latents."""

    def __init__(self, *args, **kwargs):
        kwargs['use_private_reconstruction'] = True
        super().__init__(*args, **kwargs)
