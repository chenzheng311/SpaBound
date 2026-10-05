# Reorganized from spatialVAE/losses.py; see the repository license and attribution notices.
"""
Loss functions for SpaBound

This module provides various loss functions including:
- Negative Binomial (NB) likelihood for count data
- Zero-Inflated Negative Binomial (ZINB) for sparse count data
- MSE for continuous data
- KL divergence utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EdgePreservingLoss(nn.Module):
    """
    Edge-preserving smoothness loss using Total Variation / Charbonnier penalty.

    Penalizes small differences (smooth within regions) but preserves
    large differences (boundaries).
    """

    def __init__(self, loss_type='charbonnier', epsilon=0.001):
        super(EdgePreservingLoss, self).__init__()
        self.loss_type = loss_type
        self.epsilon = epsilon

    def forward(self, z, adj, edge_weights=None):
        # Get edges from sparse adjacency
        if adj.is_sparse:
            adj_coal = adj.coalesce()
            edges = adj_coal.indices()
            weights = adj_coal.values()
        else:
            edges = torch.nonzero(adj, as_tuple=False).t()
            weights = adj[edges[0], edges[1]]

        # Filter out self-loops
        mask = edges[0] != edges[1]
        edges = edges[:, mask]
        weights = weights[mask]

        # Compute differences
        z_i = z[edges[0]]
        z_j = z[edges[1]]
        diff = z_i - z_j
        diff_norm = torch.norm(diff, p=2, dim=-1)

        # Apply robust penalty
        if self.loss_type == 'l1':
            penalty = torch.abs(diff_norm)
        elif self.loss_type == 'charbonnier':
            penalty = torch.sqrt(diff_norm**2 + self.epsilon**2)
        elif self.loss_type == 'huber':
            delta = 0.1
            mask_small = diff_norm < delta
            penalty = torch.where(
                mask_small,
                0.5 * diff_norm**2 / delta,
                diff_norm - 0.5 * delta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Weight by edge weights
        if edge_weights is not None:
            penalty = penalty * edge_weights
        else:
            penalty = penalty * weights

        return penalty.sum()
class NBLoss(nn.Module):
    """
    Negative Binomial loss for count data.

    The Negative Binomial distribution is suitable for modeling over-dispersed count data,
    which is common in spatial transcriptomics.

    Parameters
    ----------
    reduction : str
        Reduction method: 'mean', 'sum', or 'none'
    eps : float
        Small constant for numerical stability

    Notes
    -----
    NB probability mass function:
    P(x | mu, theta) = Gamma(x + theta) / (Gamma(x+1) * Gamma(theta)) * (theta/(theta+mu))^theta * (mu/(theta+mu))^x

    We use the following parameterization:
    - mu: mean of the distribution (must be positive)
    - theta: inverse dispersion parameter (must be positive, larger = less overdispersion)
    """

    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, x, mu, theta):
        """
        Compute Negative Binomial negative log-likelihood.

        Parameters
        ----------
        x : torch.Tensor
            Observed count data, shape [N, G]
        mu : torch.Tensor
            Predicted mean, shape [N, G]
        theta : torch.Tensor
            Inverse dispersion parameter, shape [1] or [G] or [N, G]

        Returns
        -------
        torch.Tensor
            Negative log-likelihood
        """
        # Ensure positivity
        mu = F.softplus(mu) + self.eps
        theta = F.softplus(theta) + self.eps

        # Clip for numerical stability
        mu = torch.clamp(mu, self.eps, 1e8)
        theta = torch.clamp(theta, self.eps, 1e8)

        # Compute NB log-likelihood
        # log P(x|mu,theta) = log Gamma(x+theta) - log Gamma(x+1) - log Gamma(theta)
        #                     + theta * log(theta/(theta+mu)) + x * log(mu/(theta+mu))

        r = theta
        p = mu / (mu + r + self.eps)

        # Clip p to avoid log(0) or log(1)
        p = torch.clamp(p, self.eps, 1 - self.eps)

        # Log-gamma terms
        log_gamma_x_r = torch.lgamma(x + r + self.eps)
        log_gamma_x_1 = torch.lgamma(x + 1 + self.eps)
        log_gamma_r = torch.lgamma(r + self.eps)

        # Log probability terms
        log_p_x = x * torch.log(p + self.eps)
        log_p_r = r * torch.log(1 - p + self.eps)

        # Total log probability
        log_prob = log_gamma_x_r - log_gamma_x_1 - log_gamma_r + log_p_x + log_p_r

        # Negative log-likelihood
        if self.reduction == 'mean':
            return -log_prob.mean()
        elif self.reduction == 'sum':
            return -log_prob.sum()
        else:
            return -log_prob


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial loss for sparse count data.

    ZINB adds a dropout probability to model excess zeros,
    which is common in single-cell and spatial transcriptomics.

    Parameters
    ----------
    reduction : str
        Reduction method: 'mean', 'sum', or 'none'
    eps : float
        Small constant for numerical stability
    """

    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.nb_loss = NBLoss(reduction='none', eps=eps)

    def forward(self, x, mu, theta, pi):
        """
        Compute Zero-Inflated Negative Binomial negative log-likelihood.

        Parameters
        ----------
        x : torch.Tensor
            Observed count data, shape [N, G]
        mu : torch.Tensor
            Predicted mean, shape [N, G]
        theta : torch.Tensor
            Inverse dispersion parameter, shape [1] or [G] or [N, G]
        pi : torch.Tensor
            Dropout probability (probability of structural zero), shape [N, G]

        Returns
        -------
        torch.Tensor
            Negative log-likelihood
        """
        # Ensure pi is in valid range
        pi = torch.sigmoid(pi)
        pi = torch.clamp(pi, self.eps, 1 - self.eps)

        # Compute NB log-likelihood
        nb_log_prob = -self.nb_loss(x, mu, theta)  # This returns negative NLL, so negate

        # For zeros: P(x=0) = pi + (1-pi) * NB(0|mu,theta)
        # For non-zeros: P(x) = (1-pi) * NB(x|mu,theta)

        # NB probability at zero
        mu_safe = F.softplus(mu) + self.eps
        theta_safe = F.softplus(theta) + self.eps
        mu_safe = torch.clamp(mu_safe, self.eps, 1e8)
        theta_safe = torch.clamp(theta_safe, self.eps, 1e8)

        # P(x=0|mu,theta) = (theta/(theta+mu))^theta
        nb_zero_log_prob = theta_safe * torch.log(theta_safe / (theta_safe + mu_safe) + self.eps)

        # Combine
        zero_mask = (x < self.eps).float()

        # Log probability for zeros
        log_prob_zero = torch.log(pi + (1 - pi) * torch.exp(nb_zero_log_prob) + self.eps)

        # Log probability for non-zeros
        log_prob_nonzero = torch.log(1 - pi + self.eps) + nb_log_prob

        # Combine based on mask
        log_prob = zero_mask * log_prob_zero + (1 - zero_mask) * log_prob_nonzero

        if self.reduction == 'mean':
            return -log_prob.mean()
        elif self.reduction == 'sum':
            return -log_prob.sum()
        else:
            return -log_prob


class MSELoss(nn.Module):
    """
    Standard MSE loss wrapper for compatibility.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, mu, theta=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Target
        mu : torch.Tensor
            Prediction
        theta : ignored
            For API compatibility
        """
        if self.reduction == 'mean':
            return F.mse_loss(x, mu)
        elif self.reduction == 'sum':
            return F.mse_loss(x, mu, reduction='sum')
        else:
            return (x - mu) ** 2


class KLAnnealer:
    """
    KL divergence annealing scheduler.

    Gradually increases the KL weight from min_weight to max_weight
    over a specified number of epochs.

    Parameters
    ----------
    start_epoch : int
        Epoch to start annealing
    end_epoch : int
        Epoch to reach max_weight
    min_weight : float
        Starting weight
    max_weight : float
        Final weight
    anneal_type : str
        'linear' or 'sigmoid'
    """

    def __init__(self, start_epoch=0, end_epoch=100,
                 min_weight=0.0, max_weight=1.0, anneal_type='linear'):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.anneal_type = anneal_type

    def get_weight(self, epoch):
        """Get KL weight for current epoch."""
        if epoch < self.start_epoch:
            return self.min_weight
        elif epoch > self.end_epoch:
            return self.max_weight
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

            if self.anneal_type == 'linear':
                return self.min_weight + progress * (self.max_weight - self.min_weight)
            elif self.anneal_type == 'sigmoid':
                # Sigmoid annealing for smoother transition
                return self.min_weight + (self.max_weight - self.min_weight) * \
                       (1 / (1 + np.exp(-10 * (progress - 0.5))))
            else:
                raise ValueError(f"Unknown anneal_type: {self.anneal_type}")


class SpatialKLRegularizer(nn.Module):
    """
    Spatial structure regularizer for latent representations.

    Encourages spatial neighbors to have similar latent representations
    by penalizing differences along spatial edges.

    Parameters
    ----------
    sigma : float
        Bandwidth for the Gaussian kernel
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, z, adj_spatial):
        """
        Compute spatial smoothness loss.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation, shape [N, D]
        adj_spatial : torch.Tensor
            Spatial adjacency matrix, shape [N, N]

        Returns
        -------
        torch.Tensor
            Spatial smoothness loss
        """
        # Get edge list from adjacency matrix
        if adj_spatial.is_sparse:
            adj_dense = adj_spatial.to_dense()
        else:
            adj_dense = adj_spatial

        edge_index = adj_dense.nonzero(as_tuple=False)
        if edge_index.size(0) == 0:
            return torch.tensor(0.0, device=z.device)

        z_i = z[edge_index[:, 0]]
        z_j = z[edge_index[:, 1]]

        # Get edge weights
        edge_weights = adj_dense[edge_index[:, 0], edge_index[:, 1]]

        # Compute weighted smoothness
        diff = (z_i - z_j) ** 2
        spatial_loss = (edge_weights.unsqueeze(-1) * diff).sum() / edge_index.size(0)

        return spatial_loss / (2 * self.sigma ** 2)


class OrthogonalRegularizer(nn.Module):
    """
    Regularizer to enforce orthogonality between shared and private representations.

    Parameters
    ----------
    weight : float
        Weight for the orthogonality loss
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, z_shared, z_private):
        """
        Compute orthogonality loss.

        Parameters
        ----------
        z_shared : torch.Tensor
            Shared latent representation
        z_private : torch.Tensor
            Private latent representation

        Returns
        -------
        torch.Tensor
            Orthogonality loss
        """
        # Normalize
        z_shared_norm = F.normalize(z_shared, dim=1)
        z_private_norm = F.normalize(z_private, dim=1)

        # Compute correlation
        correlation = torch.abs((z_shared_norm * z_private_norm).sum(dim=1))

        return self.weight * correlation.mean()


def get_reconstruction_loss(loss_type='mse', **kwargs):
    """
    Factory function to create reconstruction loss.

    Parameters
    ----------
    loss_type : str
        'mse', 'nb', or 'zinb' (case-insensitive, also accepts 'Gaussian', 'NB', 'ZINB')
    **kwargs : dict
        Additional arguments for the loss function

    Returns
    -------
    nn.Module
        Loss function module
    """
    # Normalize to lowercase for comparison
    loss_type_lower = loss_type.lower()

    # Support both naming conventions
    if loss_type_lower in ['mse', 'gaussian']:
        return MSELoss(**kwargs)
    elif loss_type_lower == 'nb':
        return NBLoss(**kwargs)
    elif loss_type_lower == 'zinb':
        return ZINBLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
