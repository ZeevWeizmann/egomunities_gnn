"""  This module provides functions for computing alignment penalties between embeddings using Sinkhorn normalization."""

import torch


def sinkhorn(K, num_iters=10, eps=1e-9):
    """   Applies Sinkhorn normalization to a matrix K.
    Args:
        K (torch.Tensor): Input matrix [n, m]
        num_iters (int): Number of Sinkhorn iterations
        eps (float): Small constant to avoid division by zero
    Returns:
        torch.Tensor: Sinkhorn-normalized matrix [n, m]
    """
    for _ in range(num_iters):
        K = K / (K.sum(dim=1, keepdim=True) + eps)
        K = K / (K.sum(dim=0, keepdim=True) + eps)
    return K

def compute_penalty(z_A, z_B, beta=1.0, num_iters=10, eps=1e-9):
    """
    Computes alignment penalty between embeddings z_A and z_B using Sinkhorn-normalized transport matrix.

    Args:
        z_A (torch.Tensor): Embeddings from source graph A [n, d]
        z_B (torch.Tensor): Embeddings from target graph B [m, d]
        beta (float): Penalty scaling coefficient
        num_iters (int): Number of Sinkhorn iterations
        eps (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: Transport penalty (scalar)
    """
    D = torch.cdist(z_A, z_B, p=2)**2  # Pairwise squared distances [n, m]
    K = torch.exp(-D / (beta + 1e-6)) + 1e-8  # Unnormalized transport kernel
    K = sinkhorn(K, num_iters=num_iters, eps=eps)  # Apply Sinkhorn normalization

    penalty = torch.sum(K * D)  # Weighted cost
    return penalty, K

