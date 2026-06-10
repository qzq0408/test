"""Boundary extraction and alignment loss for Fourier branch supervision."""

import torch
import torch.nn.functional as F


def semantic_boundary(label: torch.Tensor) -> torch.Tensor:
    """Extract binary boundary map from semantic labels [B, H, W]."""
    bd = torch.zeros_like(label, dtype=torch.float32)
    bd[:, 1:, :] = (label[:, 1:, :] != label[:, :-1, :]).float()
    bd[:, :-1, :] = torch.maximum(bd[:, :-1, :], (label[:, 1:, :] != label[:, :-1, :]).float())
    bd[:, :, 1:] = torch.maximum(bd[:, :, 1:], (label[:, :, 1:] != label[:, :, :-1]).float())
    bd[:, :, :-1] = torch.maximum(bd[:, :, :-1], (label[:, :, 1:] != label[:, :, :-1]).float())
    return bd.clamp(0.0, 1.0)


def boundary_alignment_loss(
    edge_maps_x,
    edge_maps_y,
    target: torch.Tensor,
) -> torch.Tensor:
    """Align high-frequency edge maps with semantic boundaries."""
    if not edge_maps_x:
        return torch.tensor(0.0, device=target.device, dtype=torch.float32)

    boundary = semantic_boundary(target.long())
    loss = torch.tensor(0.0, device=target.device, dtype=torch.float32)
    count = 0

    for edge_x, edge_y in zip(edge_maps_x, edge_maps_y):
        h, w = edge_x.shape[-2:]
        bd = F.interpolate(
            boundary.unsqueeze(1),
            size=(h, w),
            mode="nearest",
        ).squeeze(1)
        pred_x = edge_x.clamp(1e-6, 1.0 - 1e-6)
        pred_y = edge_y.clamp(1e-6, 1.0 - 1e-6)
        loss = loss + F.binary_cross_entropy(pred_x, bd)
        loss = loss + F.binary_cross_entropy(pred_y, bd)
        count += 2

    return loss / max(count, 1)
