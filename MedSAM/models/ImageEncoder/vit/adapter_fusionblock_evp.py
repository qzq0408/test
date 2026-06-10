import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import Adapter


class FourierBandSelector(nn.Module):
    """Explicit radial frequency-band selection with preserved phase (rFFT).

    Splits the spectrum into ``num_bands`` radial bands and learns softmax
    weights over bands. Returns filtered spatial features and a high-band
    edge map for boundary supervision.
    """

    def __init__(self, dim: int, num_bands: int = 3):
        super().__init__()
        self.num_bands = num_bands
        # Mild bias toward mid/high bands for edge-related learning
        init_logits = torch.linspace(-0.5, 1.0, num_bands)
        self.band_logits = nn.Parameter(init_logits)

    def _radial_band_masks(self, h: int, w: int, device, dtype):
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(yy ** 2 + xx ** 2)
        radius = radius / (radius.max() + 1e-8)
        edges = torch.linspace(0.0, 1.0, self.num_bands + 1, device=device, dtype=dtype)
        masks = []
        for k in range(self.num_bands):
            if k == self.num_bands - 1:
                masks.append((radius >= edges[k]).float())
            else:
                masks.append(((radius >= edges[k]) & (radius < edges[k + 1])).float())
        return torch.stack(masks, dim=0)

    def forward(self, x: torch.Tensor):
        b, h, w, _ = x.shape
        x_in = x.permute(0, 3, 1, 2).float()
        x_fft = torch.fft.rfft2(x_in, norm="ortho")
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        band_masks = self._radial_band_masks(h, w, x.device, x_in.dtype)
        weights = F.softmax(self.band_logits, dim=0)
        sel_mask = torch.einsum("k,khw->hw", weights, band_masks)

        amp_filtered = amp * sel_mask.view(1, 1, h, -1)
        x_complex = amp_filtered * torch.exp(1j * phase)
        spatial = torch.fft.irfft2(x_complex, s=(h, w), norm="ortho")
        spatial = spatial.permute(0, 2, 3, 1).type_as(x)

        high_mask = band_masks[-1].view(1, 1, h, -1)
        high_amp = amp * high_mask
        high_spatial = torch.fft.irfft2(
            high_amp * torch.exp(1j * phase), s=(h, w), norm="ortho"
        )
        edge_map = high_spatial.abs().mean(dim=1)
        edge_map = edge_map / (edge_map.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        return spatial, edge_map


class FeatureTuningBranch(nn.Module):
    """Single-modality feature tuning: down-project -> nonlinear -> residual.

    x_tune = GELU(MLP_pe(MLP_down(x))) + MLP_down(x)
    """

    def __init__(self, dim: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Linear(dim, adapter_dim)
        self.pe = nn.Linear(adapter_dim, adapter_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = self.down(x)
        return self.act(self.pe(x_down)) + x_down


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class SpatialGate(nn.Module):
    """Per-pixel fusion weights over spatial / frequency branches."""

    def __init__(self, adapter_dim: int, num_branches: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(adapter_dim * num_branches, adapter_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(adapter_dim, num_branches, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, *features):
        x_cat = torch.cat(features, dim=-1).permute(0, 3, 1, 2)
        w_map = F.softmax(self.conv(x_cat), dim=1)
        return [
            w_map[:, i : i + 1, :, :].permute(0, 2, 3, 1)
            for i in range(len(features))
        ]


class AdapterFusionBlockEVP(nn.Module):
    """Frequency-Enhanced Multi-modal Adapter (FE-MMAdapter) fusion block.

    Four-modality fusion via SpatialGate: Optical + DSM + Optical freq + DSM freq.
    Cross-modal adaptation is handled entirely by P_x / P_y (no MMAdapter).
    Shallow blocks use FourierBandSelector with optional boundary supervision.
    """

    def __init__(
        self,
        args,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        block_index: int = 0,
        enable_fourier: bool = True,
    ) -> None:
        super().__init__()
        self.args = args
        self.block_index = block_index
        self.enable_fourier = enable_fourier
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        if args.mid_dim is not None:
            adapter_dim = args.mid_dim
        else:
            adapter_dim = int(dim * 0.25)

        # Spatial adapters (frozen in training — matching original MMAdapter)
        self.Img_Adapter = Adapter(dim)
        self.DSM_Adapter = Adapter(dim)

        num_bands = getattr(args, "fourier_bands", 3)
        self.tune_x = FeatureTuningBranch(dim, adapter_dim)
        self.tune_y = FeatureTuningBranch(dim, adapter_dim)

        if enable_fourier:
            self.fourier_band_x = FourierBandSelector(dim, num_bands=num_bands)
            self.fourier_band_y = FourierBandSelector(dim, num_bands=num_bands)
            self.tune_fx = FeatureTuningBranch(dim, adapter_dim)
            self.tune_fy = FeatureTuningBranch(dim, adapter_dim)
            self.spatial_gate_x = SpatialGate(adapter_dim, num_branches=4)
            self.spatial_gate_y = SpatialGate(adapter_dim, num_branches=4)
        else:
            self.fourier_band_x = None
            self.fourier_band_y = None
            self.tune_fx = None
            self.tune_fy = None
            self.spatial_gate_x = SpatialGate(adapter_dim, num_branches=2)
            self.spatial_gate_y = SpatialGate(adapter_dim, num_branches=2)

        # Zero-init up-projection so training starts from frozen SAM behavior
        self.mlp_up_x = nn.Linear(adapter_dim, dim)
        self.mlp_up_y = nn.Linear(adapter_dim, dim)
        _zero_init_linear(self.mlp_up_x)
        _zero_init_linear(self.mlp_up_y)

        self.scale = scale
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        aux = None
        shortcutx = x
        shortcuty = y
        x = self.norm1(x)
        y = self.norm1(y)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hwx = window_partition(x, self.window_size)
            y, pad_hwy = window_partition(y, self.window_size)

        x = self.attn(x)
        y = self.attn(y)
        x = self.Img_Adapter(x)
        y = self.DSM_Adapter(y)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hwx, (H, W))
            y = window_unpartition(y, self.window_size, pad_hwy, (H, W))

        x = shortcutx + x
        y = shortcuty + y

        x_tune = self.tune_x(x)
        y_tune = self.tune_y(y)

        if self.enable_fourier:
            f_x, edge_x = self.fourier_band_x(x)
            f_y, edge_y = self.fourier_band_y(y)
            f_x_tune = self.tune_fx(f_x)
            f_y_tune = self.tune_fy(f_y)
            w_x = self.spatial_gate_x(x_tune, y_tune, f_x_tune, f_y_tune)
            w_y = self.spatial_gate_y(x_tune, y_tune, f_x_tune, f_y_tune)
            P_x_base = (
                w_x[0] * x_tune
                + w_x[1] * y_tune
                + w_x[2] * f_x_tune
                + w_x[3] * f_y_tune
            )
            P_y_base = (
                w_y[0] * x_tune
                + w_y[1] * y_tune
                + w_y[2] * f_x_tune
                + w_y[3] * f_y_tune
            )
            aux = {"edge_x": edge_x, "edge_y": edge_y}
        else:
            w_x = self.spatial_gate_x(x_tune, y_tune)
            w_y = self.spatial_gate_y(x_tune, y_tune)
            P_x_base = w_x[0] * x_tune + w_x[1] * y_tune
            P_y_base = w_y[0] * x_tune + w_y[1] * y_tune

        P_x = self.mlp_up_x(P_x_base)
        P_y = self.mlp_up_y(P_y_base)

        xn = self.norm2(x)
        yn = self.norm2(y)

        x = x + self.mlp(xn) + self.scale * P_x
        y = y + self.mlp(yn) + self.scale * P_y

        return x, y, aux


# ---------------------------------------------------------------------------
# Below are duplicated from adapter_fusionblock.py (same as original Attention,
# window helpers, MLPBlock). Kept here to make this module self-contained.
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_h, self.rel_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
