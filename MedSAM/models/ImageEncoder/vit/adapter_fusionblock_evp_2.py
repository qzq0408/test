import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import Adapter

class FourierMLP(nn.Module):
    """
    改进版 EVPv2 频域特征提取器:
    1. 使用 rfft2 加速并节省显存。
    2. 保持相位(Phase)不变，只对振幅(Amplitude)进行 MLP 注意力过滤，防止空间结构扭曲。
    3. 增加 norm="ortho" 保持梯度稳定。
    """
    def __init__(self, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp_amp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C] -> permute to [B, C, H, W]
        x_in = x.permute(0, 3, 1, 2).float()
        
        # 1. Real FFT2 (更高效)
        x_fft = torch.fft.rfft2(x_in, norm="ortho")
        
        # 2. 分离振幅(Amplitude)和相位(Phase)
        amp = torch.abs(x_fft)  # [B, C, H, W_half]
        phase = torch.angle(x_fft)
        
        # 3. 对振幅施加通道级注意力过滤 (还原回 channels last 送入 Linear)
        amp_permuted = amp.permute(0, 2, 3, 1) # [B, H, W_half, C]
        M_amp = self.mlp_amp(amp_permuted)
        amp_filtered = (amp_permuted * M_amp).permute(0, 3, 1, 2) # [B, C, H, W_half]
        
        # 4. 用过滤后的振幅和原始相位重建复数特征
        x_complex = amp_filtered * torch.exp(1j * phase)
        
        # 5. Inverse Real FFT2
        inv = torch.fft.irfft2(x_complex, s=(x_in.shape[2], x_in.shape[3]), norm="ortho")
        
        return inv.permute(0, 2, 3, 1).type_as(x)  # 回到 [B, H, W, C]


class FeatureTuningBranch(nn.Module):
    """保持不变：单模态降维调节"""
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
    """
    新增：逐像素空间门控融合网络
    根据输入特征动态生成每个像素上4种特征的融合权重 (H, W 维度上的归一化权重)
    """
    def __init__(self, adapter_dim: int, num_branches: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(adapter_dim * num_branches, adapter_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(adapter_dim, num_branches, kernel_size=3, padding=1),
        )
    
    def forward(self, *features):
        # features 包含 4 个张量，每个形状 [B, H, W, adapter_dim]
        # Concat along channel -> [B, H, W, 4 * adapter_dim] -> permute -> [B, 4*adapter_dim, H, W]
        x_cat = torch.cat(features, dim=-1).permute(0, 3, 1, 2)
        # weight map: [B, 4, H, W]
        w_map = self.conv(x_cat)
        # 在 4 个分支的维度（dim=1）上做 Softmax
        w_map = F.softmax(w_map, dim=1) 
        
        # 分离出各个分支的空间权重，并拓展为 [B, H, W, 1] 以便广播相乘
        w_list = [w_map[:, i:i+1, :, :].permute(0, 2, 3, 1) for i in range(len(features))]
        return w_list


class AdapterFusionBlockEVP(nn.Module):
    def __init__(
        self,
        args, dim: int, num_heads: int, mlp_ratio: float = 4.0, scale: float = 0.5,
        qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU, use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True, window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None, block_index: int = 0,
        enable_fourier: bool = True,
    ) -> None:
        super().__init__()
        self.args = args
        self.enable_fourier = enable_fourier
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, use_rel_pos, rel_pos_zero_init, 
                              input_size if window_size == 0 else (window_size, window_size))

        adapter_dim = args.mid_dim if args.mid_dim is not None else int(dim * 0.25)

        # Spatial adapters (SAM internal)
        self.Img_Adapter = Adapter(dim)
        self.DSM_Adapter = Adapter(dim)

        # 彻底移除旧版互相冲突的 MLPx_Adapter, MLPy_Adapter, wx_Adapter, wy_Adapter！

        # 高级频域与降维分支
        self.tune_x = FeatureTuningBranch(dim, adapter_dim)
        self.tune_y = FeatureTuningBranch(dim, adapter_dim)

        if enable_fourier:
            self.fourier_mlp_x = FourierMLP(dim)
            self.fourier_mlp_y = FourierMLP(dim)
            self.tune_fx = FeatureTuningBranch(dim, adapter_dim)
            self.tune_fy = FeatureTuningBranch(dim, adapter_dim)
            # 引入逐像素空间注意力取代全局标量 lambda
            self.spatial_gate_x = SpatialGate(adapter_dim, num_branches=4)
            self.spatial_gate_y = SpatialGate(adapter_dim, num_branches=4)
        else:
            self.spatial_gate_x = SpatialGate(adapter_dim, num_branches=2)
            self.spatial_gate_y = SpatialGate(adapter_dim, num_branches=2)

        self.mlp_up_x = nn.Linear(adapter_dim, dim)
        self.mlp_up_y = nn.Linear(adapter_dim, dim)
        _zero_init_linear(self.mlp_up_x)
        _zero_init_linear(self.mlp_up_y)

        self.scale = scale
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        shortcutx, shortcuty = x, y
        x, y = self.norm1(x), self.norm1(y)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hwx = window_partition(x, self.window_size)
            y, pad_hwy = window_partition(y, self.window_size)

        x, y = self.attn(x), self.attn(y)
        x, y = self.Img_Adapter(x), self.DSM_Adapter(y)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hwx, (H, W))
            y = window_unpartition(y, self.window_size, pad_hwy, (H, W))

        x, y = shortcutx + x, shortcuty + y

        # --- 高级联合 EVP 融合阶段 ---
        x_tune = self.tune_x(x)
        y_tune = self.tune_y(y)

        if self.enable_fourier:
            f_x_tune = self.tune_fx(self.fourier_mlp_x(x))
            f_y_tune = self.tune_fy(self.fourier_mlp_y(y))
            
            # 使用空间注意力门控获取逐像素权重
            w_x = self.spatial_gate_x(x_tune, y_tune, f_x_tune, f_y_tune)
            w_y = self.spatial_gate_y(x_tune, y_tune, f_x_tune, f_y_tune)
            
            P_x_base = w_x[0]*x_tune + w_x[1]*y_tune + w_x[2]*f_x_tune + w_x[3]*f_y_tune
            P_y_base = w_y[0]*x_tune + w_y[1]*y_tune + w_y[2]*f_x_tune + w_y[3]*f_y_tune
        else:
            w_x = self.spatial_gate_x(x_tune, y_tune)
            w_y = self.spatial_gate_y(x_tune, y_tune)
            P_x_base = w_x[0]*x_tune + w_x[1]*y_tune
            P_y_base = w_y[0]*x_tune + w_y[1]*y_tune

        P_x = self.mlp_up_x(P_x_base)
        P_y = self.mlp_up_y(P_y_base)

        xn, yn = self.norm2(x), self.norm2(y)

        # 简洁、无冗余的特征注入
        x = x + self.mlp(xn) + self.scale * P_x
        y = y + self.mlp(yn) + self.scale * P_y

        return x, y

# ======= 保留文件尾部原本的 Attention, window_partition, MLPBlock 等依赖类 =======
# (这里为了简洁，请把原来代码最下面那部分 Attention 及之后的通用代码贴在最后即可)