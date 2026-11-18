"""余弦区域匹配、关系加权占用及约束多头交叉注意力。"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        query: [B, C, D]
        key:   [B, N, D]
        value: [B, N, D]

        C: 疾病数量，例如24
        N: Patch数量，例如196
        D: 特征维度，例如512
        """
        attention_output, attention_weights = self.cross_attn(
            query=query, key=key, value=value, need_weights=True, average_attn_weights=False
        )
        query = self.norm1(query + self.dropout(attention_output))
        ff_output = self.ff(query)
        query = self.norm2(query + self.dropout(ff_output))
        return (query, attention_weights)


class RelationConstrainedDecoderBlock(nn.Module):
    """
    关系约束的多头交叉注意力解码器。

    该模块首先计算疾病查询与局部视觉特征之间的初始注意力，
    然后根据其他疾病的图像级存在概率和疾病无关矩阵，估计无关
    区域占用，并在softmax之前对注意力logits进行软约束。

    Shapes:
        query:              [B, C, D]
        key/value:          [B, N, D]
        unrelated_matrix:  [C, C]
        presence_prob:      [B, C]
        attention_weights: [B, H, C, N]
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        ff_dim,
        dropout=0.1,
        relation_strength_init=1.0,
        attention_temperature_init=0.1,
        eps=1e-06,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if relation_strength_init <= 0:
            raise ValueError("relation_strength_init must be positive.")
        if attention_temperature_init <= 0:
            raise ValueError("attention_temperature_init must be positive.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = float(eps)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.log_attention_temperature = nn.Parameter(
            torch.log(torch.tensor(attention_temperature_init, dtype=torch.float32))
        )
        raw_gamma_init = math.log(math.expm1(relation_strength_init))
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma_init, dtype=torch.float32))
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # 恒等 Q/K 投影保留 FLAIR 的初始语义对齐；V/O 投影正常学习。
        nn.init.eye_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.eye_(self.key_projection.weight)
        nn.init.zeros_(self.key_projection.bias)
        for layer in (self.value_projection, self.output_projection):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _split_heads(self, features):
        """[B, L, D] -> [B, H, L, D_h]."""
        batch_size, sequence_length, _ = features.shape
        features = features.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        return features.transpose(1, 2)

    def forward(self, query, key, value, unrelated_matrix, presence_prob, omega=1.0):
        batch_size, num_classes, hidden_dim = query.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected query dimension {self.hidden_dim}, but received {hidden_dim}."
            )
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes.")
        if key.ndim != 3 or key.shape[0] != batch_size:
            raise ValueError(
                "key/value must have shape [B, N, D] and share the batch size of query."
            )
        if key.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected key/value dimension {self.hidden_dim}, but received {key.shape[-1]}."
            )
        if unrelated_matrix.shape != (num_classes, num_classes):
            raise ValueError("unrelated_matrix must have shape [C, C].")
        if presence_prob.shape != (batch_size, num_classes):
            raise ValueError("presence_prob must have shape [B, C].")
        projected_query = self._split_heads(self.query_projection(query))
        projected_key = self._split_heads(self.key_projection(key))
        projected_value = self._split_heads(self.value_projection(value))
        projected_query = F.normalize(projected_query, dim=-1, eps=self.eps)
        projected_key = F.normalize(projected_key, dim=-1, eps=self.eps)
        attention_temperature = (
            self.log_attention_temperature.exp()
            .clamp(min=0.02, max=1.0)
            .to(dtype=projected_query.dtype)
        )
        initial_scores = (
            torch.matmul(projected_query, projected_key.transpose(-2, -1)) / attention_temperature
        )
        initial_attention = F.softmax(initial_scores, dim=-1)
        # A_tilde: [B, heads, classes, regions]。
        unrelated_matrix = unrelated_matrix.to(
            device=initial_scores.device, dtype=initial_scores.dtype
        )
        # 仅阻断存在概率参与区域惩罚这一路的梯度，全局辅助监督仍可反传。
        presence_prob = presence_prob.detach().to(
            device=initial_scores.device, dtype=initial_scores.dtype
        )
        relation_weights = unrelated_matrix.unsqueeze(0) * presence_prob.unsqueeze(1)
        occupancy_numerator = torch.einsum("bcj,bhjn->bhcn", relation_weights, initial_attention)
        off_diagonal_mask = 1.0 - torch.eye(
            num_classes, device=initial_scores.device, dtype=initial_scores.dtype
        )
        competing_presence = off_diagonal_mask.unsqueeze(0) * presence_prob.unsqueeze(1)
        # 分母不包含 U，避免归一化抵消“疾病无关性”的绝对强度。
        occupancy_denominator = competing_presence.sum(dim=-1)
        occupancy = occupancy_numerator / (occupancy_denominator[:, None, :, None] + self.eps)
        gamma = F.softplus(self.raw_gamma).to(dtype=initial_scores.dtype)
        omega = torch.as_tensor(
            omega, device=initial_scores.device, dtype=initial_scores.dtype
        ).clamp(0.0, 1.0)
        # 保持原论文公式：在区域 softmax 之前扣除关系加权占用。
        constrained_scores = initial_scores - omega * gamma * occupancy
        attention_weights = F.softmax(constrained_scores, dim=-1)
        aggregation_weights = self.attention_dropout(attention_weights)
        attention_output = torch.matmul(aggregation_weights, projected_value)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, num_classes, self.hidden_dim
        )
        attention_output = self.output_projection(attention_output)
        query = self.norm1(query + self.output_dropout(attention_output))
        ff_output = self.ff(query)
        query = self.norm2(query + self.ff_dropout(ff_output))
        normalized_attention_entropy = -(
            attention_weights.clamp_min(self.eps) * attention_weights.clamp_min(self.eps).log()
        ).sum(dim=-1) / math.log(attention_weights.shape[-1])
        attention_details = {
            "initial_attention": initial_attention,
            "occupancy": occupancy,
            "constraint_strength": gamma,
            "attention_temperature": attention_temperature,
            "initial_score_std": initial_scores.detach().std(dim=-1, unbiased=False).mean(),
            "initial_attention_std": initial_attention.detach().std(dim=-1, unbiased=False).mean(),
            "occupancy_spatial_std": occupancy.detach().std(dim=-1, unbiased=False).mean(),
            "final_attention_std": attention_weights.detach().std(dim=-1, unbiased=False).mean(),
            "attention_change": (attention_weights.detach() - initial_attention.detach())
            .abs()
            .mean(),
            "normalized_attention_entropy": normalized_attention_entropy.detach().mean(),
        }
        return (query, attention_weights, attention_details)
