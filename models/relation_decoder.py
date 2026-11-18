"""疾病语义查询、全局存在分支及自适应临床关系解码器。"""

from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from .attention import RelationConstrainedDecoderBlock


class CLIPDiseaseDecoder(nn.Module):
    def __init__(
        self,
        clip_model,
        positive_text_features,
        relation_prior,
        negative_text_features=None,
        num_layers=1,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        rho=1.0,
        logit_eps=0.01,
        tau_g_init=0.07,
        tau_a_init=0.1,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be at least 1.")
        self.clip_model = clip_model
        self.visual_finetune = False
        self.use_region_constraint = True
        self.visual_stage_paths = []
        self.num_classes = positive_text_features.shape[0]
        self.hidden_dim = positive_text_features.shape[1]
        self.rho = float(rho)
        self.logit_eps = float(logit_eps)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        self.register_buffer("positive_text_features", positive_text_features.float())
        if negative_text_features is not None:
            self.register_buffer("negative_text_features", negative_text_features.float())
        else:
            self.negative_text_features = None
        self.query_delta = nn.Parameter(torch.zeros(self.num_classes, self.hidden_dim))
        relation_prior = torch.as_tensor(relation_prior, dtype=torch.float32).clone()
        expected_shape = (self.num_classes, self.num_classes)
        if relation_prior.shape != expected_shape:
            raise ValueError(
                f"relation_prior must have shape {expected_shape}, but received {tuple(relation_prior.shape)}."
            )
        if not torch.isfinite(relation_prior).all():
            raise ValueError("relation_prior contains NaN or Inf.")
        if relation_prior.min() < 0 or relation_prior.max() > 1:
            raise ValueError("relation_prior must be within [0, 1].")
        if not torch.allclose(relation_prior, relation_prior.T, atol=1e-06):
            raise ValueError("relation_prior must be symmetric.")
        relation_prior.fill_diagonal_(1.0)
        self.register_buffer("R_pri", relation_prior)
        relation_prior_safe = relation_prior.clamp(min=self.logit_eps, max=1.0 - self.logit_eps)
        self.register_buffer("B_pri", torch.logit(relation_prior_safe))
        relation_anchor = relation_prior_safe.clone()
        relation_anchor.fill_diagonal_(1.0)
        self.register_buffer("R_anchor", relation_anchor)
        self.delta_R = nn.Parameter(torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer(
            "relation_diagonal_mask", torch.eye(self.num_classes, dtype=torch.bool)
        )
        if tau_g_init <= 0:
            raise ValueError("tau_g_init must be positive.")
        self.log_tau_g = nn.Parameter(torch.log(torch.tensor(tau_g_init, dtype=torch.float32)))
        self.decoder_layers = nn.ModuleList(
            [
                RelationConstrainedDecoderBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    attention_temperature_init=tau_a_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier_weight = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(self.num_classes))
        nn.init.xavier_uniform_(self.classifier_weight)

    def train(self, mode=True):
        """
        解码器正常训练；FLAIR保持eval固定BN/Dropout，选中的视觉权重仍可求梯度。
        """
        super().train(mode)
        self.clip_model.eval()
        return self

    def compute_relation_matrix(self):
        """
        根据固定先验和有界可学习残差计算：
            R: 自适应临床关系矩阵
            U: 临床无关矩阵
        """
        delta_R_sym = 0.5 * (self.delta_R + self.delta_R.T)
        R = torch.sigmoid(self.B_pri + self.rho * torch.tanh(delta_R_sym))
        R = torch.where(self.relation_diagonal_mask, torch.ones_like(R), R)
        U = 1.0 - R
        return (R, U)

    def forward(self, images, omega=1.0):
        """
        Args:
            images:
                输入眼底图像，形状为 [B, 3, H, W]。

            omega:
                关系约束的渐进激活系数，范围为 [0, 1]。

        Returns:
            logits:
                最终疾病预测，形状为 [B, C]。

            auxiliary_outputs:
                全局预测、关系矩阵及注意力图等辅助输出。
        """
        batch_size = images.shape[0]
        omega = float(max(0.0, min(1.0, omega)))
        scheduled_omega = omega
        omega = omega if self.use_region_constraint else 0.0
        grad_context = nullcontext() if self.visual_finetune else torch.no_grad()
        with grad_context:
            global_feature, patch_features = self.clip_model.encode_image_tokens(images)
        if (
            self.training
            and self.visual_finetune
            and torch.is_grad_enabled()
            and (not patch_features.requires_grad)
        ):
            raise RuntimeError(
                "patch_features has no gradient graph despite visual fine-tuning. Check FLAIRAdapter.encode_image_tokens in flair_utils.py for @torch.no_grad(), with torch.no_grad(), inference_mode(), or detach(). The selected visual stage must participate in patch feature extraction."
            )
        global_feature = F.normalize(global_feature.float(), dim=-1)
        patch_features = F.normalize(patch_features.float(), dim=-1)
        if global_feature.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                "The global visual feature and disease text feature must have the same dimension."
            )
        initial_queries = self.positive_text_features + self.query_delta
        initial_queries = F.normalize(initial_queries, dim=-1)
        tau_g = self.log_tau_g.exp().clamp(min=0.0001, max=100.0)
        global_logits = global_feature @ initial_queries.T / tau_g
        presence_prob = torch.sigmoid(global_logits)
        presence_prob_sg = presence_prob.detach()
        relation_matrix, unrelated_matrix = self.compute_relation_matrix()
        disease_queries = initial_queries.unsqueeze(0).expand(batch_size, -1, -1)
        all_attention_maps = []
        all_initial_attention_maps = []
        all_occupancy_maps = []
        all_constraint_strengths = []
        attention_diagnostics = []
        for decoder_layer in self.decoder_layers:
            disease_queries, attention_weights, attention_details = decoder_layer(
                query=disease_queries,
                key=patch_features,
                value=patch_features,
                unrelated_matrix=unrelated_matrix,
                presence_prob=presence_prob_sg,
                omega=omega,
            )
            all_attention_maps.append(attention_weights)
            all_initial_attention_maps.append(attention_details["initial_attention"])
            all_occupancy_maps.append(attention_details["occupancy"])
            all_constraint_strengths.append(attention_details["constraint_strength"])
            attention_diagnostics.append(
                {
                    "attention_temperature": attention_details["attention_temperature"],
                    "initial_score_std": attention_details["initial_score_std"],
                    "initial_attention_std": attention_details["initial_attention_std"],
                    "occupancy_spatial_std": attention_details["occupancy_spatial_std"],
                    "final_attention_std": attention_details["final_attention_std"],
                    "attention_change": attention_details["attention_change"],
                    "normalized_attention_entropy": attention_details[
                        "normalized_attention_entropy"
                    ],
                }
            )
        disease_features = disease_queries
        final_attention = all_attention_maps[-1]
        logits = torch.einsum("bcd,cd->bc", disease_features, self.classifier_weight)
        logits = logits + self.classifier_bias
        auxiliary_outputs = {
            "global_feature": global_feature,
            "patch_features": patch_features,
            "initial_queries": initial_queries,
            "disease_features": disease_features,
            "attention_maps": final_attention,
            "all_attention_maps": all_attention_maps,
            "all_initial_attention_maps": all_initial_attention_maps,
            "all_occupancy_maps": all_occupancy_maps,
            "constraint_strengths": all_constraint_strengths,
            "attention_diagnostics": attention_diagnostics,
            "global_logits": global_logits,
            "presence_prob": presence_prob,
            "relation_matrix": relation_matrix,
            "unrelated_matrix": unrelated_matrix,
            "tau_g": tau_g,
            "scheduled_omega": scheduled_omega,
            "effective_omega": omega,
        }
        return (logits, auxiliary_outputs)
