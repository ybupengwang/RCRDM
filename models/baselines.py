"""冻结 FLAIR 的全局线性与区域均值线性基线。"""

import torch
from torch import nn
from torch.nn import functional as F


class _FrozenFLAIRClassifier(nn.Module):
    """Base class for lightweight FLAIR comparison baselines."""

    def __init__(self, clip_model, hidden_dim, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    @staticmethod
    def _normalize(features):
        return F.normalize(features.float(), dim=-1)


class FLAIRGlobalLinear(_FrozenFLAIRClassifier):
    """Frozen FLAIR global representation followed by a linear classifier."""

    def __init__(self, clip_model, hidden_dim, num_classes):
        super().__init__(clip_model, hidden_dim, num_classes)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images, omega=1.0):
        del omega
        with torch.no_grad():
            global_feature, patch_features = self.clip_model.encode_image_tokens(images)
        global_feature = self._normalize(global_feature)
        logits = self.classifier(global_feature)
        return (
            logits,
            {
                "global_feature": global_feature,
                "patch_features": patch_features.float(),
                "global_logits": logits,
                "presence_prob": torch.sigmoid(logits),
            },
        )


class FLAIRPatchMeanLinear(_FrozenFLAIRClassifier):
    """Frozen FLAIR patch-mean representation followed by linear heads."""

    def __init__(self, clip_model, hidden_dim, num_classes):
        super().__init__(clip_model, hidden_dim, num_classes)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.global_classifier = nn.Linear(hidden_dim, num_classes)
        for layer in (self.classifier, self.global_classifier):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images, omega=1.0):
        del omega
        with torch.no_grad():
            global_feature, patch_features = self.clip_model.encode_image_tokens(images)
        global_feature = self._normalize(global_feature)
        patch_features = self._normalize(patch_features)
        patch_mean = self._normalize(patch_features.mean(dim=1))
        logits = self.classifier(patch_mean)
        global_logits = self.global_classifier(global_feature)
        return (
            logits,
            {
                "global_feature": global_feature,
                "patch_features": patch_features,
                "patch_mean": patch_mean,
                "global_logits": global_logits,
                "presence_prob": torch.sigmoid(global_logits),
            },
        )
