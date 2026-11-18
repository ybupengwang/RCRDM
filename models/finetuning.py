"""最后视觉阶段微调、分组学习率及首批次梯度检查。"""

import json
import math
import torch
from torch import optim
from .relation_decoder import CLIPDiseaseDecoder


def configure_flair_finetuning(model, mode, stage_path="auto", use_region_constraint=True):
    """Configure only our decoder; never silently unfreeze an external model/text encoder."""
    if mode not in {"frozen", "last_stage"}:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")
    adapter = model.clip_model
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    model.visual_finetune = mode != "frozen"
    model.use_region_constraint = bool(use_region_constraint)
    model.delta_R.requires_grad_(model.use_region_constraint)
    model.visual_stage_paths = []
    if model.visual_finetune:
        modules = dict(adapter.named_modules())
        if stage_path == "auto":
            candidates = [
                name
                for name in modules
                if name.split(".")[-1] == "layer4"
                and (not any((t in name.lower() for t in ("text", "bert", "language"))))
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Cannot uniquely identify the visual ResNet layer4. Candidates={candidates}. Set FLAIR_FINETUNE_CONFIG['stage_path'] to the exact visual stage path from clip_model.named_modules(). Available modules: {list(modules)[:100]}"
                )
            stage_path = candidates[0]
        if not stage_path or stage_path not in modules:
            raise ValueError(f"Unknown FLAIR stage path: {stage_path!r}")
        if any((t in stage_path.lower() for t in ("text", "bert", "language"))):
            raise ValueError("stage_path must select a visual stage, not the text encoder.")
        selected = list(modules[stage_path].parameters())
        if not selected:
            raise ValueError(f"Selected stage has no parameters: {stage_path}")
        for parameter in selected:
            parameter.requires_grad_(True)
        model.visual_stage_paths = [stage_path]
        cache = getattr(adapter, "_finetune_initial_parameters", {})
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if parameter.requires_grad:
                    if name not in cache:
                        cache[name] = parameter.detach().cpu().clone()
                    else:
                        parameter.copy_(cache[name].to(parameter.device))
        adapter._finetune_initial_parameters = cache
    adapter.eval()
    names = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    info = {
        "mode": mode,
        "stage_paths": model.visual_stage_paths,
        "use_region_constraint": model.use_region_constraint,
        "bn_running_statistics": "fixed",
        "text_encoder": "frozen",
        "trainable_visual_parameters": sum(
            (p.numel() for p in adapter.parameters() if p.requires_grad)
        ),
        "trainable_visual_names": names,
    }
    print("FLAIR fine-tuning:", json.dumps(info, ensure_ascii=False))
    return info


def build_finetune_optimizer(model, visual_lr=1e-05, head_lr=0.0001):
    visual = (
        [p for p in model.clip_model.parameters() if p.requires_grad]
        if isinstance(model, CLIPDiseaseDecoder)
        else []
    )
    visual_ids = {id(p) for p in visual}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in visual_ids]
    groups = []
    if head:
        groups.append({"params": head, "lr": head_lr, "name": "decoder_or_model"})
    if visual:
        groups.append({"params": visual, "lr": visual_lr, "name": "flair_visual"})
    if not groups:
        raise ValueError("The selected model has no trainable parameters.")
    return optim.Adam(groups, weight_decay=0.0001)


def check_visual_gradients(model):
    """Called after backward on the first batch of each epoch."""
    grads = [
        (name, p)
        for name, p in model.clip_model.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not grads:
        raise RuntimeError(
            "FLAIR visual parameters have no gradients. Check flair_utils.py: remove no_grad/inference_mode/detach from the trainable visual path."
        )
    total = 0.0
    candidate = None
    for name, p in grads:
        if not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite FLAIR gradient: {name}")
        norm = p.grad.detach().float().norm().item()
        total += norm**2
        if candidate is None and norm > 0:
            candidate = (name, p)
    print(f"FLAIR gradient norm: {math.sqrt(total):.6g} | tensors with gradients: {len(grads)}")
    if candidate is None:
        raise RuntimeError(
            "All selected FLAIR gradients are zero; inspect the visual feature path."
        )
    name, parameter = candidate
    return (name, parameter, parameter.detach().clone())
