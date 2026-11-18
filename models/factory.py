"""内置模型注册及外部 build_model 工厂的参数/输出适配。"""

import importlib
from importlib import util as importlib_util
import inspect
import os
import sys
import hashlib
import torch
from torch import nn
from .relation_decoder import CLIPDiseaseDecoder
from .baselines import FLAIRGlobalLinear, FLAIRPatchMeanLinear


class ExternalModelAdapter(nn.Module):
    """Normalize common external-model outputs to this training interface.

    Supported external forward signatures are ``forward(images)`` and
    ``forward(images, omega=...)``. The wrapped model may return either logits
    or ``(logits, auxiliary_outputs)``. This adapter avoids forcing unrelated
    baselines to implement FLAIR-specific arguments.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        forward_signature = inspect.signature(model.forward)
        self.accepts_omega = "omega" in forward_signature.parameters or any(
            (p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_signature.parameters.values())
        )

    def forward(self, images, omega=1.0):
        if self.accepts_omega:
            result = self.model(images, omega=omega)
        else:
            result = self.model(images)
        if isinstance(result, tuple):
            if len(result) == 0:
                raise ValueError("The custom model returned an empty tuple.")
            logits = result[0]
            auxiliary_outputs = result[1] if len(result) > 1 else {}
        elif isinstance(result, dict):
            if "logits" not in result:
                raise KeyError("A dictionary model output must contain 'logits'.")
            logits = result["logits"]
            auxiliary_outputs = result.get("auxiliary_outputs", {})
        else:
            logits = result
            auxiliary_outputs = {}
        if auxiliary_outputs is None:
            auxiliary_outputs = {}
        if not isinstance(auxiliary_outputs, dict):
            raise TypeError("auxiliary_outputs must be a dictionary.")
        if not torch.is_tensor(logits) or logits.ndim != 2:
            raise ValueError("The custom model must return logits with shape [B, C].")
        return (logits, auxiliary_outputs)


def build_selected_model(
    model_name,
    clip_model,
    positive_text_features,
    negative_text_features,
    relation_prior,
    num_layers,
    num_heads,
    ff_dim,
    dropout,
    class_names=None,
    model_module=None,
    model_factory="build_model",
    model_kwargs=None,
    runtime_args=None,
):
    """Build a registered model or import a user-defined model factory.

    A custom factory receives these keyword arguments and must return an
    ``nn.Module`` whose forward method accepts ``(images, omega=...)`` and
    returns ``(logits, auxiliary_outputs)``. When ``lambda_global`` is nonzero,
    ``auxiliary_outputs['global_logits']`` must have shape ``[B, C]``.
    """
    common_kwargs = {
        "clip_model": clip_model,
        "positive_text_features": positive_text_features,
        "negative_text_features": negative_text_features,
        "relation_prior": relation_prior,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "dropout": dropout,
    }
    if model_name == "relation_decoder":
        return CLIPDiseaseDecoder(
            **common_kwargs, rho=1.0, logit_eps=0.01, tau_g_init=0.07, tau_a_init=0.1
        )
    if model_name == "global_linear":
        return FLAIRGlobalLinear(
            clip_model=clip_model,
            hidden_dim=positive_text_features.shape[1],
            num_classes=positive_text_features.shape[0],
        )
    if model_name == "patch_mean":
        return FLAIRPatchMeanLinear(
            clip_model=clip_model,
            hidden_dim=positive_text_features.shape[1],
            num_classes=positive_text_features.shape[0],
        )
    if not model_module:
        raise ValueError("--model custom requires --model-module, e.g. --model-module my_models")
    try:
        if model_module.endswith(".py") or os.path.isfile(model_module):
            module_path = os.path.abspath(model_module)
            module_directory = os.path.dirname(module_path)
            if module_directory not in sys.path:
                sys.path.insert(0, module_directory)
            module_name = (
                "_fundus_external_" + hashlib.sha256(module_path.encode()).hexdigest()[:12]
            )
            spec = importlib_util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create import specification for {module_path}")
            module = importlib_util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(model_module)
    except Exception as exc:
        raise ImportError(
            f"Failed to import custom model module {model_module!r}. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(module, model_factory):
        available = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(
            f"Module {model_module!r} does not define {model_factory!r}. Available public names include: {available[:30]}"
        )
    factory = getattr(module, model_factory)
    signature = inspect.signature(factory)
    accepts_var_kwargs = any(
        (
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )
    num_classes = positive_text_features.shape[0]
    if runtime_args is not None:
        setattr(runtime_args, "num_class", num_classes)
        setattr(runtime_args, "num_classes", num_classes)
    auto_kwargs = {
        **common_kwargs,
        "args": runtime_args,
        "num_classes": num_classes,
        "num_class": num_classes,
        "n_classes": num_classes,
        "hidden_dim": positive_text_features.shape[1],
        "embed_dim": positive_text_features.shape[1],
        "feature_dim": positive_text_features.shape[1],
        "class_names": list(class_names)
        if class_names is not None
        else list(range(positive_text_features.shape[0])),
    }
    if accepts_var_kwargs:
        factory_kwargs = dict(auto_kwargs)
    else:
        factory_kwargs = {
            name: value for name, value in auto_kwargs.items() if name in signature.parameters
        }
    extra_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    factory_kwargs.update(extra_kwargs)
    try:
        model = factory(**factory_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Cannot construct custom model from {model_module!r}. Accepted parameters: {list(signature.parameters)}. Use --model-kwargs for required custom arguments."
        ) from exc
    if not isinstance(model, nn.Module):
        raise TypeError("The custom model factory must return torch.nn.Module.")
    return ExternalModelAdapter(model)
