from importlib import import_module
from typing import Any

MODEL_REGISTRY = {
    "3DCNN": ("training.models.cnn3d", "Tornet3DCNN"),
    "ResNet3D": ("training.models.resnet3d", "TornetResNet3D"),
    "SpatialCNN": ("training.models.spatialcnn" , "SpatialCNN_GRU"),
}


def normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def get_model(model_name: str, **kwargs: Any):
    if not model_name:
        raise ValueError("cfg.model.name is required to select a model.")

    normalized_name = normalize_model_name(model_name)

    matched = None
    for key, value in MODEL_REGISTRY.items():
        if normalize_model_name(key) == normalized_name:
            matched = value
            break

    if matched is None:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {supported}."
        )

    module_path, class_name = matched
    module = import_module(module_path)
    model_cls = getattr(module, class_name)
    return model_cls(**kwargs)
