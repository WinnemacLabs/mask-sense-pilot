"""Mask-fit feature extraction package."""

from importlib import import_module

__all__ = ["io", "preprocess", "breath", "features", "pf", "models", "viz"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"mask_fit_feat.{name}")
    raise AttributeError(name)
