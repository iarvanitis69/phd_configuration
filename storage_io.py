"""Optional caller-provided IO adapter; defaults to ordinary local files.

This shared package never imports an application backend. Applications may
register an adapter exposing open/os/np/tempfile and save_figure explicitly.
"""
import builtins
import os as _os
import tempfile as _tempfile
from contextlib import nullcontext

_PROVIDER = None


def set_provider(provider):
    global _PROVIDER
    _PROVIDER = provider
    from . import config_loader
    config_loader.set_file_opener(provider.open if provider is not None else builtins.open)


def open(*args, **kwargs):
    return (_PROVIDER.open if _PROVIDER is not None else builtins.open)(*args, **kwargs)


def process_context(label):
    hook = getattr(_PROVIDER, 'process_context', None)
    return hook(label) if hook is not None else nullcontext()


class ModuleProxy:
    def __init__(self, name, fallback):
        self.name, self.fallback = name, fallback

    def __getattr__(self, name):
        module = getattr(_PROVIDER, self.name, self.fallback) if _PROVIDER is not None else self.fallback
        return getattr(module, name)


def save_figure(figure, target, *args, **kwargs):
    if _PROVIDER is not None:
        return _PROVIDER.save_figure(figure, target, *args, **kwargs)
    return figure.savefig(target, *args, **kwargs)


os = ModuleProxy('os', _os)
tempfile = ModuleProxy('tempfile', _tempfile)
