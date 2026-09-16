"""Registro genérico de backends, para poder ampliar con módulos nuevos.

Para agregar un backend nuevo (por ejemplo un traductor, un motor TTS
distinto, o un conversor de voz tipo RVC) alcanza con:

    from dubbing_poc.registry import register

    @register("asr", "mi_backend_nuevo")
    class MiBackend(ASRBackend):
        ...

y después seleccionarlo desde la CLI con --asr-backend mi_backend_nuevo,
sin tocar el resto del pipeline.
"""

from typing import Callable, Dict, Type

_REGISTRIES: Dict[str, Dict[str, Type]] = {}


def register(kind: str, name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        _REGISTRIES.setdefault(kind, {})[name] = cls
        return cls

    return decorator


def get(kind: str, name: str, **kwargs):
    try:
        cls = _REGISTRIES[kind][name]
    except KeyError:
        available = list(_REGISTRIES.get(kind, {}).keys())
        raise ValueError(
            f"No hay backend '{name}' registrado para '{kind}'. "
            f"Disponibles: {available}"
        )
    return cls(**kwargs)


def available(kind: str):
    return list(_REGISTRIES.get(kind, {}).keys())
