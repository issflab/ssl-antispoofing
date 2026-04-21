from __future__ import annotations

from anti_spoofing.registries import MODEL_REGISTRY
from anti_spoofing.backends.aasist_model import Model as AASISTModel
from anti_spoofing.backends.sls_model import Model as SLSModel


@MODEL_REGISTRY.register("aasist")
def build_aasist(args, device):
    return AASISTModel(args, device)


@MODEL_REGISTRY.register("sls")
def build_sls(args, device):
    return SLSModel(args, device)


def build_model(model_name: str, args, device):
    return MODEL_REGISTRY.get(model_name)(args, device)
