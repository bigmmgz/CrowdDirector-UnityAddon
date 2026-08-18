"""
checkpoint.py — save/load ECGP checkpoints with config + metrics + version stamp.

A checkpoint bundles the model state, the model/training config, the schema/vocab/rel versions and the
metrics at save time, so evaluate.py / inference can load a fully-specified, reproducible artifact.
"""
import json
import os
import torch

from .. import schema


def save(path, model, config=None, metrics=None):
    config = config or {}
    # V2.1 pilot: a checkpoint trained with the wider V21_MODEL_CONFIG (relation_names=RELATIONS_V2,
    # option_feat_dim=OPTION_FEAT_DIM_V2) must NOT be silently stamped with the frozen v1 schema/vocab/rel
    # versions — that would be an undetectable schema/checkpoint mismatch the moment anything tries to
    # verify what this checkpoint actually is. The caller marks this explicitly via config["is_v21"];
    # every existing (v1) caller omits it and gets the exact prior stamp.
    versions = schema.stamp_v21() if config.get("is_v21") else schema.stamp()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config,
                "metrics": metrics or {}, "versions": versions}, path)
    # human-readable sidecar
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({"config": config, "metrics": metrics or {}, "versions": versions}, f, indent=2)
    return path


def load(path, model):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    return ck
