"""
inference.py — ECGP inference/demo (loads a checkpoint + vocab; never trains or generates data).

Given an EcgpWorld (effective state), encodes it and predicts, per agent, a distribution over joint
(target, type, action) options + a choice — a single real-time forward pass. `main.py` wraps this; the
optional unity_adapter can forward the chosen actions to Unity for visual execution.

Selection is a STOCHASTIC (temperature-sampled) draw from the policy's distribution, NOT a hard argmax:
in a crowd, every agent in a similar state has a near-identical distribution, so argmax makes the WHOLE
crowd deterministically pick the single top option (they all herd onto one bar/object — the collapse seen
in live logs). Sampling realises the diversity the distribution already encodes (e.g. drink 0.32 / eat 0.22
/ sit …) across the crowd, which is the correct way to roll out a stochastic policy. Set ECGP_SAMPLE=0 for
the old greedy argmax; ECGP_TEMP controls spread (1.0 = the model's own distribution, >1 flatter/more
varied, <1 sharper/greedier). Feasibility masking still applies upstream, so only valid options are drawn.
"""
import contextlib
import logging
import os
import numpy as np
import torch

from ..model import ECGPNet
from ..graph.encoder import encode
from ..graph import options as OPT
from ..training import checkpoint as ckpt

_SAMPLE = os.environ.get("ECGP_SAMPLE", "1") == "1"
_TEMP = float(os.environ.get("ECGP_TEMP", "1.0"))
log = logging.getLogger("policy.model")


class ECGPPolicy:
    def __init__(self, ckpt_path, hidden=None, layers=None, device="cpu"):
        self.device = device
        # auto-detect architecture from the checkpoint's saved config (falls back to 64x2)
        raw = torch.load(ckpt_path, map_location="cpu")
        cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
        h = hidden or cfg.get("hidden", 64)
        l = layers if layers is not None else cfg.get("layers", 2)
        # V2.1: a checkpoint trained with the wider V21_MODEL_CONFIG (relation_names=RELATIONS_V2,
        # option_feat_dim=OPTION_FEAT_DIM_V2) MUST reconstruct that same architecture here, or
        # load_state_dict either crashes on a shape mismatch or (worse) silently loads wrong — exactly the
        # "schema/checkpoint mismatch" this project treats as a hard failure. Both default to None
        # (ECGPNet's own frozen-v1 defaults) for every existing v1 checkpoint, which never set these.
        self.is_v21 = bool(cfg.get("is_v21"))
        self.model = ECGPNet(hidden=h, layers=l, relation_names=cfg.get("relation_names"),
                             option_feat_dim=cfg.get("option_feat_dim")).to(device)
        self.model.load_state_dict(raw["state_dict"] if "state_dict" in raw else raw)
        self.versions = raw.get("versions", {}) if isinstance(raw, dict) else {}
        self.hidden, self.layers = h, l
        self.ckpt_path = ckpt_path
        self.config = cfg

        # ── THE ENCODING CONTRACT — taken from the checkpoint, never from the environment ──────────
        # A model must be run under the SAME option encoding it was trained under. An env var cannot
        # express that: it is process-global, read once at import, and applies to whichever checkpoint
        # happens to load — so it corrupts BOTH directions silently (a CrowdDirect v3 net fed v2-style
        # features, or a v1/v2 net fed a feature it never saw). Neither raises; the shapes are identical.
        # crowddirect_v3/evaluation/_flagctx.py enforces exactly this per-condition for the offline
        # evaluation, and the published v3 numbers were produced with all three flags on; this is the
        # same discipline for the live runtime.
        #   directive_target_v3 — IS recorded in the config (True only for CrowdDirect v3 checkpoints).
        #   role_mask_v2 / tri_state_v2 — NOT recorded by any checkpoint written so far, but every V2.1
        #     dataset load force-enables them (ecgp/data/dataset.py) and so did the v3 training run
        #     (train_crowddirect_v3.py preflight -> _enable_v21_flags), so is_v21 is the honest
        #     inference. cfg.get() means a future retrain that DOES record them wins automatically.
        self.directive_target_v3 = bool(cfg.get("directive_target_v3", False))
        self.role_mask_v2 = bool(cfg.get("role_mask_v2", self.is_v21))
        self.tri_state_v2 = bool(cfg.get("tri_state_v2", self.is_v21))

        self.model.eval()
        self._rng = np.random.default_rng()
        log.info("[ecgp/model] ckpt=%s hidden=%d layers=%d is_v21=%s option_feat_dim=%s relations=%d "
                 "directive_target_v3=%s role_mask_v2=%s tri_state_v2=%s versions=%s",
                 os.path.basename(os.path.dirname(ckpt_path)) + "/" + os.path.basename(ckpt_path),
                 h, l, self.is_v21, getattr(self.model, "option_feat_dim", "?"),
                 len(getattr(self.model, "relation_names", []) or []),
                 self.directive_target_v3, self.role_mask_v2, self.tri_state_v2, self.versions)

    @contextlib.contextmanager
    def encoding_contract(self):
        """Apply this checkpoint's option flags for the duration of an ENCODE, then always restore them.

        Must wrap the encode, not the load: the flags are read inside options.build_options, which runs
        per call (stage1_smoke.py:61 states the same rule for the evaluation harness)."""
        prev = (OPT.DIRECTIVE_TARGET_V3, OPT.ROLE_MASK_V2, OPT.TRI_STATE_V2)
        OPT.DIRECTIVE_TARGET_V3 = self.directive_target_v3
        OPT.ROLE_MASK_V2 = self.role_mask_v2
        OPT.TRI_STATE_V2 = self.tri_state_v2
        try:
            yield
        finally:
            OPT.DIRECTIVE_TARGET_V3, OPT.ROLE_MASK_V2, OPT.TRI_STATE_V2 = prev

    def _select(self, p):
        """Pick an option index: temperature-sampled from p (default) or greedy argmax (ECGP_SAMPLE=0)."""
        if not _SAMPLE or len(p) == 1:
            return int(np.argmax(p))
        q = np.clip(p, 1e-9, None) ** (1.0 / max(_TEMP, 1e-3))
        s = q.sum()
        if not np.isfinite(s) or s <= 0:
            return int(np.argmax(p))
        return int(self._rng.choice(len(p), p=q / s))

    @torch.no_grad()
    def act(self, world):
        """Return {agent_id: {"chosen": {target_id,target_type,action}, "prob":.., "options":[...]}}."""
        with self.encoding_contract():          # options are built inside encode() — see encoding_contract
            enc = encode(world, hierarchy=self.is_v21, option_feat_v2=self.is_v21)
        out = {}
        for a, _lg, logp, _u in self.model.forward(enc, self.device):
            p = logp.exp().cpu().numpy()
            opts = a["options"]
            i = self._select(p)
            o = opts[i]
            out[a["agent_id"]] = {
                "chosen": {"target_id": o.target_id, "target_type": o.target_type, "action": o.action},
                "prob": float(p[i]),
                "options": [{"target_id": x.target_id, "target_type": x.target_type,
                             "action": x.action, "p": round(float(p[j]), 4)}
                            for j, x in enumerate(opts)],
            }
        return out
