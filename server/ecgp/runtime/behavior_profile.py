"""
behavior_profile.py — Trait-conditioned Action Prior (TAP): a behavioral-profile LAYER on top of the
frozen ECGP policy.

Personality does NOT branch behavior. It reweights the GNN's own per-agent option distribution by a
product-of-experts (log-linear / energy) composition:

    pi_final(o)  ∝  pi_GNN(o) * exp( beta * <theta, phi(o)> )

  * pi_GNN(o) — the frozen policy's option probabilities (UNCHANGED; never bypassed).
  * phi(o)    — an interpretable OPTION-feature vector, derived from the option itself (social / consume /
                rest / work / explore / stay / group), NOT hardcoded per agent-type.
  * theta     — a continuous per-profile weight vector. Initialised from an OCEAN/role prior and then
                CALIBRATED to behavioral signatures (ecgp.calibration.fit_profiles) — not authored by hand.
  * beta      — global gain. beta == 0 recovers the frozen ECGP EXACTLY (clean ablation).

No GNN retraining, no teacher/dataset/vocab change: this is pure inference-time composition.
"""
from dataclasses import dataclass, field
import math

# ── interpretable option-feature vocabulary (phi dimensions) ─────────────────
FEATURES = ("social", "consume", "rest", "work", "explore", "stay", "group")

# base ECGP action -> its primary dispositional feature. Physiological actions (relieve) and structural
# ones (walk/continue/leave) carry NO dispositional feature — personality must not bias a bladder run.
_ACTION_FEATURE = {
    "talk": "social", "help": "social",
    "eat": "consume", "drink": "consume",
    "sit": "rest", "rest": "rest",
    "work": "work",
    "observe": "explore",
    "idle": "stay",
}


def option_features(o, agent, ctx) -> dict:
    """phi(o): interpretable features of ONE option, from the option + light agent context. Deterministic,
    agent-type-agnostic (the SAME map for everyone — only theta differs per profile)."""
    phi = {f: 0.0 for f in FEATURES}
    base = _ACTION_FEATURE.get(o.get("action"))
    if base:
        phi[base] = 1.0
    tt, tid = o.get("target_type"), o.get("target_id")
    cur = getattr(agent, "current_zone", "") or ""
    # moving to a NON-current zone reads as exploratory; staying/idling in the current zone reads as 'stay'
    if tt == "zone" and tid:
        phi["explore" if tid != cur else "stay"] += 0.5
    # group affinity: the option targets a groupmate, or a zone currently holding groupmates
    gm = ctx.get("group_members") or ()
    gz = ctx.get("group_zones") or ()
    if (tt == "agent" and tid in gm) or (tt == "zone" and tid in gz):
        phi["group"] += 1.0
    return phi


# ── profiles: a continuous theta per behavioral type ─────────────────────────
@dataclass
class BehaviorProfile:
    name: str
    theta: dict = field(default_factory=dict)   # feature -> weight (calibrated; missing => 0)
    group_id: str = None
    spill_rate: float = 0.0                      # sub-layer B (Stage 2, micro-events) — unused here

    def vec(self) -> dict:
        return {f: float(self.theta.get(f, 0.0)) for f in FEATURES}


# Interpretable ROLE prior for theta (a reasonable starting point BEFORE calibration). Kept deliberately
# coarse — it is NOT tuned to any evaluation scenario; calibration does the fitting on separate data.
_ROLE_PRIOR = {
    "worker":   {"work": 1.2, "stay": 0.6, "explore": -0.6, "social": -0.2},
    "staff":    {"work": 1.0, "stay": 0.2, "group": -0.3},
    "family":   {"social": 1.0, "group": 1.2, "explore": 0.2, "work": -0.6},
    "tourist":  {"explore": 1.0, "consume": 0.6, "work": -0.8, "social": 0.2},
    "socializer": {"social": 1.2, "group": 0.8, "work": -0.4},
    "loner":    {"social": -1.0, "group": -0.8, "explore": 0.3},
    "casual":   {},
}

# role keyword -> canonical profile name (staff/worker/family/tourist…), same spirit as DatasetRole.
_ROLE_KEYWORDS = (
    ("staff", "staff"), ("barista", "staff"), ("waiter", "staff"), ("bartender", "staff"),
    ("nurse", "staff"), ("security", "staff"), ("teacher", "staff"), ("clerk", "staff"),
    ("worker", "worker"), ("employee", "worker"), ("business", "worker"), ("office", "worker"),
    ("family", "family"), ("parent", "family"), ("child", "family"), ("kid", "family"),
    ("tourist", "tourist"), ("visitor", "tourist"), ("traveler", "tourist"), ("student", "tourist"),
    ("social", "socializer"), ("loner", "loner"),
)

# OCEAN trait -> per-feature deltas (trait centred at 0.5). Gives per-agent INDIVIDUATION on top of the
# profile theta, so two tourists with different openness are not clones. Fixed, psychology-grounded, untuned.
_OCEAN_MAP = {
    "extraversion":      {"social": 1.0, "group": 0.6},
    "openness":          {"explore": 1.0, "consume": 0.2},
    "conscientiousness": {"work": 0.8, "stay": 0.4, "explore": -0.3},
    "agreeableness":     {"social": 0.5, "group": 0.4},
    "neuroticism":       {"rest": 0.4, "explore": -0.3},
}
_OCEAN_GAIN = 0.6


def profile_name_for(role: str) -> str:
    r = (role or "").lower()
    for kw, name in _ROLE_KEYWORDS:
        if kw in r:
            return name
    return "casual"


def role_prior_theta(role: str) -> dict:
    """The untuned interpretable theta for a role's profile (calibration starts from this)."""
    return dict(_ROLE_PRIOR.get(profile_name_for(role), {}))


def _norm_trait(v: float) -> float:
    """OCEAN value -> [-1, 1] centred at neutral, accepting either 0..1 or 0..100 conventions."""
    v = float(v)
    return (v - 0.5) * 2.0 if v <= 1.0 else (v - 50.0) / 50.0


def agent_theta(agent, profile: BehaviorProfile) -> dict:
    """theta for THIS agent = profile theta + OCEAN individuation. Continuous, per-agent — never a switch."""
    theta = profile.vec()
    bf = getattr(agent, "big_five", None) or {}
    for trait, fmap in _OCEAN_MAP.items():
        if trait not in bf or bf[trait] is None:
            continue
        v = _norm_trait(bf[trait])
        for f, w in fmap.items():
            theta[f] += _OCEAN_GAIN * w * v
    return theta


# ── the product-of-experts reweighting ───────────────────────────────────────
def reweight(options, theta, agent, ctx, beta) -> list:
    """In-place PoE reweighting of the GNN option probs. beta == 0 is a strict no-op (returns options
    untouched — bitwise-identical downstream), guaranteeing the frozen-ECGP ablation."""
    if not beta or not options:
        return options
    for o in options:
        phi = option_features(o, agent, ctx)
        logit = sum(theta.get(f, 0.0) * phi[f] for f in FEATURES)
        o["p"] = o["p"] * math.exp(beta * logit)
    z = sum(o["p"] for o in options)
    if z > 0:
        for o in options:
            o["p"] /= z
    return options


def dominant_feature(o, agent, ctx) -> str:
    """The single feature an option most expresses (for measuring behavioral signatures). '' if none."""
    phi = option_features(o, agent, ctx)
    f = max(FEATURES, key=lambda k: phi[k])
    return f if phi[f] > 0 else ""
