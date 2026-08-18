"""
social.py — ECGP Lightweight Social Affinity (relationships that develop at runtime).

ONE symmetric edge per meaningful agent pair holding {relationship_type, affinity 0..1,
interaction_count, persistent}. Affinity changes ONLY when a social action COMPLETES; the discrete type
is derived from affinity + count. Family / authored `persistent` ties never change automatically. The
ECGP graph reads relationship_type + affinity + same_group; it never PREDICTS a relationship change —
relationships stay deterministic runtime state. (The completion gate + per-pair cooldown live in
ecgp/teacher/social.py, layered over this core.)
"""
from dataclasses import dataclass

RELATIONSHIP_TYPES = ["stranger", "acquaintance", "friend", "family", "colleague"]
PERSISTENT_TYPES = {"family", "colleague"}       # authored/structural — not auto-relabeled

# affinity change per COMPLETED social action
INTERACTION_DELTA = {"talk": 0.05, "spend_time": 0.08, "help": 0.15, "comfort": 0.15,
                     "reject": -0.10, "argue": -0.10,
                     # a DIRECTED introduction ("Alex talk to Sam" typed by the user) carries more
                     # weight than ambient chatter — the user is deliberately building a tie, and at
                     # talk's 0.05 it took ~12 recorded chat-ticks to reach friend, invisible in a demo.
                     "directed_talk": 0.12}

# seed affinity for an AUTHORED tie (accepts a few legacy type aliases, normalised below)
_SEED_AFFINITY = {"stranger": 0.05, "acquaintance": 0.30, "friend": 0.65, "family": 0.90,
                  "colleague": 0.55, "close_friend": 0.85, "leader_follower": 0.60,
                  "staff_team": 0.60, "same_tour_group": 0.45}
# a prior interaction history so an authored friend isn't immediately demoted by the count gate
_SEED_COUNT = {"acquaintance": 3, "friend": 6, "close_friend": 8, "colleague": 8,
               "family": 10, "leader_follower": 6, "staff_team": 6, "same_tour_group": 4}
_NORMALIZE = {"close_friend": "friend", "leader_follower": "colleague",
              "staff_team": "colleague", "same_tour_group": "acquaintance"}


def _clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _key(a, b):
    return (a, b) if a <= b else (b, a)


@dataclass
class Relationship:
    agent_a: str
    agent_b: str
    relationship_type: str = "stranger"
    affinity: float = 0.0
    interaction_count: int = 0
    persistent: bool = False


def _normalize(rtype):
    return _NORMALIZE.get(rtype, rtype if rtype in RELATIONSHIP_TYPES else "acquaintance")


def make_relationship(a, b, rtype="stranger", strength=None) -> Relationship:
    """Build a seeded symmetric Relationship for an authored tie. Symmetric (keyed by sorted pair)."""
    norm = _normalize(rtype)
    aff = _clamp01(strength) if strength is not None else _SEED_AFFINITY.get(rtype, _SEED_AFFINITY.get(norm, 0.30))
    k = _key(a, b)
    return Relationship(agent_a=k[0], agent_b=k[1], relationship_type=norm, affinity=aff,
                        interaction_count=_SEED_COUNT.get(rtype, 0), persistent=norm in PERSISTENT_TYPES)


def derive_type(rel: Relationship) -> str:
    """Discrete type from affinity + interaction_count. Persistent/authored ties are left unchanged."""
    if rel.persistent:
        return rel.relationship_type
    if rel.affinity >= 0.60 and rel.interaction_count >= 5:
        rel.relationship_type = "friend"
    elif rel.affinity >= 0.25:
        rel.relationship_type = "acquaintance"
    else:
        rel.relationship_type = "stranger"
    return rel.relationship_type


def get_or_create(world, a, b) -> Relationship:
    k = _key(a, b)
    rel = world.relationships.get(k)
    if rel is None:
        rel = Relationship(agent_a=k[0], agent_b=k[1])
        world.relationships[k] = rel
    return rel


def record_interaction(world, a, b, interaction_type="talk") -> Relationship:
    """Fold one COMPLETED social action into the pair's affinity + type. Persistent ties don't change
    (only their interaction_count ticks up). Single entry point behaviour/events call."""
    if a == b:
        return None
    rel = get_or_create(world, a, b)
    rel.interaction_count += 1
    if not rel.persistent:
        rel.affinity = _clamp01(rel.affinity + INTERACTION_DELTA.get(interaction_type, 0.0))
        derive_type(rel)
    return rel


def apply_social_op(world, op: dict) -> bool:
    """Apply one social EventPatch op (relationship_set | relationship_delta). Deterministic runtime
    state; the GNN never predicts these."""
    kind = op.get("op") or op.get("kind")
    a = op.get("source_id") or op.get("agent_a") or op.get("agent")
    b = op.get("target_id") or op.get("agent_b") or op.get("target_agent")
    if not a or not b:
        return False
    if kind == "relationship_set":
        rel = make_relationship(a, b, op.get("relationship_type", "acquaintance"), op.get("affinity"))
        if "persistent" in op:
            rel.persistent = bool(op["persistent"])
        world.relationships[_key(a, b)] = rel
        return True
    if kind == "relationship_delta":
        rel = get_or_create(world, a, b)
        rel.affinity = _clamp01(rel.affinity + op.get("affinity_delta", 0.0))
        if not rel.persistent:
            derive_type(rel)
        return True
    return False
