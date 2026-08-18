"""
Agent need model.

Maslow is used only as an intuitive priority heuristic (its strict hierarchy is not
strongly supported empirically); the higher tiers are grounded in Self-Determination
Theory (autonomy / competence / relatedness). The fields are exactly your existing
9-need vector plus one new SDT field (competence), so nothing downstream churns.

All needs are 0..100. For most, HIGHER = more urgent (hunger, thirst, stress...).
`energy` is inverted: LOW energy is urgent, so we score on (100 - energy).
"""

from dataclasses import dataclass, field, asdict

# group name -> canonical need fields (SDT/Maslow grouping over the existing vector)
NEED_GROUPS: dict[str, tuple[str, ...]] = {
    "physiological":  ("hunger", "thirst", "bladder", "energy"),
    "safety_comfort": ("stress",),
    "relatedness":    ("loneliness", "groupAffinity"),
    "competence":     ("taskProgress",),   # NEW (SDT) — how "productive/effective" the agent feels
    "autonomy":       ("curiosity",),
    "esteem_status":  ("status",),
}

# Maslow-style priority tiers used only to break ties in the scorer.
# 1 = physiological (block everything), 2 = safety, 3 = psychological (flat: SDT needs).
GROUP_TIER: dict[str, int] = {
    "physiological": 1, "safety_comfort": 2,
    "relatedness": 3, "competence": 3, "autonomy": 3, "esteem_status": 3,
}

# A need counts as "urgent" past these thresholds (energy handled inverted).
THRESHOLDS: dict[str, float] = {
    "hunger": 70, "thirst": 70, "bladder": 70, "energy": 30,  # energy: urgent when BELOW 30
    "stress": 60,
    "loneliness": 65, "groupAffinity": 65,
    "taskProgress": 60, "curiosity": 60, "status": 55,
}


@dataclass
class Needs:
    hunger:       float = 20.0
    thirst:       float = 20.0
    bladder:      float = 5.0
    energy:       float = 85.0
    stress:       float = 10.0
    loneliness:   float = 15.0
    groupAffinity: float = 30.0
    taskProgress: float = 30.0
    curiosity:    float = 50.0
    status:       float = 20.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Needs":
        known = {f: float(d[f]) for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)

    def pressure(self, field_name: str) -> float:
        """0..100 urgency of a single need (energy inverted so low energy = high pressure)."""
        v = getattr(self, field_name)
        return (100.0 - v) if field_name == "energy" else v

    def is_urgent(self, field_name: str) -> bool:
        v = getattr(self, field_name)
        thr = THRESHOLDS.get(field_name, 101)
        return (v < thr) if field_name == "energy" else (v > thr)


def urgent_tier(needs: Needs) -> int:
    """Lowest Maslow tier with an unmet need (1..3); 5-style default when all satisfied."""
    for tier in (1, 2, 3):
        for group, fields in NEED_GROUPS.items():
            if GROUP_TIER[group] != tier:
                continue
            if any(needs.is_urgent(f) for f in fields):
                return tier
    return 3  # nothing urgent -> free to pursue engagement/curiosity


def top_need(needs: Needs) -> tuple[str, float]:
    """The single most pressing need field and its raw pressure (0..100).

    An *urgent* lower-tier need dominates (a starving agent won't socialize), but a merely
    mild lower-tier need does not — so we add a large tier bonus only past the threshold.
    When nothing is urgent, the highest raw pressure wins (free to pursue engagement).
    """
    best_field, best_pressure, best_cmp = "curiosity", 0.0, -1.0
    for group, fields in NEED_GROUPS.items():
        tier = GROUP_TIER[group]
        for f in fields:
            p = needs.pressure(f)
            cmp = p + ((4 - tier) * 1000 if needs.is_urgent(f) else 0)
            if cmp > best_cmp:
                best_field, best_pressure, best_cmp = f, p, cmp
    return best_field, best_pressure
