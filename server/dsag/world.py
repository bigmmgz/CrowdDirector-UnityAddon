"""
Runtime instances: agents and zones (the per-simulation state, distinct from static assets).

These are lightweight; the LLM generates them at scene creation and they evolve each tick.
Big Five / relationships are optional fields here (runtime), never on the asset.
"""

from dataclasses import dataclass, field
from typing import Optional
from .needs import Needs


@dataclass
class ZoneInstance:
    id: str
    label: str
    zone_type: str
    zone_function: list                     # vocabulary.ZONE_FUNCTIONS
    affordances: list = field(default_factory=list)
    center: tuple = (0.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneInstance":
        return cls(
            id=d["id"], label=d.get("label", d["id"]),
            zone_type=d.get("zone_type", "generic"),
            zone_function=list(d.get("zone_function", [])),
            affordances=list(d.get("affordances", [])),
            center=tuple(d.get("center", (d.get("x", 0.0), d.get("y", 0.0)))),
        )


@dataclass
class AgentInstance:
    id: str
    name: str
    role: str = "casual_young"
    social_status: str = "regular"
    needs: Needs = field(default_factory=Needs)
    current_zone: str = ""
    big_five: dict = field(default_factory=dict)   # optional (runtime) — OCEAN traits
    group_id: str = None                           # social group (family/party) — drives group-cohesion feature
    busy_ticks: int = 0
    last_action: str = "idle"

    @classmethod
    def from_dict(cls, d: dict) -> "AgentInstance":
        return cls(
            id=d["id"], name=d.get("name", d["id"]),
            role=d.get("role", "casual_young"),
            social_status=d.get("social_status", "regular"),
            needs=Needs.from_dict(d.get("needs", {})),
            current_zone=d.get("current_zone", ""),
            big_five=dict(d.get("big_five", {})),
            group_id=d.get("group_id"),
        )
