"""
Controlled vocabularies (the ontology layer).

Every tag the LLM may emit is drawn from these fixed sets, so retrieval and the
Scene-Affordance Graph always resolve. Keep these small and Tier-0 focused; deferred
concepts (proxemics zones, group formations) are listed but marked DEFERRED so the
schema is forward-compatible without implying we evaluate them yet.

Grounding (cite in the paper only what the system actually uses):
  needs        — Maslow (heuristic) + Self-Determination Theory (autonomy/competence/relatedness)
  personality  — Big Five / OCEAN (runtime instance field, not asset metadata)
  relations    — Fiske Relational Models Theory
  affordances  — Gibson affordance theory + "smart objects"
  scripts      — Schank/Abelson script (frame) theory
  scenes       — Places / ADE20K style theme + functional-zone split
"""

from typing import Iterable

# ── Scenes / zones ───────────────────────────────────────────────────────────
SCENE_THEMES = (
    "cafe", "restaurant", "office", "park", "airport",
    "hospital", "school", "nightclub", "shop", "home", "street",
)
SCENE_CONTEXTS = (
    "food_service", "workplace", "education", "healthcare",
    "transport", "leisure", "retail", "domestic", "outdoor_public",
)
ZONE_FUNCTIONS = (
    "access", "circulation", "service", "seating", "waiting",
    "work", "hygiene", "activity", "staff_only", "emergency",
)

# ── Characters ───────────────────────────────────────────────────────────────
ROLES = (
    "barista", "waiter", "bartender", "business_person", "student",
    "tourist", "elderly", "parent_child", "casual_young", "staff",
    "security", "nurse", "teacher", "patient", "traveler",
)
SOCIAL_STATUS = ("regular", "vip", "staff", "guest")           # VIP is status, NOT personality
AGE_GROUPS = ("child", "teen", "adult", "senior")

# runtime "archetype" shorthand -> gross Big Five tendencies (docs; not enforced)
PERSONALITY_ARCHETYPES = (
    "reserved_professional", "sociable_extravert", "curious_explorer",
    "quiet_regular", "anxious_newcomer",
)

# ── Actions (atomic, AVA-style) ──────────────────────────────────────────────
ACTIONS = (
    "idle", "walk", "sit", "talk", "drink", "eat", "work",
    "queue", "wave", "clean", "browse", "observe", "leave",
)
VIEWS = ("front", "back", "left", "right")

# ── Affordances (what a zone/prop offers an agent) ───────────────────────────
AFFORDANCES = (
    "order", "pay", "receive", "drink", "eat", "sit", "work",
    "read", "browse", "queue", "talk_to_staff", "wait",
    "dispose", "clean", "refill", "observe", "perform", "dance",
)

# ── Props / smart objects ────────────────────────────────────────────────────
PROP_CLASSES = (
    "seating", "surface", "food_drink", "device", "signage",
    "container", "decor", "sanitation", "barrier", "service_tool",
)

# ── Needs (grouped; see needs.py) ────────────────────────────────────────────
NEED_GROUP_NAMES = (
    "physiological", "safety_comfort", "relatedness",
    "competence", "autonomy", "esteem_status",
)

# ── Social relationships (Fiske) ─────────────────────────────────────────────
RELATIONAL_MODELS = (
    "communal_sharing", "authority_ranking", "equality_matching",
    "market_pricing", "asocial",
)
RELATIONSHIP_STATUS = ("stranger", "acquaintance", "friend", "avoiding")

# ── Events / scripts ─────────────────────────────────────────────────────────
EVENT_TYPES = (
    "locomotion", "physiological", "service_transaction", "social_interaction",
    "work_task", "attention_event", "maintenance", "emergency",
    "schedule_transition", "social_status_event", "ambient",
)
PATCH_OPS = (
    "add_node", "remove_node", "update_node_state",
    "add_edge", "remove_edge",
    "activate_script", "deactivate_script",
    "enable_affordance", "disable_affordance",
    "modify_need_rate", "modify_zone_cost",
)

# ── DEFERRED (schema-reserved, not yet evaluated) ────────────────────────────
PROXEMIC_ZONES = ("intimate", "personal", "social", "public")            # DEFERRED
GROUP_FORMATIONS = (                                                       # DEFERRED
    "single", "pair_side_by_side", "pair_face_to_face", "line_queue",
    "small_circle", "loose_cluster", "v_group", "leader_follower",
)

# ── Validation helpers ───────────────────────────────────────────────────────
_ALL = {
    "scene_theme": SCENE_THEMES, "scene_context": SCENE_CONTEXTS,
    "zone_function": ZONE_FUNCTIONS, "role": ROLES, "social_status": SOCIAL_STATUS,
    "age_group": AGE_GROUPS, "action": ACTIONS, "view": VIEWS,
    "affordance": AFFORDANCES, "prop_class": PROP_CLASSES,
    "relational_model": RELATIONAL_MODELS, "event_type": EVENT_TYPES,
    "patch_op": PATCH_OPS,
}


def is_valid(dimension: str, value: str) -> bool:
    """True if `value` is in the controlled vocabulary for `dimension`."""
    return value in _ALL.get(dimension, ())


def coerce(dimension: str, value: str, default: str) -> str:
    """Return value if valid for the dimension, else default (keeps the graph clean)."""
    return value if is_valid(dimension, value) else default


def invalid_terms(dimension: str, values: Iterable[str]) -> list[str]:
    """Return the subset of values not in the controlled vocabulary (for validation logs)."""
    allowed = _ALL.get(dimension, ())
    return [v for v in values if v not in allowed]
