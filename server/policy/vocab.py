"""
vocab.py — the FROZEN ordered vocabularies + derived tensor dimensions (SCHEMA_ECGP.md rev. 3.1 §VOCAB).

Every one-hot/multi-hot in the encoder indexes into these exact lists. Open vocabularies end in "UNK"
(caught by `index_of`); closed ones normalize unknowns to a named default. All node/option/pointer
dimensions are COMPUTED from the vocabularies here, and asserted by tests/test_dims.py — so a vocab
edit can never silently desync the schema's stated dims.
"""

HIDDEN = 64  # GNN hidden width (pointer-input dim depends on it)

# ── controlled vocabularies (order is frozen) ────────────────────────────────
NEEDS_V1 = ["hunger", "thirst", "bladder", "energy", "stress", "loneliness",
            "groupAffinity", "status", "curiosity"]                       # closed (9)
OCEAN_V1 = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]  # closed (5)
EMOTION_V1 = ["valence", "arousal", "panic"]                              # closed (3); stress is in NEEDS
SOCIAL_ROLES_V1 = ["visitor", "staff", "performer", "security", "guide", "tourist", "UNK"]     # (7)
ZONE_FUNCTIONS_V1 = ["access", "circulation", "service", "seating", "activity", "hygiene",
                     "work", "waiting", "staff_only", "emergency", "UNK"]                       # (11)
OBJECT_CLASS_V1 = ["seat", "surface", "dispenser", "food", "fixture", "display", "appliance",
                   "container", "UNK"]                                                          # (9)
                   # Retained for OBJECT_STATES_V1/AFFORDANCES_V1 compatibility; objects are indexed by
                   # OBJECT_ROLE_V2 below.

# Functional-role taxonomy: objects are represented by ROLE, not asset name, so "chair"/"bench"/"sofa"
# all encode identically (seat) and a prop category held out at test time still lands on a role seen in
# training. `sanitation` is separate from `fixture`, which would otherwise lump toilets in with lamps,
# plants, doors and signs.
OBJECT_ROLE_V2 = ["provider", "consumable", "seat", "surface", "fixture", "workstation",
                  "sanitation", "cluster_root", "UNK"]                                           # closed (9)
OBJECT_STATES_V1 = ["available", "clean", "dirty", "full", "empty", "UNK"]                      # (6)
AFFORDANCES_V1 = ["drink", "eat", "relieve", "sit", "rest", "work", "observe", "talk", "help", "UNK"]  # (10)
ACTIONS_V1 = ["walk", "drink", "eat", "relieve", "sit", "rest", "talk", "help", "work",
              "observe", "idle", "leave", "continue", "UNK"]              # 13 concrete + UNK = 14
GROUP_TYPES_V1 = ["family", "friends", "tour_group", "staff_team", "strangers", "UNK"]          # (6)
GROUP_ROLES_V1 = ["leader", "follower", "member", "staff", "child", "parent", "guide", "UNK"]   # (8)
RELATIONSHIP_TYPES_V1 = ["stranger", "acquaintance", "friend", "family", "colleague"]           # closed (5)
GROUP_MODES_V1 = ["independent", "follow_leader", "move_together", "regroup", "split", "help",
                  "avoid", "evacuate", "UNK"]                             # (9) — group head only
EVENT_OP_KINDS_V1 = ["zone_hazard", "zone_attraction", "block_edge", "disable_affordance",
                     "object_state", "object_attraction", "agent_directive", "group_directive",
                     "emotion_delta", "relationship_set", "relationship_delta", "event_clear"]   # closed (12)
TARGET_TYPES_V1 = ["self", "zone", "object", "agent"]                     # closed (4); NO group

# ── relation vocabulary (23, with inverses) — ecgp-rel-1.0 ───────────────────
RELATIONS_V1 = [
    "located_in", "contains_agent",
    "object_in_zone", "contains_object",
    "member_of", "has_member",
    "follows", "followed_by",
    "group_goal", "goal_of_group",
    "has_operation", "operation_of",
    "event_affects_zone", "zone_affected_by_event",
    "event_affects_object", "object_affected_by_event",
    "event_affects_agent", "agent_affected_by_event",
    "event_affects_group", "group_affected_by_event",
    "nearby", "social_relation", "zone_adjacent",
]
# inverse pairs (directed); the last three are self-inverse (bidirectional, emitted both ways)
INVERSE = {
    "located_in": "contains_agent", "object_in_zone": "contains_object",
    "member_of": "has_member", "follows": "followed_by", "group_goal": "goal_of_group",
    "has_operation": "operation_of", "event_affects_zone": "zone_affected_by_event",
    "event_affects_object": "object_affected_by_event", "event_affects_agent": "agent_affected_by_event",
    "event_affects_group": "group_affected_by_event",
}
BIDIRECTIONAL = ("nearby", "social_relation", "zone_adjacent")

# ── hierarchy relations — OBJECT-OBJECT structural relations, ADDITIVE to RELATIONS_V1 (which is left
# untouched, so a checkpoint trained on exactly 23 relations keeps working). A model opts in by indexing
# RELATIONS_V2. These expose the `parent_id` grounding carried on SmartObject (e.g. a chair bound to a
# table) to the graph as edges.
HIERARCHY_RELATIONS_V2 = [
    "part_of", "has_part",           # e.g. chair PART_OF a dining_cluster_root
    "supports", "supported_by",      # e.g. table SUPPORTS a plate
    "on", "has_on",                  # e.g. plate ON a table (a looser placement relation than supports)
    "near",                          # symmetric proximity between two OBJECTS (distinct from agent 'nearby')
    "associated_with",               # symmetric catch-all (e.g. condiment stand ASSOCIATED_WITH a counter)
    "provides", "provided_by",       # e.g. coffee machine PROVIDES the drink a cup then carries
]
RELATIONS_V2 = RELATIONS_V1 + HIERARCHY_RELATIONS_V2
INVERSE_V2 = dict(INVERSE, part_of="has_part", supports="supported_by", on="has_on", provides="provided_by")
BIDIRECTIONAL_V2 = BIDIRECTIONAL + ("near", "associated_with")
N_RELATIONS_V2 = len(RELATIONS_V2)

# ── frozen scalar constants ──────────────────────────────────────────────────
CAP_MAX = 8            # object capacity normalizer
TTL_MAX = 60           # event/operation ttl normalizer
OPS_MAX = 8            # event n_ops normalizer
EPS_SCALE = 1.0        # per-scene spatial-scale floor (§5.6)
NOMINAL_SPEED = 1.0    # eta normalizer speed
PATH_FACTOR = 1.5      # path/eta scale multiplier on S
MAX_OPTIONS = 64

# per-target action limits (§5.2)
ACT_PER_OBJECT = 2
SELF_ACTIONS = ("continue", "idle", "rest")

# ── derived dimensions (COMPUTED from the vocabularies above) ────────────────
ACTIONS_DIM = len(ACTIONS_V1)                # 14

AGENT_DIM = (len(NEEDS_V1) + len(OCEAN_V1) + len(EMOTION_V1)
             + len(SOCIAL_ROLES_V1) + len(GROUP_ROLES_V1)
             + 3            # nav/crowd: local_density, congestion, collision_risk
             + 3            # group geometry: dist_centroid, dist_leader, has_group
             + ACTIONS_DIM  # current-action one-hot
             + 2)           # flags: is_staff, under_active_event

ZONE_DIM = len(ZONE_FUNCTIONS_V1) + 5        # + [blocked, has_disabled_aff, has_blocked_edge, is_hazard, is_event_target]

OBJECT_DIM = (len(OBJECT_CLASS_V1) + len(AFFORDANCES_V1) + len(OBJECT_STATES_V1)
              + 4)          # [available, occupancy/cap, capacity/CAP_MAX, usage/10]

EVENT_DIM = 4                                  # [ttl_norm, is_evacuation, severity, n_ops_norm]
OPERATION_DIM = len(EVENT_OP_KINDS_V1) + 4     # op one-hot + [magnitude, severity, remaining_ttl_norm, active]
GROUP_DIM = len(GROUP_TYPES_V1) + 5            # type one-hot + [cohesion, follow, stress, size, has_goal]

OPTION_STRUCT_DIM = 9   # [feasible, nav_reachable, path_dist_norm, eta_norm, action_compatible,
                        #  event_relevant, capacity_ok, same_group, affinity]
OPTION_FEAT_DIM = OPTION_STRUCT_DIM + ACTIONS_DIM      # 9 + 14 = 23
POINTER_INPUT_DIM = 3 * HIDDEN + OPTION_FEAT_DIM       # 3*64 + 23 = 215

# ── hierarchy-aware option-feature dimensions — a DISTINCT, wider tensor from the v1 one above, which is
# left unchanged so existing checkpoints keep their input width. Adds three Option fields from options.py
# (functional_role one-hot over OBJECT_ROLE_V2, expected_need_reduction, hierarchy_complete). Callers opt
# in explicitly via `encode(..., option_feat_v2=True)`.
OPTION_STRUCT_DIM_V2 = OPTION_STRUCT_DIM + len(OBJECT_ROLE_V2) + 2   # +role one-hot +2 scalars
OPTION_FEAT_DIM_V2 = OPTION_STRUCT_DIM_V2 + ACTIONS_DIM
POINTER_INPUT_DIM_V2 = 3 * HIDDEN + OPTION_FEAT_DIM_V2

NODE_FEATURE_DIM = {
    "agent": AGENT_DIM, "zone": ZONE_DIM, "object": OBJECT_DIM,
    "group": GROUP_DIM, "event": EVENT_DIM, "operation": OPERATION_DIM,
}
NODE_TYPES = list(NODE_FEATURE_DIM)
N_RELATIONS = len(RELATIONS_V1)

# ── helpers ──────────────────────────────────────────────────────────────────
def index_of(value, vocab):
    """Index of `value` in `vocab`; falls back to the UNK slot if present, else -1 (closed vocab)."""
    if value in vocab:
        return vocab.index(value)
    return vocab.index("UNK") if "UNK" in vocab else -1


def onehot(value, vocab):
    v = [0.0] * len(vocab)
    i = index_of(value, vocab)
    if i >= 0:
        v[i] = 1.0
    return v


def multihot(values, vocab):
    v = [0.0] * len(vocab)
    for x in values:
        i = index_of(x, vocab)
        if i >= 0:
            v[i] = 1.0
    return v


# expected dims per the frozen schema text (rev. 3.1) — the single source of truth the tests pin to.
EXPECTED = {
    "ACTIONS_DIM": 14, "AGENT_DIM": 54, "ZONE_DIM": 16, "OBJECT_DIM": 29,
    "EVENT_DIM": 4, "OPERATION_DIM": 16, "GROUP_DIM": 11,
    "OPTION_FEAT_DIM": 23, "POINTER_INPUT_DIM": 215, "N_RELATIONS": 23,
    "TARGET_TYPES": 4,
}
