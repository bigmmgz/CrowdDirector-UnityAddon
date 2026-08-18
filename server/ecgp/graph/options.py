"""
options.py — priority-bucket candidate-option generation + dedup (SCHEMA_ECGP.md rev. 3.1 §5).

An option is the joint triple (target_id, target_type, action) over REAL graph nodes (self = the agent's
own node; groups are NEVER a target — a group directive expands to goal-zone / leader / member options).
Options are built in ordered buckets: mandatory buckets are never truncated; optional fallback buckets
are truncated to MAX_OPTIONS. Everything is then deduped by (target_id, target_type, action), keeping the
highest-priority (lowest bucket) occurrence and merging event/relevance flags. Feasibility is the AND of
hard constraints (the only thing the model masks on); path/eta/attraction are soft.
"""
import os
from dataclasses import dataclass, field

from .. import vocab as V
from .. import social as cs
from .world import infer_object_role_v2

# top-need → the affordance action that relieves it (for action_compatible / urgent-object bucket)
NEED_ACTION = {"thirst": "drink", "hunger": "eat", "bladder": "relieve", "energy": "rest",
               "loneliness": "talk", "curiosity": "observe", "status": "observe",
               "stress": "rest", "groupAffinity": "talk"}

# Role-compatibility hard mask, off by default. Existing checkpoints were trained on the UNMASKED option
# set, so enabling this by default would feed live inference a different candidate distribution from the
# one the model saw in training. A training run turns it on (ECGP_ROLE_MASK_V2=1) for both its teacher
# and its own deployment, so the two stay consistent — the same convention as `hierarchy=` on encode().
#
# What it enforces: never offer sit@table or eat@chair. Role incompatibility is a hard mask, the same
# tier as affordance/capacity/hazard, not a soft preference — but only where the role is POSITIVELY
# known. An unclassified ('UNK') object falls back to its own affordance list, so a prop that matched no
# keyword is never wrongly blocked.
ROLE_MASK_V2 = os.environ.get("ECGP_ROLE_MASK_V2", "0") == "1"

# V2.1 tri-state feasibility (section 7) — OFF by default, same train/live-skew reasoning as ROLE_MASK_V2:
# the frozen checkpoint was trained with a BINARY feasible mask where a busy (over-capacity) object was
# HARD-MASKED (feasible=0), identical to a permanently-invalid one. That conflates "all seats currently
# occupied" (temporarily_unavailable — worth WAITING for, or comparing against an alternative cluster) with
# "provider disabled forever" / "role-incompatible" / "unreachable" (structurally_invalid — hard-mask,
# never worth reconsidering). With TRI_STATE_V2 on: only structural reasons hard-mask; capacity/disabled/
# hazard become a SOFT `temporarily_unavailable` + `expected_wait` feature the model can weigh instead.
TRI_STATE_V2 = os.environ.get("ECGP_TRI_STATE_V2", "0") == "1"

# V3 directive grounding (ECGP_DIRECTIVE_TARGET_V3=1) — OFF by default so every existing checkpoint and
# every frozen dataset re-encodes BYTE-IDENTICALLY.
#
# The bug it fixes: when an agent is ordered somewhere, `overlays.directive(agent_id)` holds the
# destination and the TEACHER reads it (teacher/rule_policy.py:38) — but nothing on the student's side
# ever did. Measured on a 12-agent cafe world with and without a directive
# (crowddirect/evaluation/prove_directive_invisible.py): the agent node gains one boolean ("something is
# happening to me"), an event node appears with NO edges (directives are applied as
# `apply_patch(pid, [], ttl)` — an empty operation list — so encoder.py never emits event_affects_zone /
# event_affects_agent), and every option feature is bit-identical. The network could see THAT it was
# ordered but never WHERE, so it reacted toward arbitrary targets. That is why the `directive` category
# scores 0.00 for every learned model — v1 and v2, full-graph and no-message-passing alike — while the
# teacher scores 1.00 on it.
#
# The fix reuses the EXISTING event_relevant feature rather than adding a dimension, so option_feat_dim
# is unchanged and stored teacher labels stay aligned: it sets no new option, removes none, and reorders
# none — only the value of one already-encoded feature changes, and only on options whose target IS this
# agent's ordered destination. ecgp/tests/test_directive_target_v3.py proves that invariance on frozen
# worlds. Semantically it is also right: for a directed agent, the ordered destination IS the
# event-relevant target.
DIRECTIVE_TARGET_V3 = os.environ.get("ECGP_DIRECTIVE_TARGET_V3", "0") == "1"
_ROLE_ACTIONS_V2 = {
    "sit": {"seat"}, "rest": {"seat", "surface", "cluster_root"},
    "drink": {"provider", "consumable"}, "eat": {"provider", "consumable", "cluster_root"},
    "relieve": {"sanitation"}, "work": {"workstation"},
}


def _agent_may_use(world, agent, obj) -> bool:
    """Per-agent access control: an object may declare `allowed_roles`, and then ONLY agents matching one
    of those tokens can use it (a guest-room bed only the family sleeps in, a staff-only back office).

    A token matches the agent's social_role ("staff", "security", ...) OR its group's type ("family",
    "friends", "tour_group", ...), so a scene can restrict by either without needing two fields. No
    `allowed_roles` (the default) means unrestricted, so every existing object is unaffected.

    This is a STRUCTURAL constraint like role/reachability — it never changes, so a barred agent should
    not keep the option alive hoping it clears."""
    allowed = getattr(obj, "allowed_roles", None)
    if not allowed:
        return True
    allowed = {str(r).lower() for r in allowed}
    if (getattr(agent, "social_role", "") or "").lower() in allowed:
        return True
    gid = getattr(agent, "group_id", None)
    if gid:
        grp = (getattr(world, "groups", None) or {}).get(gid)
        if grp is not None and (getattr(grp, "group_type", "") or "").lower() in allowed:
            return True
    return False


def _role_compatible(obj, action):
    if not ROLE_MASK_V2:
        return True
    allowed = _ROLE_ACTIONS_V2.get(action)
    if allowed is None:
        return True                                       # action has no role constraint (talk/observe/help…)
    role = getattr(obj, "functional_role", None) or infer_object_role_v2(getattr(obj, "object_type", ""))
    return role == "UNK" or role in allowed

# bucket caps in OPTIONS (§5.2)
CAP = {"M0": 3, "M1": 16, "M2": 4, "M3": 1, "M4": 1, "M5": 1, "M6": 8,
       "O0": 12, "O1": 4, "O2": 6, "O3": 12, "O4": 8}
MANDATORY_BUCKETS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")


@dataclass
class Option:
    target_id: str
    target_type: str          # self|zone|object|agent
    action: str
    bucket: str = "O4"
    target_global: int = -1   # filled by the encoder
    feasible: float = 0.0
    nav_reachable: float = 0.0
    path_distance: float = -1.0
    eta: float = -1.0
    action_compatible: float = 0.0
    event_relevant: float = 0.0
    capacity_ok: float = 1.0
    same_group: float = 0.0
    affinity: float = 0.0
    is_reserved: float = 0.0
    is_current: float = 0.0
    # V2.1 additions (section 4) — populated whenever ROLE_MASK_V2-relevant info is cheaply available;
    # UNUSED by the frozen v1 encoder (OPTION_STRUCT_DIM stays 9 unless a caller opts into a wider v2
    # option-feature tensor), so adding these fields cannot desync the deployed model's pointer-input shape.
    functional_role: str = "UNK"          # provider/consumable/seat/surface/fixture/workstation/sanitation/…
    expected_need_reduction: float = 0.0  # magnitude of this action's matching-need affordance effect, 0..1
    hierarchy_complete: float = 1.0       # 1.0 unless a required linked part (e.g. cluster's seat) is missing
    # Tri-state feasibility, populated only when TRI_STATE_V2 is on. Both default to 0.0
    # (feasible/available), so a caller that does not opt in sees no behavioural difference.
    temporarily_unavailable: float = 0.0  # busy/disabled/hazardous RIGHT NOW but not structurally invalid
    expected_wait: float = 0.0            # 0..1, how full/occupied the target currently is
    # The raw quantities behind `expected_wait`'s single normalized number, plus context on whether
    # waiting is worthwhile at all. Same opt-in convention and inert defaults as the two fields above.
    occupancy: float = 0.0                # current occupancy count (raw, not normalized)
    capacity: float = 1.0                 # raw capacity (>=1)
    queue_length: float = 0.0             # agents currently reserving/waiting (len(reservations))
    alternative_provider_count: float = 0.0   # OTHER reachable, structurally-valid objects offering this
                                              # same action elsewhere in the scene — "is it worth waiting
                                              # here, or is there somewhere else to go instead"
    reservation_possible: float = 0.0     # could an agent still get in line for this target at all (never
                                          # true for a structurally invalid target; a capacity=0 misconfig
                                          # is the only way a real object fails this)
    expected_availability_change: float = 0.0   # +1.0 = expect it to free up soon (busy-but-not-disabled/
                                                # hazardous, i.e. the current occupant will eventually finish);
                                                # 0.0 = no particular expectation (already free, or disabled/
                                                # hazardous with no known clear time)

    def key(self):
        return (self.target_id, self.target_type, self.action)


def _bucket_rank(b):
    order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "O0", "O1", "O2", "O3", "O4"]
    return order.index(b)


# ── feature computation for one candidate ─────────────────────────────────────
def _path_to_zone(world, agent, zone_id):
    return world.zone_path(agent.zone, zone_id)


def _event_targets(world):
    """(set of zone ids, set of object ids, set of agent ids) referenced by ACTIVE operations."""
    zt, ot, at = set(), set(), set()
    for op in world.operations.values():
        if op.event_id not in world.overlays.patches:
            continue
        if op.target_type == "zone":
            zt.add(op.target_id)
        elif op.target_type == "object":
            ot.add(op.target_id)
        elif op.target_type == "agent":
            at.add(op.target_id)
        elif op.target_type == "group":
            g = world.groups.get(op.target_id)
            if g:
                if g.leader_id:
                    at.add(g.leader_id)
                if g.separated_member:
                    at.add(g.separated_member)
                if g.group_goal:
                    zt.add(g.group_goal)
    return zt, ot, at


def _mk_option(world, agent, target_type, target_id, action, bucket, ev):
    ez, eo, ea = ev
    o = Option(target_id=target_id, target_type=target_type, action=action, bucket=bucket)

    if target_type == "self":
        o.nav_reachable, o.path_distance, o.eta = 1.0, 0.0, 0.0
        o.capacity_ok = 1.0
        if action == "continue":
            cur = agent.current_target
            o.is_current = 1.0 if cur else 0.0
            o.action_compatible = 1.0 if cur else 0.0
            o.feasible = 1.0 if _current_still_valid(world, agent) else 0.0
        else:                                   # idle / rest
            o.action_compatible = 1.0
            o.feasible = 1.0                     # always available floor
        return o

    if target_type == "zone":
        z = world.zones.get(target_id)
        if not z:
            return None
        dist, reach = _path_to_zone(world, agent, target_id)
        o.nav_reachable = 1.0 if reach else 0.0
        o.path_distance = dist
        o.eta = dist / V.NOMINAL_SPEED if reach else -1.0
        o.event_relevant = 1.0 if target_id in ez else 0.0
        if DIRECTIVE_TARGET_V3 and world.overlays.directive(agent.id) == target_id:
            o.event_relevant = 1.0          # this agent's ORDERED destination (see flag comment)
        hazard = world.overlays.is_hazard(target_id)
        if action == "leave":
            o.action_compatible = 1.0
            o.feasible = 1.0 if reach else 0.0          # leave feasible ⟺ reachable exit (§5.1)
        else:                                            # walk
            o.action_compatible = 1.0
            o.feasible = 1.0 if (reach and not hazard) else 0.0
        return o

    if target_type == "object":
        obj = world.objects.get(target_id)
        if not obj:
            return None
        dist, reach = _path_to_zone(world, agent, obj.zone_id)
        o.nav_reachable = 1.0 if reach else 0.0
        o.path_distance = dist
        o.eta = dist / V.NOMINAL_SPEED if reach else -1.0
        o.event_relevant = 1.0 if target_id in eo else 0.0
        if DIRECTIVE_TARGET_V3 and world.overlays.directive(agent.id) == target_id:
            o.event_relevant = 1.0          # a directive naming an OBJECT rather than a zone
        o.capacity_ok = 1.0 if obj.capacity_ok() else 0.0
        o.is_reserved = 1.0 if agent.reserved_object == target_id else 0.0
        affords = action in world.object_affordances(obj)
        role_ok = _role_compatible(obj, action) and _agent_may_use(world, agent, obj)
        disabled = world.overlays.is_disabled(target_id, action) or \
            world.overlays.is_disabled(obj.zone_id, action)
        hazard = world.overlays.is_hazard(obj.zone_id)
        o.action_compatible = 1.0 if (affords and role_ok) else 0.0
        if TRI_STATE_V2:
            # A permanently disabled object is structural rather than temporary: unlike a timed disable
            # it never clears on its own, so it is never worth waiting for. Distinct from `disabled`
            # above, which folds both cases together for the legacy feasibility computation.
            permanently_disabled = (world.overlays.is_permanently_disabled(target_id, action)
                                    or world.overlays.is_permanently_disabled(obj.zone_id, action))
            # STRUCTURAL (hard mask, never worth reconsidering): unreachable, wrong action, wrong role,
            # removed from the scene, or permanently disabled. TEMPORARY (soft — stays a live option,
            # just flagged): disabled right now but not permanently, hazardous right now, or at
            # capacity. All of the temporary cases can clear.
            structurally_invalid = ((not reach) or (not affords) or (not role_ok)
                                    or getattr(obj, "removed", False) or permanently_disabled)
            o.feasible = 0.0 if structurally_invalid else 1.0
            temp_unavailable = (disabled or hazard or not obj.capacity_ok()) and not structurally_invalid
            o.temporarily_unavailable = 1.0 if temp_unavailable else 0.0
            cap = max(1, int(getattr(obj, "capacity", 1) or 1))
            occ = int(getattr(obj, "occupancy", 0) or 0)
            resv = getattr(obj, "reservations", {}) or {}
            o.capacity = float(cap)
            o.occupancy = float(occ)
            o.queue_length = float(len(resv))
            if temp_unavailable:
                free = max(0, cap - occ - len(resv))
                o.expected_wait = 1.0 if (disabled or hazard) else max(0.0, min(1.0, 1.0 - free / cap))
                # busy-but-not-disabled/hazardous means the current occupant(s) will eventually finish and
                # free a slot -- a genuine, expectable IMPROVEMENT. A disabled/hazardous target has no such
                # expectation from this signal alone (it may clear, but not because someone finishes using it).
                o.expected_availability_change = 1.0 if (not disabled and not hazard) else 0.0
            # reservation is possible whenever the target isn't structurally invalid and could ever hold a
            # slot at all -- always true for a real (capacity>=1) object; the only way this is ever 0.0 for
            # a non-structurally-invalid object is a capacity=0 misconfiguration.
            o.reservation_possible = 1.0 if (not structurally_invalid and cap > 0) else 0.0
            if not structurally_invalid:
                o.alternative_provider_count = float(_count_alternative_providers(world, agent, obj, action))
        else:
            o.feasible = 1.0 if (reach and affords and role_ok and not disabled and obj.capacity_ok()
                                 and not hazard) else 0.0
        o.functional_role = getattr(obj, "functional_role", None) or infer_object_role_v2(
            getattr(obj, "object_type", ""))
        aff = next((a for a in getattr(obj, "affordances", []) if a.action == action), None)
        if aff is not None and aff.need_effects:
            # magnitude of this affordance's strongest need-effect, normalized to 0..1 (100-point need scale)
            o.expected_need_reduction = min(1.0, max(abs(v) for v in aff.need_effects.values()) / 100.0)
        return o

    if target_type == "agent":
        other = world.agents.get(target_id)
        if not other or other.id == agent.id:
            return None
        dist, reach = _path_to_zone(world, agent, other.zone)
        o.nav_reachable = 1.0 if reach else 0.0
        o.path_distance = dist
        o.eta = dist / V.NOMINAL_SPEED if reach else -1.0
        o.event_relevant = 1.0 if target_id in ea else 0.0
        o.same_group = 1.0 if (agent.group_id and agent.group_id == other.group_id) else 0.0
        rel = world.relationships.get(cs._key(agent.id, other.id))
        o.affinity = rel.affinity if rel else 0.0
        o.action_compatible = 1.0
        if action == "help":
            worthy = (max(other.needs.values()) >= 70) or o.same_group or \
                (rel is not None and rel.relationship_type in ("family", "friend"))
            o.feasible = 1.0 if (reach and worthy) else 0.0
        else:                                            # talk
            o.feasible = 1.0 if reach else 0.0
        return o
    return None


def _count_alternative_providers(world, agent, obj, action) -> int:
    """How many OTHER structurally-valid, reachable objects in the scene offer this same `action`, so the
    model can weigh queueing here against going elsewhere. A plain reachable + affords + role_ok +
    not-removed count mirroring `structurally_invalid`'s own checks, not a full tri-state re-evaluation
    of every candidate: this counts alternatives, it does not rank them."""
    n = 0
    for oid, other in world.objects.items():
        if oid == obj.id or getattr(other, "removed", False):
            continue
        if action not in world.object_affordances(other):
            continue
        if not _role_compatible(other, action):
            continue
        if not _agent_may_use(world, agent, other):
            continue          # an object this agent is barred from is not an alternative FOR THIS AGENT
        _, reach = world.zone_path(agent.zone, other.zone_id)
        if reach:
            n += 1
    return n


def _current_still_valid(world, agent):
    cur = agent.current_target
    if not cur:
        return False
    ttype, tid, _act = cur
    if ttype == "object":
        obj = world.objects.get(tid)
        return bool(obj) and obj.capacity_ok() and not world.overlays.is_hazard(obj.zone_id)
    if ttype == "zone":
        return world.zone_path(agent.zone, tid)[1] and not world.overlays.is_hazard(tid)
    if ttype == "agent":
        return tid in world.agents
    return True


# ── bucket assembly ───────────────────────────────────────────────────────────
def _object_actions(world, obj, agent):
    """Up to ACT_PER_OBJECT affordance actions, urgency-ordered."""
    affs = world.object_affordances(obj)
    urgent = [NEED_ACTION.get(n) for n in agent.urgent_needs(3)]
    ordered = [a for a in urgent if a in affs] + [a for a in affs if a not in urgent]
    seen, out = set(), []
    for a in ordered:
        if a not in seen:
            seen.add(a); out.append(a)
    return out[:V.ACT_PER_OBJECT]


def _nearest_zones(world, agent, n):
    ds = []
    for zid in world.zones:
        dist, reach = _path_to_zone(world, agent, zid)
        if reach:
            ds.append((dist, zid))
    ds.sort()
    return [z for _d, z in ds[:n]]


def _nearest_objects(world, agent, n, pred=None):
    ds = []
    for oid, obj in world.objects.items():
        if pred and not pred(obj):
            continue
        dist, reach = _path_to_zone(world, agent, obj.zone_id)
        if reach:
            ds.append((dist, oid))
    ds.sort()
    return [o for _d, o in ds[:n]]


def _group_mate_index(world):
    """agent_id -> [other agent ids sharing that agent_id's group_id] — built ONCE per world instance and
    cached on it, not scanned fresh per agent. Safe to cache: group_id is set at world-construction time
    and never mutated mid-simulation anywhere in this codebase (verified — the only per-tick mutations
    touch needs/pos/reservations/relationships, never .group_id), and a cloned/freshly-built EcgpWorld
    never inherits this attribute (clone() only copies the fields it explicitly lists), so a stale index
    can never leak across world instances. Matches EAgent.group_id equality DIRECTLY (not world.groups'
    EGroup.members list) because at least one real caller (ecgp/calibration/validate_tap.py) sets
    agent.group_id on dsag agents AFTER dsag_bridge.build_scene_model() already froze scene.social_groups —
    live_bridge.build_world copies that later group_id onto EAgent, but never registers a matching EGroup
    for it, so relying on EGroup.members here would silently drop those group-mates."""
    idx = getattr(world, "_group_mate_idx_cache", None)
    if idx is not None:
        return idx
    idx = {}
    for a in world.agents.values():
        if a.group_id:
            idx.setdefault(a.group_id, []).append(a.id)
    world._group_mate_idx_cache = idx
    return idx


def build_options(world, agent):
    """Full priority-bucket option list for one agent: built, deduped, ordered, capped."""
    ev = _event_targets(world)
    ez, eo, ea = ev
    raw = []

    def add(tt, tid, action, bucket):
        o = _mk_option(world, agent, tt, tid, action, bucket, ev)
        if o is not None:
            raw.append(o)

    # ── M0 self ──
    if agent.current_target:
        add("self", agent.id, "continue", "M0")
    add("self", agent.id, "idle", "M0")
    add("self", agent.id, "rest", "M0")

    # ── M1 event-referenced zones + objects ──
    for zid in list(ez)[:CAP["M1"]]:
        add("zone", zid, "walk", "M1")
    for oid in list(eo):
        obj = world.objects.get(oid)
        if obj:
            for act in _object_actions(world, obj, agent) or ["observe"]:
                add("object", oid, act, "M1")

    # ── M2 safety exits ──
    for zid in world.exit_zone_ids()[:CAP["M2"]]:
        add("zone", zid, "leave", "M2")

    # ── M3 current target ──
    if agent.current_target:
        tt, tid, act = agent.current_target
        if tt != "self":
            add(tt, tid, act, "M3")

    # ── M4 reserved target ──
    if agent.reserved_object and agent.reserved_object in world.objects:
        obj = world.objects[agent.reserved_object]
        acts = _object_actions(world, obj, agent) or ["observe"]
        add("object", agent.reserved_object, acts[0], "M4")

    # ── M5 group goal ──
    g = world.groups.get(agent.group_id) if agent.group_id else None
    if g and g.group_goal:
        add("zone", g.group_goal, "walk", "M5")

    # ── M6 event-referenced agents (talk/help), even without a prior tie ──
    for aid in list(ea)[:CAP["M6"]]:
        if aid != agent.id and aid in world.agents:
            add("agent", aid, "help", "M6")
            add("agent", aid, "talk", "M6")

    # ── O0 urgent-need objects ──
    urgent_acts = {NEED_ACTION.get(n) for n in agent.urgent_needs(2)}
    o0 = _nearest_objects(world, agent, 6,
                          pred=lambda o: urgent_acts & set(world.object_affordances(o)))
    for oid in o0:
        for act in _object_actions(world, world.objects[oid], agent):
            add("object", oid, act, "O0")

    # ── O1 role-relevant objects (staff) ──
    if agent.is_staff:
        for oid in _nearest_objects(world, agent, CAP["O1"],
                                    pred=lambda o: "work" in world.object_affordances(o)):
            add("object", oid, "work", "O1")

    # ── O2 socially relevant agents ──
    # Scale audit fix: this used to scan EVERY other agent per agent (O(N) per call, O(N^2) per tick across
    # the crowd) to find the sparse set of real relationships/group-mates. world.relationships and group
    # membership are both sparse/bounded by construction (a handful of ties per agent, not O(N)), so
    # iterating them directly — instead of all agents — is the same result in far less work.
    social_ids = []
    for rel in world.relationships.values():
        if rel.relationship_type == "stranger":
            continue
        other_id = (rel.agent_b if rel.agent_a == agent.id else
                   (rel.agent_a if rel.agent_b == agent.id else None))
        if other_id is not None and other_id in world.agents:
            social_ids.append(other_id)
    if agent.group_id:
        for mate_id in _group_mate_index(world).get(agent.group_id, []):
            if mate_id != agent.id and mate_id not in social_ids:
                social_ids.append(mate_id)
    for aid in social_ids[:CAP["O2"]]:
        add("agent", aid, "talk", "O2")

    # ── O3 nearest fallback zones ──
    for zid in _nearest_zones(world, agent, CAP["O3"]):
        add("zone", zid, "walk", "O3")

    # ── O4 nearest fallback objects ──
    for oid in _nearest_objects(world, agent, CAP["O4"]):
        obj = world.objects[oid]
        acts = _object_actions(world, obj, agent)
        if acts:
            add("object", oid, acts[0], "O4")

    return _dedup_order_cap(raw)


def _dedup_order_cap(raw):
    """Dedup by (target_id,target_type,action): keep highest-priority bucket, merge flags. Then order
    (bucket, path asc, target_id, action index) and cap optional to MAX_OPTIONS (§5.5)."""
    best = {}
    for o in raw:
        k = o.key()
        cur = best.get(k)
        if cur is None or _bucket_rank(o.bucket) < _bucket_rank(cur.bucket):
            if cur is not None:                          # merge metadata from the superseded copy
                o.event_relevant = max(o.event_relevant, cur.event_relevant)
                o.same_group = max(o.same_group, cur.same_group)
                o.affinity = max(o.affinity, cur.affinity)
                o.is_reserved = max(o.is_reserved, cur.is_reserved)
                o.is_current = max(o.is_current, cur.is_current)
            best[k] = o
        else:                                            # keep higher-priority, absorb this one's flags
            cur.event_relevant = max(cur.event_relevant, o.event_relevant)
            cur.same_group = max(cur.same_group, o.same_group)
            cur.affinity = max(cur.affinity, o.affinity)
            cur.is_reserved = max(cur.is_reserved, o.is_reserved)
            cur.is_current = max(cur.is_current, o.is_current)

    opts = list(best.values())

    def order_key(o):
        d = o.path_distance if o.path_distance >= 0 else 1e6
        ai = V.ACTIONS_V1.index(o.action) if o.action in V.ACTIONS_V1 else len(V.ACTIONS_V1)
        return (_bucket_rank(o.bucket), d, o.target_id, ai)
    opts.sort(key=order_key)

    mand = [o for o in opts if o.bucket in MANDATORY_BUCKETS]
    opt = [o for o in opts if o.bucket not in MANDATORY_BUCKETS]
    room = max(0, V.MAX_OPTIONS - len(mand))            # mandatory overflow → optional empty (§5.2)
    return mand + opt[:room]
