"""
world.py — the typed EFFECTIVE-STATE world the ECGP encoder / options / rollout-teacher operate on.

Self-contained; reuses two shared pieces:
  - `dsag.smart_object.SmartObject` for objects (carrying capacity/reservations, §9);
  - `ecgp.social.Relationship` for the Lightweight Social Affinity edges (read-only here; ECGP's
    completion-gated updates live in ecgp/social.py).
Holds an `OverlaySet` so every query the encoder makes is over effective (base ⊕ overlay) state.
Navigation is a small zone-adjacency graph; `block_edge` overlays remove adjacencies at query time.
"""
from dataclasses import dataclass, field
import copy
import heapq
import math

from .. import vocab as V

# object_type / prop_id keyword → OBJECT_CLASS_V1
_CLASS_KEYWORDS = [
    ("seat", ("chair", "stool", "seat", "bench", "sofa", "couch")),
    ("surface", ("table", "desk", "counter", "bar")),
    ("dispenser", ("fountain", "machine", "dispenser", "cooler", "fridge", "tap", "sink", "urn", "station")),
    ("food", ("plate", "food", "dish", "tray", "meal")),
    ("appliance", ("stove", "oven", "computer", "monitor", "tv", "screen", "register")),
    ("display", ("shelf", "display", "rack", "art", "painting", "exhibit", "board")),
    ("container", ("bin", "crate", "box", "basket", "cart")),
    ("fixture", ("toilet", "urinal", "lamp", "plant", "door", "sign")),
]


def infer_object_class(object_type: str) -> str:
    t = (object_type or "").lower()
    for cls, keys in _CLASS_KEYWORDS:
        if any(k in t for k in keys):
            return cls
    return "UNK"


# ── functional-role taxonomy (OBJECT_ROLE_V2) — additive; `infer_object_class` above is unchanged, so a
# checkpoint trained on OBJECT_CLASS_V1 keeps working. Order matters (first match wins), the same
# convention as _CLASS_KEYWORDS. `sanitation` is a separate role from `fixture`, which would otherwise
# lump toilets in with lamps, plants, doors and signs; a held-out sanitation prop such as a "restroom
# stall" then shares an embedding neighbourhood with training toilets rather than with decor.
#
# The keyword table is grounded against the real asset catalog Unity resolves at runtime (44 objects
# across cafe, living_room, bathroom, bedroom, gym, hospital, office, museum and shop), not written in
# the abstract; running the classifier over all 44 names surfaced 11 that fell through to UNK (sink, tv,
# shower, bath, bed, wardrobe, treadmill, weights, medicine, printer, cart).
#
# Three of those do not map cleanly onto the eight roles and are approximations rather than confident
# calls: bed -> seat (its affordance normalizes toward rest, which seat already permits), wardrobe ->
# surface (a browse-a-container piece, the same family as shelf and bookshelf), and treadmill/weights ->
# workstation (an actively operated station, the closest available fit).
_ROLE_KEYWORDS_V2 = [
    ("sanitation", ("toilet", "urinal", "restroom", "washroom", "lavatory", "sink", "shower", "bath",
                    "bathtub")),
    ("workstation", ("desk", "computer", "monitor", "register", "till", "workbench", "screen", "printer",
                     "treadmill", "weights", "exercise")),
    ("provider", ("fountain", "machine", "dispenser", "cooler", "fridge", "tap", "urn", "station",
                  "stove", "oven", "counter", "bar", "stall")),   # NOT bare "coffee" -- matched
                  # "coffee_table"/"coffee_cup" before they reached surface/consumable (found by sanity check)
    ("consumable", ("plate", "food", "dish", "tray", "meal", "glass", "cup", "mug", "bottle",
                    "pastry", "snack", "drink", "glassware", "utensil", "medicine")),
    ("seat", ("chair", "stool", "seat", "bench", "sofa", "couch", "bed")),
    ("surface", ("table", "shelf", "island", "bookcase", "bookshelf", "wardrobe")),
    ("fixture", ("lamp", "plant", "door", "sign", "display", "rack", "art", "painting", "exhibit",
                 "board", "bin", "crate", "box", "basket", "cart", "tv", "television")),
]


def infer_object_role_v2(object_type: str) -> str:
    """Functional role (provider/consumable/seat/surface/fixture/workstation/sanitation) by object_type
    keyword. `cluster_root` is never inferred here -- it is a SYNTHESIZED role assigned only by
    `dsag_bridge.build_dining_clusters` (no raw asset is intrinsically a cluster)."""
    t = (object_type or "").lower()
    for role, keys in _ROLE_KEYWORDS_V2:
        if any(k in t for k in keys):
            return role
    return "UNK"


@dataclass
class EAgent:
    id: str
    name: str = ""
    social_role: str = "visitor"
    group_role: str = "member"
    pos: tuple = (0.0, 0.0)
    zone: str = ""
    needs: dict = field(default_factory=lambda: {n: 30.0 for n in V.NEEDS_V1})   # 0..100
    ocean: dict = field(default_factory=lambda: {o: 0.5 for o in V.OCEAN_V1})     # 0..1
    valence: float = 0.0     # -1..1
    arousal: float = 0.0     # 0..1
    panic: float = 0.0       # 0..100
    group_id: str = None
    last_action: str = "idle"
    current_target: tuple = None      # (target_type, target_id, action) — the ongoing action, or None
    reserved_object: str = None       # object_id the agent holds a reservation on
    is_staff: bool = False
    # ── complete-sim runtime state (SCHEMA §7 full transition model) ──
    left: bool = False                # has left the floor plan (evacuated / went home) → disappeared
    heading: str = None               # zone id the agent is currently walking toward
    action_progress: int = 0          # ticks remaining to COMPLETE the arrived action (0 = not acting)
    resv_token: str = None            # active reservation token on `reserved_object`
    # transient per-tick reward signals set by the stepper (read by the teacher's reward fn)
    r_relief: float = 0.0
    r_failed: bool = False
    r_delayed: bool = False
    r_social: float = 0.0
    r_left_now: bool = False
    r_completed: bool = False
    chain_state: object = None        # Gate 2: ChainState (ecgp.graph.chain_state) for an active macro-chain

    def urgent_needs(self, k=2):
        from ..teacher.need_pressure import top_pressure_needs   # rank by PRESSURE, not raw value
        return top_pressure_needs(self.needs, k)

    @property
    def expressiveness(self):
        return 0.25 + 0.75 * self.ocean.get("extraversion", 0.5)

    @property
    def susceptibility(self):
        return 0.20 + 0.6 * self.ocean.get("neuroticism", 0.5) + 0.2 * self.ocean.get("agreeableness", 0.5)

    @property
    def composure(self):
        return 0.5 * self.ocean.get("conscientiousness", 0.5) + 0.5 * (1 - self.ocean.get("neuroticism", 0.5))


@dataclass
class EZone:
    id: str
    zone_function: str = "activity"
    center: tuple = (0.0, 0.0)
    is_exit: bool = False
    adjacency: list = field(default_factory=list)      # neighbouring zone ids (base nav graph)


@dataclass
class EGroup:
    id: str
    group_type: str = "strangers"
    leader_id: str = None
    members: list = field(default_factory=list)
    cohesion: float = 0.5
    follow_strength: float = 0.5
    group_goal: str = None            # zone id
    group_stress: float = 0.0
    separated_member: str = None      # a member currently detached (for 'separated family')


@dataclass
class EEvent:
    id: str
    is_evacuation: bool = False
    ttl: int = 25
    severity: float = 0.5
    op_ids: list = field(default_factory=list)


@dataclass
class EOperation:
    id: str
    event_id: str
    op_kind: str
    magnitude: float = 0.0
    severity: float = 0.0
    remaining_ttl: int = 25
    active: bool = True
    target_type: str = None           # zone|object|agent|group
    target_id: str = None
    target_id2: str = None            # second zone for block_edge (the edge (target_id, target_id2))
    action: str = None                # for disable_affordance / object_state payloads


_EMPTY_NAV_SIG = frozenset()          # shared no-blocked-edges signature (the common case; avoids allocation)


class EcgpWorld:
    def __init__(self):
        from .overlays import OverlaySet
        self.agents: dict = {}
        self.zones: dict = {}
        self.objects: dict = {}        # id -> dsag SmartObject
        self.groups: dict = {}
        self.events: dict = {}
        self.operations: dict = {}
        self.overlays = OverlaySet()
        self.relationships: dict = {}          # (a,b sorted) -> ecgp.social.Relationship
        self.social_cooldowns: dict = {}       # (a,b sorted) -> last completed-interaction tick
        self.tick_no = 0
        self.scenario_id = "scene"
        self.trajectory_id = "scene__seed0"

    # ── branchable state (SCHEMA §7 item 2): exact deep clone + signature ─────
    def clone(self) -> "EcgpWorld":
        """Independent branch clone of the COMPLETE mutable simulation state, so evaluating one rollout option
        cannot contaminate another's branch. FAST path: the stepper never mutates the nav graph or the event
        definitions (`block_edge`/ttl live in `overlays`, not in zones/events/operations), so those are SHARED
        by reference; only the state the stepper actually writes — agents' needs/pos/action/emotion, object
        state+reservations, groups, overlays, relationships — is copied. ~5-6× faster than a blanket deepcopy
        at IDENTICAL isolation (verified: byte-identical teacher labels vs deepcopy). Falls back to deepcopy on
        any unexpected structure so correctness never depends on the fast path."""
        try:
            from dataclasses import replace
            w = EcgpWorld.__new__(EcgpWorld)
            w.zones = self.zones                 # nav graph — read-only during a rollout → SHARE
            w.events = self.events               # event defs — read-only during a rollout → SHARE
            w.operations = self.operations       # operation defs — read-only during a rollout → SHARE
            w.scenario_id = self.scenario_id
            w.trajectory_id = self.trajectory_id
            w.tick_no = self.tick_no
            # Copy everything the stepper mutates: needs are mutated in place, so they need a fresh dict,
            # and ocean likewise. chain_state is a mutable object the stepper advances in place, and
            # replace() alone would share one instance across branches — exactly the cross-branch
            # contamination this method exists to prevent — so it gets an explicit deep copy.
            w.agents = {k: replace(a, needs=dict(a.needs), ocean=dict(a.ocean),
                                   chain_state=copy.deepcopy(a.chain_state) if a.chain_state is not None else None)
                        for k, a in self.agents.items()}
            w.objects = {k: copy.deepcopy(o) for k, o in self.objects.items()}
            w.groups = {k: copy.deepcopy(g) for k, g in self.groups.items()}
            w.overlays = copy.deepcopy(self.overlays)
            w.relationships = {k: copy.deepcopy(r) for k, r in self.relationships.items()}
            w.social_cooldowns = dict(self.social_cooldowns)
            return w
        except Exception:
            return copy.deepcopy(self)           # safety net: never let a speedup compromise correctness

    def signature(self) -> tuple:
        """A hashable snapshot of the mutable state used to prove clone/branch isolation in tests."""
        ag = tuple(sorted(
            (a.id, round(a.pos[0], 4), round(a.pos[1], 4), a.zone, a.left, a.heading,
             a.action_progress, a.resv_token, a.current_target,
             tuple(round(a.needs[n], 4) for n in sorted(a.needs)),
             round(a.panic, 4), round(a.stress if False else a.needs.get("stress", 0.0), 4),
             # chain_state is mutable, stepper-advanced state, so it has to be part of the isolation
             # fingerprint; otherwise a clone() that shared one instance across branches would go
             # undetected by this very check.
             None if a.chain_state is None else
             (a.chain_state.chain_id, a.chain_state.stage.value, a.chain_state.seat_id,
              a.chain_state.stage_started_tick))
            for a in self.agents.values()))
        obj = tuple(sorted((o.id, o.state, o.occupancy, tuple(sorted(o.reservations)))
                           for o in self.objects.values()))
        ov = (tuple(sorted(self.overlays.patches.items(), key=lambda kv: kv[0])),
              # FOUND+FIXED (flaky test_branch.py, ~10-20% of runs): str(frozenset) is NOT a stable key —
              # two value-EQUAL frozensets can repr in a different element order depending on construction
              # history (insertion order affects the internal hash-table layout even under one fixed
              # PYTHONHASHSEED), so a clone's blocked-edge frozenset could stringify differently from the
              # original's despite being the identical edge. tuple(sorted(k)) is order-independent and
              # genuinely stable for a 2-element string frozenset.
              tuple(sorted((tuple(sorted(k)), tuple(sorted(self.overlays.owners("blocked", k))))
                           for k in self.overlays._blocked)),
              tuple(sorted((z, tuple(sorted(self.overlays.owners("hazard", z))))
                           for z in self.overlays._hazard)))
        return (self.tick_no, ag, obj, ov)

    # ── construction ─────────────────────────────────────────────────────────
    def add_agent(self, a): self.agents[a.id] = a
    def add_zone(self, z): self.zones[z.id] = z
    def add_object(self, o): self.objects[o.id] = o
    def add_group(self, g):
        self.groups[g.id] = g
        for m in g.members:
            if m in self.agents:
                self.agents[m].group_id = g.id

    def add_event(self, ev, ops):
        """Register an event + its operations and push overlays onto the OverlaySet (with provenance)."""
        self.events[ev.id] = ev
        payload = []
        for op in ops:
            self.operations[op.id] = op
            ev.op_ids.append(op.id)
            payload.append(self._op_to_overlay(op))
        self.overlays.apply_patch(ev.id, payload, ttl=ev.ttl, evac=ev.is_evacuation)

    @staticmethod
    def _op_to_overlay(op: EOperation) -> dict:
        d = {"op": op.op_kind, "severity": op.severity, "delta": op.magnitude}
        if op.target_type == "zone":
            d["zone"] = op.target_id
        elif op.target_type == "object":
            d["object"] = op.target_id
        elif op.target_type == "agent":
            d["agent"] = op.target_id
        elif op.target_type == "group":
            d["group"] = op.target_id
        if op.target_id2 is not None:      # block_edge: second endpoint
            d["zone_b"] = op.target_id2
        if op.action is not None:          # disable_affordance / object_state payload
            d["action"] = op.action
            d["state"] = op.action if op.op_kind == "object_state" else d.get("state")
        return d

    # ── scene scale (§5.6) ─────────────────────────────────────────────────────
    def scene_scale(self) -> float:
        xs = [z.center[0] for z in self.zones.values()]
        ys = [z.center[1] for z in self.zones.values()]
        if not xs:
            return V.EPS_SCALE
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return max(diag, V.EPS_SCALE)

    def exit_zone_ids(self) -> list:
        ex = [z.id for z in self.zones.values() if z.is_exit or z.zone_function == "access"]
        return ex or ([next(iter(self.zones))] if self.zones else [])

    # ── navigation over EFFECTIVE state (block_edge overlays remove adjacencies) ──
    def _neighbors(self, zid):
        z = self.zones.get(zid)
        if not z:
            return []
        return [n for n in z.adjacency
                if n in self.zones and not self.overlays.is_blocked_edge(zid, n)]

    def _nav_cache(self):
        """Per-instance (distance, hops) memo for zone paths. Keyed by the active blocked-edge signature so
        it auto-invalidates if a block_edge overlay changes (block_edge is the ONLY thing that alters the
        effective zone graph); agent positions never affect zone-to-zone paths, so the memo is valid for the
        whole rollout. Recomputed lazily per clone (fast clone doesn't run __init__)."""
        bl = self.overlays._blocked
        sig = _EMPTY_NAV_SIG if not bl else frozenset(
            k for k in bl if any(self.overlays._active(p) for p in bl[k]))
        c = self.__dict__.get("_navc")
        if c is None or c[0] != sig:
            c = (sig, {}, {}); self.__dict__["_navc"] = c
        return c

    def zone_path(self, src_zone, dst_zone):
        """Dijkstra over zone centers respecting block_edge overlays → (distance, reachable). Memoized."""
        if not src_zone or not dst_zone or src_zone not in self.zones or dst_zone not in self.zones:
            return (-1.0, False)
        if src_zone == dst_zone:
            return (0.0, True)
        pcache = self._nav_cache()[1]
        cached = pcache.get((src_zone, dst_zone))
        if cached is not None:
            return cached
        pq, seen = [(0.0, src_zone)], set()
        while pq:
            d, z = heapq.heappop(pq)
            if z == dst_zone:
                pcache[(src_zone, dst_zone)] = (d, True); return (d, True)
            if z in seen:
                continue
            seen.add(z)
            cz = self.zones[z].center
            for n in self._neighbors(z):
                if n not in seen:
                    nc = self.zones[n].center
                    heapq.heappush(pq, (d + math.hypot(cz[0] - nc[0], cz[1] - nc[1]), n))
        pcache[(src_zone, dst_zone)] = (-1.0, False); return (-1.0, False)

    def zone_hops(self, src_zone, dst_zone):
        """BFS hop count over EFFECTIVE adjacency → (hops, reachable). Used by the rollout teacher for
        decision-step timing (one hop ≈ one tick). Memoized (see _nav_cache)."""
        if not src_zone or not dst_zone or src_zone not in self.zones or dst_zone not in self.zones:
            return (-1, False)
        if src_zone == dst_zone:
            return (0, True)
        hcache = self._nav_cache()[2]
        cached = hcache.get((src_zone, dst_zone))
        if cached is not None:
            return cached
        frontier, seen = [(src_zone, 0)], {src_zone}
        while frontier:
            z, h = frontier.pop(0)
            for n in self._neighbors(z):
                if n == dst_zone:
                    hcache[(src_zone, dst_zone)] = (h + 1, True); return (h + 1, True)
                if n not in seen:
                    seen.add(n); frontier.append((n, h + 1))
        hcache[(src_zone, dst_zone)] = (-1, False); return (-1, False)

    def next_hop(self, src_zone, dst_zone):
        """First zone to step to along the shortest effective path (or src if none/arrived)."""
        if src_zone == dst_zone or src_zone not in self.zones:
            return src_zone
        best = None
        for n in self._neighbors(src_zone):
            h, reach = self.zone_hops(n, dst_zone)
            if reach and (best is None or h < best[0]):
                best = (h, n)
        return best[1] if best else src_zone

    def local_density(self, agent) -> float:
        """Fraction of agents within one scene-scale unit radius (cheap crowd feature)."""
        if len(self.agents) < 2:
            return 0.0
        r = self.scene_scale() * 0.15
        n = sum(1 for b in self.agents.values()
                if b.id != agent.id and math.hypot(b.pos[0] - agent.pos[0], b.pos[1] - agent.pos[1]) <= r)
        return min(1.0, n / 8.0)

    def object_class(self, obj) -> str:
        return infer_object_class(getattr(obj, "object_type", "") or getattr(obj, "id", ""))

    def object_affordances(self, obj) -> list:
        return [a.action for a in obj.affordances]

    def effective_object_state(self, obj) -> str:
        return self.overlays.object_state(obj.id) or obj.state
