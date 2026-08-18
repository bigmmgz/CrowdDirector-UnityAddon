"""
ambient.py — small WORLD events on a timer, so a running scene feels alive without anyone typing into the
director field.

This is the hybrid half of the demo. It deliberately does NOT decide anything for an agent: CrowdDirect v3
still chooses every action. What this does is change the WORLD around them — a spill appears, a delivery
arrives, new customers walk in, someone finishes up and leaves — and the policy reacts to that changed
world on its next tick. Nothing here writes to `_COMMIT`, picks a target, or overrides a decision, so the
"driven by v3" property is preserved exactly.

Everything is built from mechanics that already existed and were only reachable via a typed EventPatch:
  spill      -> policy.runtime.spill.spawn_spill  (staff clean it, the zone shows a 'closed' sign meanwhile)
  delivery   -> refills a stocked food source early; Unity's bound portion props reappear
  arrival    -> queues a spawn_agent mutation, mirrored to Unity as new characters at the entrance
  departure  -> marks one non-staff agent as leaving; the existing egress path walks it out and despawns it

Cadence is per-event with its own cooldown, so two events never land on the same tick, and a scene can
retune or disable any of them via `ambient_events` in its config. Set ECGP_AMBIENT=0 to turn the whole
thing off and get a purely need-driven scene back.
"""
import logging
import os
import random

log = logging.getLogger("policy.live")

ENABLED = os.environ.get("ECGP_AMBIENT", "1") == "1"

# kind -> (min_gap_ticks, chance_per_eligible_tick). Tuned for a ~3s tick: something happens every
# 30-60s without the scene turning into a slot machine.
DEFAULTS = {
    "spill":      (30, 0.18),
    "delivery":   (14, 0.30),
    "arrival":    (12, 0.35),
    "departure":  (16, 0.25),
    # ── Tier-A "life-sim" events. These change the world through the FOUR channels the policy was actually
    # trained on (zone_attraction / object_attraction / zone_hazard / disable_affordance) and nothing else.
    # Rarer than the four above: each one visibly reshapes the crowd, so they should feel like occasions.
    "free_samples":   (36, 0.20),
    "rush_hour":      (50, 0.18),
    "performance":    (52, 0.18),
    "breakdown":      (70, 0.10),
    "section_closed": (90, 0.10),
    "closing_soon":   (140, 0.08),
}

# TTLs in ticks (patch.ttl is decremented once per director tick and the patch is dropped at 0).
_TTL = {"free_samples": 22, "rush_hour": 34, "performance": 44,
        "breakdown": 30, "section_closed": 28, "closing_soon": 26}

_last = {}          # kind -> tick it last fired
_rng = random.Random(20260807)


def configure(scene_cfg: dict):
    """Per-scene overrides: {"ambient_events": {"spill": {"every": 20, "chance": 0.3}, "arrival": false}}."""
    _last.clear()
    cfg = (scene_cfg or {}).get("ambient_events")
    if cfg is None:
        return dict(DEFAULTS)
    out = dict(DEFAULTS)
    for kind, v in cfg.items():
        if v is False:
            out.pop(kind, None)
        elif isinstance(v, dict):
            gap, chance = out.get(kind, (15, 0.3))
            out[kind] = (int(v.get("every", gap)), float(v.get("chance", chance)))
    return out


def _due(kind, tick, gap, chance):
    if tick - _last.get(kind, -999) < gap:
        return False
    if _rng.random() > chance:
        return False
    _last[kind] = tick
    return True


def _busy_zones(scene):
    """Zones with agents in them — a spill in an empty room nobody visits is invisible."""
    counts = {}
    for a in scene.agents.values():
        if a.current_zone:
            counts[a.current_zone] = counts.get(a.current_zone, 0) + 1
    return [z for z, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


# ── Tier-A helpers ───────────────────────────────────────────────────────────────────────────────────
# The events below are built ONLY from op kinds the deployed checkpoint saw in training
# (zone_attraction 34%, zone_hazard 20%, disable_affordance 10%, object_attraction 10%). Op kinds that
# never appeared in the 216-shard corpus — object_state, emotion_delta, relationship_* , event_clear —
# are invisible to the policy and are deliberately NOT used: an event built on them could only ever be a
# scripted animation, not something the crowd reasons about.
#
# MAGNITUDE IS NOT A TUNING KNOB. The encoder stores zone attraction as a single BIT (`attraction != 0`)
# and object nodes carry no attraction feature at all. Intensity has to come from WHAT is targeted (how
# many objects, which competitors are disabled, hazard severity), never from a bigger delta.

def _tag(kind, key=""):
    return f"ambient_{kind}:{key}" if key else f"ambient_{kind}"


def _has_patch(scene, prefix):
    return any(str(getattr(p, "display_name", "")).startswith(prefix)
               for p in (getattr(scene, "active_patches", []) or []))


def _emit(scene, story, kind, key, event_type, ops, ttl, line):
    """Append one ambient ScenePatch and its director-log line.

    `origin="ambient"` is what keeps this OUT of the deterministic party/gather branch — see
    live_bridge._attracted_zones. Without it a positive zone_attraction here would seize every eligible
    agent for the patch's whole ttl and CrowdDirect v3 would stop deciding for the duration.
    """
    from scene.patch import ScenePatch
    from policy.runtime.live_bridge import AMBIENT_PATCH_FLAG
    patches = getattr(scene, "active_patches", None)
    if patches is None:
        patches = scene.active_patches = []
    patches.append(ScenePatch(event_type=event_type, display_name=_tag(kind, key), ttl=ttl,
                              ops=ops, origin=AMBIENT_PATCH_FLAG))
    story.append(line)
    log.info(f"[ambient] {kind} {key} ttl={ttl} ops={[o.op for o in ops]}")
    return 1


def _zname(scene, zid):
    from policy.runtime.live_bridge import _zname as z
    return z(scene, zid)


def _zones_by_function(scene, *fns):
    out = []
    for zid, z in scene.zones.items():
        zf = getattr(z, "zone_function", None) or []
        if isinstance(zf, str):
            zf = [zf]
        if any(f in zf for f in fns):
            out.append(zid)
    return out


def _live_objects(scene, action):
    """Objects currently offering `action` — excluding removed ones, so a breakdown can't pick a machine
    that is already out of service and report a second failure for it."""
    return [o for o in scene.objects.values()
            if not getattr(o, "removed", False)
            and any(getattr(a, "action", None) == action for a in getattr(o, "affordances", []))]


# ── event coherence guards ───────────────────────────────────────────────────────────────────────────
# Tier-A events must not CONTRADICT each other on screen. Observed in a live run: closing_soon shut the
# service zones at tick 9 (a "closed" sign over the food counter ~45s into the day), then rush_hour and
# free_samples fired anyway and pulled a crowd to the counter that was showing the sign. Two rules fix it:
# attraction events skip a zone whose eat/drink is currently disabled, and closing only starts late in the
# scene and never during an active attraction.

CLOSING_MIN_TICK = 60          # no "closing soon" in the first ~5 minutes of a scene

def _service_disabled(scene, zid):
    """True when any active patch disables eat/drink at this zone (closing, restock, section_closed…)."""
    for p in (getattr(scene, "active_patches", []) or []):
        for op in getattr(p, "ops", []):
            if op.op == "disable_affordance" and op.action in ("eat", "drink") \
                    and (op.zone == zid or (op.zone is None and op.zone_function is None and not op.object)):
                return True
    return False


def _closure_active(scene):
    """True while ANY ambient closure is still showing — one closed sign at a time is an event, several at
    once reads as the venue shutting down."""
    return (_has_patch(scene, _tag("section_closed", "")) or _has_patch(scene, _tag("closing_soon", "")))


def _attraction_active(scene):
    return (_has_patch(scene, _tag("rush_hour", "")) or _has_patch(scene, _tag("free_samples", ""))
            or _has_patch(scene, _tag("performance", "")))


def tick(scene, cadence, story):
    """Run one ambient step. `story` collects human-readable lines for the director log. Returns the number
    of events fired (0 or 1 — at most one per tick, so the scene never lurches)."""
    if not ENABLED or not cadence:
        return 0
    t = int(getattr(scene, "tick_no", 0) or 0)
    if t < 4:                                   # let the scene settle before anything happens
        return 0

    # ── a spill appears in a busy room; the existing lifecycle sends staff to clean it ────────────────
    if "spill" in cadence and _due("spill", t, *cadence["spill"]):
        zones = [z for z in _busy_zones(scene) if z in scene.zones]
        already = set(getattr(scene, "_spills", {}) or {})
        if zones and len(already) < 2:
            zid = zones[0]
            try:
                from policy.runtime.spill import spawn_spill
                spawn_spill(scene, zid)
                from policy.runtime.live_bridge import _zname
                story.append(f"someone spilled a drink in the {_zname(scene, zid)} — staff are on it")
                log.info(f"[ambient] spill in {zid}")
                return 1
            except Exception as e:
                log.warning(f"[ambient] spill failed: {e}")

    # ── a delivery tops a food source back up early ──────────────────────────────────────────────────
    if "delivery" in cadence and _due("delivery", t, *cadence["delivery"]):
        low = [o for o in scene.objects.values()
               if getattr(o, "stock", None) is not None and o.stock is not None and o.stock >= 0
               and o.stock < _full_stock(o) and not getattr(o, "removed", False)]
        if low:
            o = min(low, key=lambda x: x.stock)
            o.stock = _full_stock(o)
            if "empty" in getattr(o, "states", []) and o.state == "empty":
                o.state = "full"
            story.append("a delivery arrived — the food counter is stocked again")
            log.info(f"[ambient] delivery restocked {o.id} -> {o.stock}")
            return 1

    # ── new customers walk in ────────────────────────────────────────────────────────────────────────
    _forced_arrival = getattr(scene, "_arrival_burst", 0) > 0     # post-fire refill: skip the dice
    if "arrival" in cadence and (_forced_arrival or _due("arrival", t, *cadence["arrival"])):
        if len(scene.agents) < _MAX_AGENTS:
            zid = _access_zone(scene)
            if zid:
                n = _rng.choice((1, 1, 2))
                spawned = [_spawn_visitor(scene, zid, i) for i in range(n)]
                q = getattr(scene, "pending_mutations", None) or {"removed_objects": [], "spawned_agents": []}
                q["spawned_agents"] = q.get("spawned_agents", []) + spawned
                scene.pending_mutations = q
                story.append(f"{n} new visitor{'s' if n > 1 else ''} came in")
                log.info(f"[ambient] arrival x{n} at {zid}: {[s['id'] for s in spawned]}")
                if _forced_arrival:
                    scene._arrival_burst = max(0, getattr(scene, "_arrival_burst", 0) - 1)
                return 1

    # ── someone finishes up and heads out ────────────────────────────────────────────────────────────
    if "departure" in cadence and _due("departure", t, *cadence["departure"]):
        from policy.runtime import live_bridge as LB
        leaving = set(getattr(LB, "_LEAVING", {}) or {})
        cands = [a for aid, a in scene.agents.items()
                 if aid not in leaving and (getattr(a, "role", "") or "").lower() not in _STAFF_ROLES]
        if len(cands) > 8:                       # never empty the venue out
            a = _rng.choice(cands)
            LB._LEAVING[a.id] = 0
            story.append(f"{getattr(a, 'name', a.id)} headed home")
            log.info(f"[ambient] departure {a.id}")
            return 1

    fired = _tier_a(scene, cadence, story, t)
    if fired:
        return fired
    return 0


# ── Tier-A events ────────────────────────────────────────────────────────────────────────────────────

def _tier_a(scene, cadence, story, t):
    from scene.patch import PatchOp

    # 1) FREE SAMPLES — one object gets the crowd's attention. The narrowest possible event: a single
    #    object_attraction and nothing else, so it exercises the object channel in isolation.
    if "free_samples" in cadence and _due("free_samples", t, *cadence["free_samples"]):
        cands = _live_objects(scene, "eat") or _live_objects(scene, "drink")
        cands = [o for o in cands if not _has_patch(scene, _tag("free_samples", o.id))
                 and not _service_disabled(scene, o.zone_id)]   # never advertise a closed counter
        if cands:
            o = _rng.choice(cands)
            return _emit(scene, story, "free_samples", o.id, "attraction",
                         [PatchOp(op="object_attraction", object=o.id, delta=50.0)],
                         _TTL["free_samples"],
                         f"free samples are out at the {_zname(scene, o.zone_id)}")

    # 2) RUSH HOUR — the service area gets busy: the zone pulls, and its counter pulls harder. Deliberately
    #    NOT a spawn_agent; `arrival` above already owns "more people come in", and mixing the two would
    #    make the venue grow every rush.
    if "rush_hour" in cadence and _due("rush_hour", t, *cadence["rush_hour"]):
        zids = [z for z in _zones_by_function(scene, "service")
                if not _has_patch(scene, _tag("rush_hour", z))
                and not _service_disabled(scene, z)]            # a closed counter cannot have a rush
        if zids:
            zid = zids[0]
            ops = [PatchOp(op="zone_attraction", zone=zid, delta=55.0)]
            counters = [o for o in _live_objects(scene, "eat") + _live_objects(scene, "drink")
                        if o.zone_id == zid]
            if counters:
                ops.append(PatchOp(op="object_attraction", object=counters[0].id, delta=45.0))
            return _emit(scene, story, "rush_hour", zid, "attraction", ops, _TTL["rush_hour"],
                         f"it's getting busy at the {_zname(scene, zid)}")

    # 3) LIVE PERFORMANCE — the activity room pulls. Only ONE zone is touched, so needs elsewhere still
    #    compete for the crowd rather than the whole map emptying into the stage.
    if "performance" in cadence and _due("performance", t, *cadence["performance"]):
        zids = [z for z in _zones_by_function(scene, "activity")
                if not _has_patch(scene, _tag("performance", z))]
        if zids:
            zid = _rng.choice(zids)
            ops = [PatchOp(op="zone_attraction", zone=zid, delta=60.0)]
            watchable = [o for o in _live_objects(scene, "observe") if o.zone_id == zid]
            if watchable:
                ops.append(PatchOp(op="object_attraction", object=watchable[0].id, delta=50.0))
            return _emit(scene, story, "performance", zid, "attraction", ops, _TTL["performance"],
                         f"live music started in the {_zname(scene, zid)}")

    # 4) EQUIPMENT BREAKDOWN — ONE machine goes out of service, object-scoped so the rest of the room keeps
    #    working. Fires only when an ALTERNATIVE provider of the same action still exists; breaking the last
    #    coffee machine in the venue is not an interesting event, it is a starved need with nowhere to go.
    if "breakdown" in cadence and _due("breakdown", t, *cadence["breakdown"]):
        for action in ("drink", "eat"):
            live = [o for o in _live_objects(scene, action)
                    if not _has_patch(scene, _tag("breakdown", o.id))]
            if len(live) >= 2:
                o = _rng.choice(live)
                return _emit(scene, story, "breakdown", o.id, "object_disable",
                             [PatchOp(op="disable_affordance", object=o.id, action=action)],
                             _TTL["breakdown"],
                             f"the {action} machine in the {_zname(scene, o.zone_id)} broke down")

    # 5) SECTION CLOSED — a room is shut for a while. Both ops are needed and they do different jobs:
    #    disable_affordance is the STRUCTURAL closure (it makes the room's options infeasible), while
    #    zone_hazard is the soft keepaway that also steers pass-through traffic. Hazard alone would NOT be
    #    enough — under tri-state feasibility a hazard leaves object options feasible and merely flags them
    #    temporarily unavailable, so agents would keep walking in.
    if ("section_closed" in cadence and not _closure_active(scene)
            and _due("section_closed", t, *cadence["section_closed"])):
        busy = set(_busy_zones(scene)[:2])          # never shut the room the crowd is currently in
        rest_zones = {o.zone_id for o in scene.objects.values()
                      if any(getattr(a, "action", None) in ("rest", "sleep")
                             for a in getattr(o, "affordances", []))}
        zids = [z for z in _zones_by_function(scene, "seating", "work")
                if z not in busy and z not in rest_zones          # never close the bedroom/nap spots
                and not _has_patch(scene, _tag("section_closed", z))]
        if zids:
            zid = _rng.choice(zids)
            ops = [PatchOp(op="zone_hazard", zone=zid, delta=60.0, severity=0.45)]
            for act in ("sit", "rest", "work", "eat", "drink"):
                ops.append(PatchOp(op="disable_affordance", zone=zid, action=act))
            return _emit(scene, story, "section_closed", zid, "hazard", ops, _TTL["section_closed"],
                         f"the {_zname(scene, zid)} is closed for a while")

    # 6) CLOSING SOON — a two-stage wind-down, one stage per firing. Stage 1 stops service; stage 2 makes
    #    the seating areas unattractive so the room drains toward the exit. Deliberately NOT a global
    #    "leave" directive: that is the agent-directive channel and would override the policy outright.
    if ("closing_soon" in cadence and t >= CLOSING_MIN_TICK and not _attraction_active(scene)
            and not _has_patch(scene, _tag("section_closed", ""))
            and _due("closing_soon", t, *cadence["closing_soon"])):
        stage = getattr(scene, "_ambient_closing_stage", 0)
        if stage == 0:
            zids = _zones_by_function(scene, "service")
            if zids:
                ops = [PatchOp(op="disable_affordance", zone=z, action=a)
                       for z in zids for a in ("eat", "drink")]
                scene._ambient_closing_stage = 1
                return _emit(scene, story, "closing_soon", "service", "object_disable", ops,
                             _TTL["closing_soon"], "last orders — the counter has stopped serving")
        else:
            zids = _zones_by_function(scene, "seating")
            if zids:
                ops = []
                for z in zids:
                    ops.append(PatchOp(op="zone_hazard", zone=z, delta=40.0, severity=0.35))
                    ops.append(PatchOp(op="disable_affordance", zone=z, action="sit"))
                scene._ambient_closing_stage = 0
                return _emit(scene, story, "closing_soon", "seating", "hazard", ops,
                             _TTL["closing_soon"], "the seating areas are being cleared for closing")
    return 0


_MAX_AGENTS = int(os.environ.get("ECGP_AMBIENT_MAX_AGENTS", "40"))

_STAFF_ROLES = {"staff", "barista", "waiter", "server", "bartender", "cashier", "security", "cook", "chef"}


def _spawn_visitor(scene, zid, i):
    """Add ONE new customer as a REAL agent in the graph and return the spawn dict Unity needs.

    Both halves matter and the first one is the one that bites: an entry queued into `spawned_agents`
    WITHOUT registering an AgentInstance gives Unity a body that no policy tick ever decides for (it stands
    still forever), and a dict without an `id` makes Unity's AgentConfig.id null — which then throws
    ArgumentNullException the moment any neighbour keys a relationship dictionary on it. So: same id source
    (`_next_agent_id`) and same payload shape as the graph-edit `spawn_agent` op, no shortcuts.
    """
    from scene.patch import _next_agent_id
    from scene.world import AgentInstance
    from scene.needs import Needs

    aid = _next_agent_id(scene)
    role = _visitor_role(scene)
    n = 1 + sum(1 for a in scene.agents.values() if str(getattr(a, "name", "")).startswith("Visitor "))
    name = f"Visitor {n}"
    # An arriving customer is mildly hungry/thirsty — it walks in with a reason to go somewhere, rather
    # than idling at the door until needs drift up on their own.
    needs = Needs(hunger=float(45 + _rng.randint(0, 25)), thirst=float(45 + _rng.randint(0, 25)),
                  curiosity=float(55 + _rng.randint(0, 25)))
    scene.agents[aid] = AgentInstance(id=aid, name=name, role=role, needs=needs, current_zone=zid)

    # Spawn AT THE DOOR (Unity-exported egress gate), not at the zone centre — a visitor who appears
    # mid-room reads as teleporting in. The small x-jitter keeps a multi-arrival burst from stacking.
    from scene_bridge import entry_point
    cx, cy = entry_point(scene, near_zone=zid)
    return {"id": aid, "name": name, "role": role, "agent_type": role, "zone_id": zid,
            "x": float(cx + (i - 0.5) * 0.6), "y": float(cy), "needs": needs.as_dict()}


def _visitor_role(scene):
    """Borrow a role the scene already uses for its customers, so the newcomer is in-vocabulary for the
    encoder AND resolves to a real character sprite in Unity. Falls back to the generic civilian role."""
    roles = [r for r in ((getattr(a, "role", "") or "").lower() for a in scene.agents.values())
             if r and r not in _STAFF_ROLES]
    return _rng.choice(roles) if roles else "casual_young"


def _full_stock(o):
    """This source's OWN full level (set when the scene authored its stock), not the global default —
    otherwise a delivery hands a counter more portions than it has food props to show."""
    import scene_bridge
    return int(getattr(o, "_full_stock", None) or scene_bridge.FOOD_STOCK)


def _access_zone(scene):
    for zid, z in scene.zones.items():
        fns = getattr(z, "zone_function", None) or []
        if "access" in fns:
            return zid
    return next(iter(scene.zones), None)
