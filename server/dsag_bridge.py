"""
Bridge between the WebSocket server's scene JSON and the pure `dsag` engine.

Lives in the server (not the pure package) because it knows the server's zone/agent JSON
shapes and Unity's action format. Keeps `dsag` engine-agnostic and open-sourceable.

Used ONLY on the "Generate with DSAG" path (director_mode == "dsag"); the existing LLM
director path is untouched.
"""

import os

from dsag import SceneModel, ZoneInstance, AgentInstance, Needs
from dsag.templates import build_objects
from dsag.behavior import STAFF_ROLES
from dsag.smart_object import SmartObject, Affordance, PolicyRule

FOOD_STOCK = int(os.environ.get("ECGP_FOOD_STOCK", "4"))   # visible portions per eat-source

# zone_type -> zone_function(s) the affordance scorer reasons over
_ZONE_FUNC = {
    "entrance": ["access"], "exit": ["access"], "gate": ["access"], "lobby": ["access"],
    "counter": ["service"], "bar": ["service"], "water": ["service"], "kitchen": ["service"],
    "table": ["seating"], "seating": ["seating"], "lounge": ["seating"], "booth": ["seating"],
    "toilet": ["hygiene"], "restroom": ["hygiene"],
    "activity": ["activity"], "stage": ["activity"], "dance": ["activity"],
    "work": ["work"], "desk": ["work"],
    # Wider venues (bedroom / outdoor / gym / library / gallery / shop). Without these every such room fell
    # through to "circulation", so the policy saw a corridor rather than a place with a purpose — and zone
    # FUNCTION, not the label, is what the graph reasons over. Keywords are matched as substrings against
    # zone_type + id, so a zone_type of "bedroom" or "garden" is enough.
    "bed": ["seating"], "sleep": ["seating"],
    "garden": ["activity"], "outdoor": ["activity"], "courtyard": ["activity"], "terrace": ["activity"],
    "gym": ["activity"], "fitness": ["activity"], "play": ["activity"],
    "library": ["work"], "study": ["work"], "reading": ["work"], "office": ["work"],
    "gallery": ["activity"], "exhibit": ["activity"], "museum": ["activity"],
    "shop": ["service"], "store": ["service"], "retail": ["service"],
    "waiting": ["waiting"], "queue": ["waiting"],
}

# personality_type / agent_type hints -> a role in the dsag vocabulary
_STAFF_HINTS = ("barista", "waiter", "staff", "bartender", "cashier", "server", "nurse", "security", "clerk")
_ROLE_BY_PERSONALITY = {
    "vip": "business_person", "socializer": "casual_young",
    "extrovert": "tourist", "introvert": "student", "loner": "elderly",
}


def _zone_function(zone: dict) -> list:
    zt = (zone.get("zone_type", "") + " " + zone.get("id", "")).lower()
    for key, fn in _ZONE_FUNC.items():
        if key in zt:
            return fn
    return ["circulation"]


def _role_for(agent: dict) -> str:
    """The agent's controlled ROLE. An AUTHORED role wins — a hand-built scene states what each agent IS, and
    guessing from `agent_type` keywords both loses information (barista and waiter both collapsed to the bare
    string "staff", so nothing downstream could give them different duty posts) and merges distinct types
    (`regular`/`remote_worker` -> one role). The keyword derivation below stays as the fallback for generated
    scenes, where no role is authored."""
    authored = (agent.get("role") or "").strip().lower()
    if authored:
        return authored
    blob = f"{agent.get('agent_type','')} {agent.get('personality_type','')}".lower()
    if any(h in blob for h in _STAFF_HINTS):
        return "staff"
    return _ROLE_BY_PERSONALITY.get(agent.get("personality_type", ""), "casual_young")


def build_scene_model(zones: list[dict], agents: list[dict]) -> SceneModel:
    """Construct a dsag SceneModel from the server's generated scene."""
    s = SceneModel()
    zone_ids = []
    for z in zones:
        cx = z.get("x", 0.0) + z.get("w", 0.0) * 0.5
        cy = z.get("y", 0.0) + z.get("h", 0.0) * 0.5
        s.add_zone(ZoneInstance(
            id=z["id"], label=z.get("label", z["id"]), zone_type=z.get("zone_type", "generic"),
            zone_function=_zone_function(z), affordances=[], center=(cx, cy)))
        zone_ids.append(z["id"])

    # Where every agent starts. Match on the computed zone FUNCTION first — that is the contract; the literal
    # word "entrance" is just one of several spellings that produce it (lobby/foyer/gate/exit all map to
    # `access`). Keying only on the substring meant renaming the entrance's zone_type to "lobby" silently
    # dropped every agent into zones[0] instead — the whole crowd spawned in the restroom and stayed there.
    start_zone = next((z["id"] for z in zones if "access" in _zone_function(z)),
                      next((z["id"] for z in zones
                            if "entrance" in (z.get("zone_type", "") + z.get("id", "")).lower()),
                           zone_ids[0] if zone_ids else ""))

    have_staff = False
    for a in agents:
        role = _role_for(a)
        have_staff = have_staff or role in STAFF_ROLES
        s.add_agent(AgentInstance(
            id=a["id"], name=a.get("name", a["id"]), role=role,
            social_status="vip" if a.get("personality_type") == "vip" else "regular",
            needs=Needs.from_dict(a.get("needs", {})), current_zone=start_zone,
            group_id=a.get("group_id")))

    # Ensure at least one staff agent so object-service cascades resolve visibly.
    if not have_staff and s.agents:
        first = next(iter(s.agents.values()))
        first.role = "staff"

    # SOCIAL UNITS (item 6): rebuild the group table from the tagged agents so the graph can create groups +
    # authored relationships (family/friend). {group_id: {"type", "members"}} — build_world consumes it.
    groups: dict = {}
    for a in agents:
        gid = a.get("group_id")
        if gid:
            groups.setdefault(gid, {"type": a.get("group_type", "friend"), "members": []})["members"].append(a["id"])
    s.social_groups = {g: d for g, d in groups.items() if len(d["members"]) >= 2}

    for obj in build_objects(zones):
        s.add_object(obj)
    return s


# ── GAP 1: ground the behavior graph in the ACTUAL placed scene (Unity export) ────
# build_scene_model() above instantiates IDEALIZED template objects (one cup/table per zone type, at
# pos (0,0)). Once Unity places the real props and exports the scene graph, we REPLACE those template
# objects with the concrete placed ones — real ids, zones, interaction points, affordances, need_effects.
# The GNN/rule candidate set (encode.py builds `prop:<id>` from scene.objects) then targets objects that
# actually exist on the map and are nav-reachable, and the emitted action carries the real interaction
# point so Unity walks the agent to that exact object rather than a zone centre.

# affordance-set / prop-id -> a state-machine archetype, so the "smart object" cascades (cup empties ->
# needs_clearing; table gets dirty -> needs_cleaning) still emerge on the REAL placed object.
_STATEFUL_SIT   = ("table", "seat", "chair", "bench", "couch", "sofa", "stool", "booth", "desk")
_DISPENSER_KEYS = ("machine", "fountain", "dispenser", "cooler", "fridge", "tap", "sink", "station", "urn")

# Actions the frozen ECGP policy + rule engine understand (ECGP ACTIONS_V1 minus control tokens). The
# LimeZu smart-object catalog uses richer verbs (pay/buy/order/wash/clean/browse…) that are NOT in the
# frozen action vocabulary — left raw they encode as UNK (indistinguishable to the model) and can't be
# learned. We normalize every grounded affordance action to the nearest SUPPORTED semantic action, chosen
# by the affordance's dominant beneficial need effect (principled) with a verb-alias fallback. Effects and
# object identity are preserved; only the action LABEL the policy sees is normalized.
_SUPPORTED_ACTIONS = {"walk", "drink", "eat", "relieve", "sit", "rest", "talk", "help", "work", "observe"}
_NEED_TO_ACTION = {"hunger": "eat", "thirst": "drink", "bladder": "relieve", "energy": "rest",
                   "loneliness": "talk", "groupAffinity": "talk", "curiosity": "observe",
                   "status": "observe", "stress": "rest"}
_VERB_ALIAS = {"wash": "relieve", "clean": "work", "cook": "work", "serve": "work", "prepare": "work",
               "browse": "observe", "read": "observe", "watch": "observe", "view": "observe",
               "play": "observe", "dance": "talk", "order": "eat", "pay": "eat", "buy": "eat",
               "purchase": "eat", "checkout": "eat", "use": "observe", "wait": "observe"}


def normalize_action(action: str, need_effects: dict = None) -> str:
    """Map a (possibly out-of-vocab) affordance verb to a SUPPORTED ECGP action. Prefers the action whose
    need matches the affordance's dominant beneficial effect; else a verb alias; else 'observe'."""
    a = (action or "").strip().lower()
    if a in _SUPPORTED_ACTIONS:
        return a
    if need_effects:
        best, mag = None, 0.0
        for n, dv in need_effects.items():
            benefit = dv if n == "energy" else -dv       # energy higher = better; others lower = better
            if benefit > mag:
                mag, best = benefit, n
        if best in _NEED_TO_ACTION:
            return _NEED_TO_ACTION[best]
    return _VERB_ALIAS.get(a, "observe")


def _object_from_export(o: dict) -> SmartObject:
    """Build one grounded SmartObject from an exported scene_graph smart-object dict."""
    oid  = o.get("smart_object_id")
    zid  = o.get("zone_id")
    pid  = (o.get("prop_id") or "").lower()
    affs = list(o.get("affordances") or [])
    need = dict(o.get("need_effects") or {})
    ix   = float(o.get("ix", o.get("x", 0.0)) or 0.0)
    iy   = float(o.get("iy", o.get("y", 0.0)) or 0.0)
    a    = set(affs)

    if "sit" in a and any(k in pid for k in _STATEFUL_SIT):
        states, state = ["clean", "dirty"], "clean"
        aff_objs = [Affordance("sit", requires_state="clean", changes_state_to="clean",
                               need_effects=need or {"energy": 4, "stress": -5})]
        policy = [PolicyRule(when={"usage_count_gte": 3}, set_state="dirty", emit="needs_cleaning")]
    elif "drink" in a and any(k in pid for k in _DISPENSER_KEYS):
        states, state = ["available"], "available"                    # refillable source: stateless
        aff_objs = [Affordance("drink", need_effects=need or {"thirst": -25})]
        policy = []
    elif "drink" in a:                                                # a cup/glass empties
        states, state = ["full", "empty"], "full"
        aff_objs = [Affordance("drink", requires_state="full", changes_state_to="empty",
                               need_effects=need or {"thirst": -30})]
        policy = [PolicyRule(when={"state": "empty"}, emit="needs_clearing")]
    elif "eat" in a:
        states, state = ["full", "empty"], "full"
        aff_objs = [Affordance("eat", requires_state="full", changes_state_to="empty",
                               need_effects=need or {"hunger": -40})]
        policy = [PolicyRule(when={"state": "empty"}, emit="needs_clearing")]
    elif "relieve" in a:
        states, state = ["available"], "available"
        aff_objs = [Affordance("relieve", need_effects=need or {"bladder": -70})]
        policy = []
    else:                                                            # generic stateless object
        states, state = ["available"], "available"
        aff_objs = [Affordance(x, need_effects=(need if i == 0 else {})) for i, x in enumerate(affs)]
        if not aff_objs:
            aff_objs = [Affordance("observe", need_effects={"curiosity": -15})]
        policy = []

    # Normalize every affordance's action to a SUPPORTED ECGP action (pay/buy/wash → eat/relieve/…) and
    # dedup, so the frozen policy never sees an UNK action. The ORIGINAL catalog verb is kept as a Unity
    # execution `variant` (for rendering only — the model never sees it). need_effects/identity preserved.
    norm, seen = [], set()
    for aff in aff_objs:
        raw = aff.action
        aff.action = normalize_action(raw, aff.need_effects)
        if raw and raw.lower() != aff.action:
            aff.variant = raw.lower()
        if aff.action not in seen:
            seen.add(aff.action)
            norm.append(aff)
    aff_objs = norm

    so = SmartObject(id=oid, object_type=(pid or (affs[0] if affs else "prop")), zone_id=zid,
                     states=states, state=state, affordances=aff_objs, policy=policy, pos=(ix, iy))
    so.label = pid.replace("_", " ") or oid   # richer node text for the open-vocab (v2) encoder
    # The human name, when the export carried one. Only reached on the REBUILD path (an object the server
    # didn't already have grounded); without it a rebuilt object silently reverts to naming itself by asset id
    # in the object log while Unity's panel still looks right — a split-brain that's hard to spot.
    so.display_name = str(o.get("display_name") or "")
    so.grounded = True                         # marks a real placed object with a valid interaction point
    # V2.1 section 5: the catalog is the AUTHORITATIVE source for functional_role, not keyword inference —
    # `infer_object_role_v2` (ecgp/graph/world.py) is consumed only as a fallback (options.py's
    # `getattr(obj, "functional_role", None) or infer_object_role_v2(...)`) for objects that never came
    # from this catalog (e.g. a hand-built test object). Looked up by prop_id via scene_spec's already-
    # loaded catalog index — no re-parsing the JSON per object.
    try:
        from scene_spec import _BY_ID as _CATALOG_BY_ID
        entry = _CATALOG_BY_ID.get(pid)
        if entry is not None and entry.get("functional_role_v2"):
            so.functional_role = entry["functional_role_v2"]
    except Exception:
        pass   # catalog lookup is a best-effort enrichment; grounding must never fail because of it
    return so


def seed_objects_from_spec(scene: SceneModel, smart_objects: list) -> int:
    """Replace the scene's IDEALIZED template objects with the ones a PREBAKED scene authored itself.

    build_scene_model() always calls dsag.templates.build_objects(zones), which invents one generic object
    per zone type at a made-up position (`z0_chair_1`, ...). For a generated scene that is the right seed —
    Unity later exports what it actually placed and ground_scene_in_graph() swaps them out.

    For a PREBAKED level it is wrong from the first tick: the level's objects are authored in the config with
    fixed positions, so the template ids are objects that exist nowhere. The server sends them in the first
    `object_states` message, Unity doesn't recognise the ids, and SimRenderer.SetObjects falls back to
    SCATTERING a sprite for each one on a grid across the zone — the stray smart-object dots in empty floor.
    Seeding from the config makes the server and Unity agree from tick 0, so that fallback never fires.
    Returns the number seeded (0 leaves the template objects alone)."""
    seeded: dict[str, SmartObject] = {}
    for so in smart_objects or []:
        oid, zid = so.get("smart_object_id"), so.get("zone_id")
        if not oid or zid not in scene.zones:
            continue
        affs = list(so.get("affordances") or ([so["affordance"]] if so.get("affordance") else []))
        obj = _object_from_export({
            "smart_object_id": oid, "zone_id": zid, "prop_id": so.get("prop_id") or oid,
            "affordances": affs, "need_effects": dict(so.get("need_effects") or {}),
            "ix": so.get("pos_x", 0.0), "iy": so.get("pos_y", 0.0),
            "x": so.get("pos_x", 0.0), "y": so.get("pos_y", 0.0),
        })
        obj.capacity = int(so.get("capacity", getattr(obj, "capacity", 1)) or 1)
        obj.parent_id = so.get("parent_id") or None
        # FUNCTIONAL ROLE — state it, don't let it be guessed. `object_type` defaults to prop_id, and
        # options._role_compatible falls back to keyword inference over that string, so a prop named
        # "Kitchen_Singles_48x48_186" infers role UNK and its `drink` becomes permanently INFEASIBLE
        # (drink requires provider/consumable). An authored scene knows what its objects are, so let it say:
        #   functional_role: provider | consumable | seat | surface | fixture | workstation | sanitation
        if so.get("functional_role"):
            obj.functional_role = so["functional_role"]
        if so.get("object_type"):
            obj.object_type = so["object_type"]
        # DISPLAY NAME — what a HUMAN calls this thing ("Coffee Machine"), as opposed to `object_type`, which
        # is the ASSET id the level was built from ("Kitchen_Singles_48x48_186"). Every user-facing string —
        # Unity's inspect panel title and the amber object log — reads this first and only falls back to
        # prettifying the asset id when a scene didn't author one (a runtime spill, a generated scene).
        if so.get("display_name"):
            obj.display_name = str(so["display_name"])
        # FOLLOW-ON: the id of an object this one always sends the agent to next (toilet -> sink). Enforced
        # deterministically by live_bridge when the use completes, so it is a HABIT rather than something
        # the policy has to learn and can skip.
        if so.get("follow_on"):
            obj.follow_on = so["follow_on"]
        # ACCESS CONTROL: only agents whose social_role OR group type is listed may use this object
        # (a guest-room bed only the family sleeps in). Absent = unrestricted. See options._agent_may_use.
        if so.get("allowed_roles"):
            obj.allowed_roles = list(so["allowed_roles"])
        # DUTY-POST SLOTS: authored standing positions for staff posted AT this object (the receptionists
        # behind the counter). Ordered — slot 0/1 are the prime "behind the desk" spots, later ones flank.
        # Read by live_bridge's staff-post rule; absent = the generic ring targeting is used instead.
        if so.get("post_slots"):
            obj.post_slots = [(float(p[0]), float(p[1])) for p in so["post_slots"] if len(p) >= 2]
        # PORTIONS: how many servings this source holds. Defaults to FOOD_STOCK for any eat-source; an
        # authored scene sets it to match the number of food props it bound as the visible stock, so the
        # counter runs out exactly when the last plate disappears.
        if so.get("stock") is not None:
            obj.stock = int(so["stock"])
            # remember what FULL means for THIS source. Restock paths previously reset every source to the
            # global FOOD_STOCK, which would hand a counter more portions than it has food props bound.
            obj._full_stock = obj.stock
        # These come out ALREADY `grounded` (via _object_from_export), which is load-bearing rather than
        # incidental: Unity's scene_graph export arrives moments later and ground_scene_in_graph() only
        # PRESERVES an object it already considers grounded — an ungrounded one is rebuilt from the export,
        # and the export carries none of the authored semantics set below (functional_role, follow_on,
        # allowed_roles, stock, display_name). Verified end-to-end: all of them survive re-grounding.
        seeded[oid] = obj
    if not seeded:
        return 0
    scene.objects = seeded
    for z in scene.zones.values():
        z.affordances = sorted({a.action for o in seeded.values()
                                if o.zone_id == z.id for a in o.affordances})
    return len(seeded)


def ground_scene_in_graph(scene: SceneModel, graph: dict) -> int:
    """Reconcile `scene.objects` with the concrete objects Unity placed + exported. Only nav-reachable
    objects in known zones are kept, so the policy can never target an unplaced/unreachable object.

    INCREMENTAL by design (Unity may re-export the graph mid-run): for a smart_object_id we've ALREADY
    grounded, we UPDATE placement (position / zone) but PRESERVE the object's live runtime state — its
    state-machine `state` (dirty table, empty cup), `usage_count`, `occupied_by`, `state_ticks` and which
    `once` policy rules have fired. New ids are created; ids no longer exported are dropped. This stops a
    repeated export from silently resetting a broken/dirty/occupied object back to its defaults.
    Returns the number of grounded objects (0 leaves the current objects untouched)."""
    # ENTRY POINTS: where the level's real doors are (each egress portal's inside gate, validated
    # walkable by Unity's own egress-clearance pass). Arrival spawns (ambient visitors, VIP, musician)
    # use these so new agents walk IN through the entrance instead of materialising at a zone centre.
    eps = (graph or {}).get("entry_points") or []
    scene.entry_points = [(float(p[0]), float(p[1])) for p in eps
                          if isinstance(p, (list, tuple)) and len(p) >= 2]
    objs = (graph or {}).get("smart_objects") or []
    prev = scene.objects
    grounded: dict[str, SmartObject] = {}
    for o in objs:
        oid, zid = o.get("smart_object_id"), o.get("zone_id")
        if not oid or zid not in scene.zones:
            continue
        if not o.get("nav_reachable", True):
            continue
        existing = prev.get(oid)
        if existing is not None and getattr(existing, "grounded", False):
            existing.zone_id = zid                                     # refresh placement only
            ix = float(o.get("ix", o.get("x", existing.pos[0])) or existing.pos[0])
            iy = float(o.get("iy", o.get("y", existing.pos[1])) or existing.pos[1])
            existing.pos = (ix, iy)
            grounded[oid] = existing                                   # PRESERVE state/usage/occupancy/policy
        else:
            grounded[oid] = _object_from_export(o)                     # first sighting -> build fresh
        # scene-grammar link (Phase A): seat AT table / item ON surface / fixture WITH counter — used for
        # facing (sit toward the table) and the eat/drink chains (source -> seat -> consume).
        grounded[oid].parent_id = o.get("parent_id") or None
        grounded[oid].relation = o.get("relation") or None
        # V2.1 section 4 (Gate 1): the AUTHORED cluster link, round-tripped from prop_grammar.py's PASS 5
        # through the REAL Unity export (SimRenderer.BuildSceneGraph -> SceneGraphObject.cluster_id /
        # provides_cluster_id) — this is the authoritative alternative to same-zone proximity guessing.
        grounded[oid].cluster_id = o.get("cluster_id") or None
        grounded[oid].provides_cluster_id = o.get("provides_cluster_id") or None
        # prop CENTRE (pos above is the interaction stand-point): mount actions (sit/relieve) put the agent
        # ON the prop itself, so both positions are needed.
        ix0, iy0 = grounded[oid].pos
        grounded[oid].prop_pos = (float(o.get("x", ix0) or ix0), float(o.get("y", iy0) or iy0))
        # FOOD STOCK: an eat-source carries visible portions; each consumption takes one (Unity draws them),
        # empty -> zone restocks ('be back soon'). Preserved across re-grounding (only set on first sight).
        if not hasattr(grounded[oid], "stock") and any(a.action == "eat" for a in grounded[oid].affordances):
            grounded[oid].stock = FOOD_STOCK
    if not grounded:
        return 0
    scene.objects = grounded
    for z in scene.zones.values():                 # refresh each zone's advertised affordances
        z.affordances = sorted({a.action for o in grounded.values()
                                if o.zone_id == z.id for a in o.affordances})
    return len(grounded)


# ── dsag decision -> Unity action (reuses existing move_to_zone / idle handlers) ──
def _wire_action(scene: SceneModel, rec) -> dict:
    """Trace record -> Unity wire action. When the decision targeted a concrete smart object (rec.target
    is an object id, not a zone), carry its smart_object_id + real interaction point so Unity navigates to
    the ACTUAL placed object; otherwise emit a plain zone move. Previously the object target was silently
    collapsed to the agent's current zone (`rec.target if rec.target in scene.zones else current_zone`)."""
    agent = scene.agents.get(rec.agent_id)
    obj = scene.objects.get(rec.target)
    if obj is not None:
        if getattr(obj, "grounded", False):        # real placed object: id + interaction point Unity can resolve
            return {"agent_id": rec.agent_id, "action": "move_to_zone", "zone_id": obj.zone_id,
                    "smart_object_id": obj.id, "target_x": obj.pos[0], "target_y": obj.pos[1],
                    "reason": rec.reason}
        # ungrounded template object (no real placed id/pos yet) -> plain zone move
        return {"agent_id": rec.agent_id, "action": "move_to_zone", "zone_id": obj.zone_id,
                "reason": rec.reason}
    tz = rec.target if rec.target in scene.zones else (agent.current_zone if agent else "")
    if not tz and agent:
        tz = agent.current_zone
    # EMERGENCY: route the agent OUTSIDE the floor plan (through the exit) rather than clumping them at an
    # interior exit zone, so the building actually empties. Unity walks them to the off-map point and hides
    # them on arrival (evacuate flag). rec.reason == "evacuate" tags the deterministic evac decisions.
    if getattr(scene, "is_emergency", lambda: False)() and rec.need == "evacuate":
        ox, oy = _outside_point(scene, tz)
        return {"agent_id": rec.agent_id, "action": "move_to_zone", "zone_id": tz,
                "target_x": ox, "target_y": oy, "evacuate": True, "reason": rec.reason}
    return {"agent_id": rec.agent_id, "action": "move_to_zone" if tz else "idle",
            "zone_id": tz, "reason": rec.reason}


# world half-extents used by grid_layout (X in [-8,8], Y in [-5,5]); "outside" is just past a boundary.
_WORLD_X, _WORLD_Y, _OUT_MARGIN = 8.0, 5.0, 3.5


def entry_point(scene: SceneModel, near_zone: str = None):
    """Where an ARRIVING agent should appear: just inside the level's real door (Unity-exported
    egress-portal gate), so new visitors walk in through the entrance instead of materialising mid-room.
    `near_zone` picks the door closest to that zone when the level has several. Falls back to the exit
    zone's centre (then origin) when no graph has been exported yet — same visible behaviour as before."""
    eps = getattr(scene, "entry_points", None) or []
    if eps:
        if near_zone and near_zone in scene.zones:
            cx, cy = scene.zones[near_zone].center
            return min(eps, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        return eps[0]
    z = scene.zones.get(near_zone) if near_zone else None
    if z is None:
        z = next((zz for zz in scene.zones.values() if getattr(zz, "is_exit", False)), None)
    return tuple(z.center) if z is not None else (0.0, 0.0)


def _outside_point(scene: SceneModel, exit_zone_id: str):
    """A point just BEYOND the world boundary nearest the exit zone — where evacuating agents walk off the
    floor plan. Falls back to the left edge if the exit has no known centre."""
    z = scene.zones.get(exit_zone_id)
    cx, cy = z.center if z is not None else (-_WORLD_X, 0.0)
    dl, dr = cx + _WORLD_X, _WORLD_X - cx            # dist to left / right boundary
    db, dt = cy + _WORLD_Y, _WORLD_Y - cy            # dist to bottom / top boundary
    m = min(dl, dr, db, dt)
    if m == dl:  return (-_WORLD_X - _OUT_MARGIN, cy)
    if m == dr:  return ( _WORLD_X + _OUT_MARGIN, cy)
    if m == db:  return (cx, -_WORLD_Y - _OUT_MARGIN)
    return (cx, _WORLD_Y + _OUT_MARGIN)


def build_dining_clusters(scene: SceneModel) -> dict:
    """V2.1 section 5 — synthesize a `cluster_root` SmartObject per table that has >=1 GENUINELY BOUND chair
    and a reachable provider. Graph-side ONLY (ecgp's EcgpWorld, wired in live_bridge.build_world) — does NOT
    touch scene.objects / Unity's object list, so this cannot affect the rule-based engine or anything Unity
    renders; it gives the GNN one honest macro-option ("eat@dining_cluster_root") instead of forcing it to
    reason about table+chair+counter as three independent, ungrounded picks.

    IMPORTANT provenance note (V2.1 section 2 follow-up): the chair<->table link below reads the REAL
    authored relation from `prop_grammar.py`'s scene-GENERATION-time functional-cluster grammar (its `at`
    relation — a chair is bound to its zone's shared surface when the grammar places it, same mechanism the
    machine/register "with" and consumable "on" relations use), not a runtime heuristic — `parent_id` alone
    would also match an "on" (food-on-table) or "with" (machine-with-counter) child, so `relation == "at"`
    is checked explicitly to be precise about which authored edge actually means "seat belongs to this
    table". The provider link is STILL same-zone proximity, not an authored relation — prop_grammar.py's
    PASS 1/2 already consolidates one shared surface + provider PER ZONE for that zone's own affordances (a
    "dining cluster" is naturally single-zone in the current grammar), so same-zone is a reasonable proxy
    today, but there is no explicit PROVIDES/PROVIDED_BY edge yet (section 2's ask) — a real, disclosed gap:
    extending prop_grammar.py to author an explicit provider<->cluster edge (for the cross-zone case, e.g. a
    separate kitchen serving a dining room) is follow-on work, not done here.

    GATE 1 UPDATE: the PROVIDES gap above is now closed for scenes grounded from a REAL Unity export —
    `cluster_id`/`provides_cluster_id` are authored by `prop_grammar.py`'s PASS 5, round-tripped through
    `SimRenderer.BuildSceneGraph` (SceneSpec.cs/SimRenderer.cs) and read back by
    `ground_scene_in_graph`/`_object_from_export` onto each grounded SmartObject. When that authored data is
    present, it is used AUTHORITATIVELY (no same-zone guessing at all, including for a cross-zone provider —
    the exact case the old heuristic could never handle). The same-zone heuristic below remains a FALLBACK
    for scenes that never went through prop_grammar.py (e.g. dsag/templates.py's stage-1 path)."""
    from ecgp.graph.world import infer_object_role_v2
    authored: dict[str, dict] = {}
    for oid, o in scene.objects.items():
        cid = getattr(o, "cluster_id", None)
        if cid:
            entry = authored.setdefault(cid, {"table": None, "chairs": [], "provider": None})
            if infer_object_role_v2(o.object_type) == "surface":
                entry["table"] = oid
            elif getattr(o, "relation", None) == "at" and infer_object_role_v2(o.object_type) == "seat":
                entry["chairs"].append(oid)
        pcid = getattr(o, "provides_cluster_id", None)
        if pcid and not getattr(o, "removed", False):
            authored.setdefault(pcid, {"table": None, "chairs": [], "provider": None})["provider"] = oid
    clusters = {}
    for cid, entry in authored.items():
        if not (entry["table"] and entry["chairs"] and entry["provider"]):
            continue
        table = scene.objects[entry["table"]]
        chairs = [scene.objects[c] for c in entry["chairs"]]
        n_slots = sum(int(getattr(c, "interaction_slots", None) or c.capacity or 1) for c in chairs)
        cluster = SmartObject(id=cid, object_type="dining_cluster", zone_id=table.zone_id,
                              states=["default"], state="default", pos=table.pos, capacity=n_slots)
        cluster.affordances = [Affordance("eat", need_effects={"hunger": -70})]
        cluster.functional_role = "cluster_root"
        cluster.members = {"table": entry["table"], "chairs": entry["chairs"], "provider": entry["provider"]}
        # V2.1 Gate 1: give encode()'s hierarchy pass (ecgp/graph/encoder.py) the field it actually reads
        # (`provider_object_id`) so the authored provides_cluster_id link becomes a real PROVIDES/PROVIDED_BY
        # graph edge (cluster --provided_by--> provider, provider --provides--> cluster), not just a scalar
        # attribute that survives grounding but is never turned into an edge.
        cluster.provider_object_id = entry["provider"]
        clusters[cid] = cluster
    if clusters:
        return clusters

    # FALLBACK: no authored cluster_id anywhere in this scene -> same-zone heuristic.
    for oid, o in scene.objects.items():
        if infer_object_role_v2(o.object_type) != "surface":
            continue
        chairs = [c for c in scene.objects.values()
                  if getattr(c, "parent_id", None) == oid and getattr(c, "relation", None) == "at"
                  and infer_object_role_v2(c.object_type) == "seat"]
        if not chairs:
            continue
        providers = [p for p in scene.objects.values()
                     if p.zone_id == o.zone_id and infer_object_role_v2(p.object_type) == "provider"
                     and not getattr(p, "removed", False)]
        if not providers:
            continue
        cid = f"cluster_{oid}"
        n_slots = sum(int(getattr(c, "interaction_slots", None) or c.capacity or 1) for c in chairs)
        cluster = SmartObject(id=cid, object_type="dining_cluster", zone_id=o.zone_id,
                              states=["default"], state="default", pos=o.pos,
                              capacity=n_slots)
        cluster.affordances = [Affordance("eat", need_effects={"hunger": -70})]
        cluster.functional_role = "cluster_root"
        cluster.members = {"table": oid, "chairs": [c.id for c in chairs], "provider": providers[0].id}
        cluster.provider_object_id = providers[0].id
        clusters[cid] = cluster
    return clusters


def build_leave_action(scene: SceneModel, aid: str, reason: str = "ecgp:leave_home") -> dict:
    """THE single constructor for a 'walk to the exit and OFF the map' action (a personal/group leave, an
    end-of-day departure, or a frustration give-up). One definition so the ECGP decision precedence and the
    server's safety layer produce byte-identical actions and can never drift. `reason` distinguishes the caller."""
    exit_id = scene.exit_zone_id()
    ox, oy = _outside_point(scene, exit_id)
    return {"agent_id": aid, "action": "move_to_zone", "zone_id": exit_id,
            "target_x": ox, "target_y": oy, "evacuate": True, "reason": reason}


def tick_to_unity(scene: SceneModel):
    """Run one dsag tick; return (actions, object_states, events) in the server's wire format."""
    events = scene.tick()
    actions = [_wire_action(scene, rec)
               for rec in scene.trace.records if rec.tick == scene.tick_no]
    object_states = [o.render_state() for o in scene.objects.values()]
    event_list = [e.as_dict() for e in events]
    return actions, object_states, event_list
