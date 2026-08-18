"""
spill.py — Stage 2 smart-object lifecycle (maintenance cascade):

    spill spawned → local area restricted → a staff member is dispatched a grounded clean task
    → reservation + navigation → cleaning completes → restriction clears → spill object removed.

Deterministic scripted cascade (like the emergency responders) — NOT a learned decision. The spill is a real
SmartObject with a `clean` affordance (execution variant of the supported `work` action, so it drives an
existing animation clip); the restriction is an ordinary hazard ScenePatch tagged `spill:<id>` (negative
attraction + disabled affordances) so the crowd avoids the area until it is removed on completion.
"""
import os
import time

from scene.smart_object import SmartObject, Affordance

_CLEAN_TICKS = int(os.environ.get("ECGP_CLEAN_TICKS", "2"))     # dwell to finish cleaning after arrival
_MIN_CLEAN_SEC = float(os.environ.get("ECGP_MIN_CLEAN_SEC", "10"))  # a spill stays at least this long (real seconds)
_SPILL_ATTRACT = -60.0                                          # how strongly the crowd is repelled from a spill
_AUTO = os.environ.get("ECGP_SPILLS", "0") == "1"              # stochastic 'energetic patron drops a drink'
_SPILL_RATE = float(os.environ.get("ECGP_SPILL_RATE", "0.01"))
_MAX_ACTIVE = 2


def maybe_autospawn(scene):
    """Sub-layer-B micro-event: an energetic non-staff patron occasionally drops a drink where it stands.
    OFF by default (ECGP_SPILLS=1) and rate-limited so the floor never turns to chaos."""
    if not _AUTO:
        return
    if len(getattr(scene, "_spills", {}) or {}) >= _MAX_ACTIVE:
        return
    import random
    for aid, a in scene.agents.items():
        if _social_role(a) == "staff" or not getattr(a, "current_zone", ""):
            continue
        if getattr(a.needs, "energy", 50) > 70 and random.random() < _SPILL_RATE:
            spawn_spill(scene, a.current_zone)
            return


def _social_role(a):
    from .live_bridge import _social_role as sr
    return sr(getattr(a, "role", ""))


def _restriction_patch(sid, zone_id):
    from scene.patch import ScenePatch, PatchOp
    return ScenePatch(
        event_type="hazard", display_name=f"spill:{sid}", global_directive="avoid", ttl=10 ** 6,
        ops=[PatchOp(op="zone_attraction", zone=zone_id, delta=_SPILL_ATTRACT),
             PatchOp(op="disable_affordance", zone=zone_id, action="sit"),
             PatchOp(op="disable_affordance", zone=zone_id, action="drink"),
             PatchOp(op="disable_affordance", zone=zone_id, action="eat")])


def _queue(scene, key, item):
    q = getattr(scene, "pending_mutations", None) or {"removed_objects": [], "spawned_agents": []}
    q.setdefault(key, []).append(item)
    scene.pending_mutations = q


def spawn_spill(scene, zone_id, sid=None) -> str:
    """Spawn a spill in `zone_id`: create the SmartObject, restrict the area, register the lifecycle, and
    queue a Unity spawn. Returns the spill id."""
    spills = getattr(scene, "_spills", None)
    if spills is None:
        spills = scene._spills = {}
    sid = sid or f"spill_{len(spills)}"
    cx, cy = scene.zones[zone_id].center if zone_id in scene.zones else (0.0, 0.0)
    obj = SmartObject(id=sid, object_type="spill", zone_id=zone_id, states=["dirty", "clean"], state="dirty",
                      affordances=[Affordance("work", need_effects={"taskProgress": -30}, variant="clean")],
                      pos=(float(cx), float(cy)))
    scene.objects[sid] = obj
    patch = _restriction_patch(sid, zone_id)
    patches = getattr(scene, "active_patches", None)
    if patches is None:
        patches = scene.active_patches = []
    patches.append(patch)
    spills[sid] = {"zone": zone_id, "cleaner": None, "token": None, "progress": 0, "patch": patch,
                   "spawned_at": time.time()}
    _queue(scene, "spawned_objects", {"id": sid, "object_type": "spill", "zone_id": zone_id,
                                      "x": float(cx), "y": float(cy)})
    from .live_bridge import _olog, _zname
    _olog(scene, f"a spill appeared in the {_zname(scene, zone_id)} — closed for cleaning")
    return sid


def _nearest_free_staff(scene, zone, spills):
    busy = {s["cleaner"] for s in spills.values() if s["cleaner"]}
    from .live_bridge import _zone_dist
    staff = [aid for aid, a in scene.agents.items() if _social_role(a) == "staff" and aid not in busy]
    if not staff:
        return None
    return min(staff, key=lambda aid: _zone_dist(scene, scene.agents[aid].current_zone, zone))


def _finish_spill(scene, sid):
    """Cleaning complete: clear the restriction patch, remove the spill object, queue the Unity despawn."""
    st = scene._spills.pop(sid, None)
    if not st:
        return
    # Drop the spill's own restriction patch AND any OTHER closure on the same zone (e.g. the LLM's
    # 'toilet is dirty' hazard, which has its own ttl) — so cleaning FULLY reopens the zone at once and the
    # 'cleaned' message matches the closed sign disappearing (they were out of sync before).
    from scene.patch import zone_matches
    z = scene.zones.get(st["zone"])
    patches = [p for p in getattr(scene, "active_patches", []) if p is not st["patch"]]
    if z is not None:
        patches = [p for p in patches
                   if not any(o.op == "disable_affordance" and o.zone and zone_matches(o, z) for o in p.ops)]
    scene.active_patches = patches
    scene.objects.pop(sid, None)
    _queue(scene, "removed_objects", sid)
    from .live_bridge import _olog, _zname
    _olog(scene, f"staff cleaned up the {_zname(scene, st['zone'])} — it's open again")


def tick_spills(scene):
    """Advance every active spill one step. Returns {cleaner_id: action_override}; performs the reservation,
    navigation, completion, restriction-clear and removal. Safe no-op when there are no spills."""
    spills = getattr(scene, "_spills", None)
    if not spills:
        return {}
    overrides, done = {}, []
    for sid, st in list(spills.items()):
        obj = scene.objects.get(sid)
        if obj is None:
            done.append(sid)
            continue
        zone = st["zone"]
        # (re)assign a free staff cleaner + RESERVE the spill for them
        if st["cleaner"] is None or st["cleaner"] not in scene.agents:
            cleaner = _nearest_free_staff(scene, zone, spills)
            if cleaner:
                st["cleaner"] = cleaner
                st["token"] = obj.reserve(cleaner)
        if not st["cleaner"]:
            continue                                    # no staff available yet — spill persists, area stays restricted
        cl = scene.agents[st["cleaner"]]
        # NAVIGATION: dispatch the cleaner to the spill (deterministic clean task, reason drives the clip)
        overrides[st["cleaner"]] = {"agent_id": st["cleaner"], "action": "move_to_zone", "zone_id": zone,
                                    "smart_object_id": sid, "target_x": obj.pos[0], "target_y": obj.pos[1],
                                    "exec_action": "clean", "reason": "spill:clean"}
        if cl.current_zone == zone:                     # ARRIVED → claim the reservation + accrue cleaning
            if st["token"] and obj.occupancy == 0:
                obj.claim(st["cleaner"], st["token"]); st["token"] = None
            st["progress"] += 1
            # COMPLETION requires BOTH the cleaning dwell AND a minimum real-time (so a cleanup visibly takes
            # ~10s of scrubbing, not one tick) — the staff keeps cleaning until both are met.
            if st["progress"] >= _CLEAN_TICKS and (time.time() - st["spawned_at"]) >= _MIN_CLEAN_SEC:
                obj.complete(st["cleaner"], obj.affordances[0])
                obj.release(st["cleaner"])
                done.append(sid)
        else:
            cl.current_zone = zone                      # step the cleaner toward the spill (Unity animates the walk)
    for sid in done:
        _finish_spill(scene, sid)
    return overrides
