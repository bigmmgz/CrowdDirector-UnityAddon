"""
Affordance-constrained behaviour: pick each agent's action per tick, grounded in the
Scene-Affordance Graph (needs -> afforded actions on reachable zones/objects).

Deterministic and fast — the per-tick backbone. The LLM can optionally refine the output,
but nothing here calls it, so runs are reproducible and cleanly ablatable.
"""

import math
from dataclasses import dataclass
from typing import Optional
from .needs import top_need

STAFF_ROLES = {"staff", "barista", "waiter", "bartender", "nurse", "security"}

# which zone_functions plausibly satisfy a given need (when no object addresses it directly)
NEED_ZONEFUNC = {
    "thirst": ["service"], "hunger": ["service"], "bladder": ["hygiene"],
    "energy": ["seating"], "stress": ["access", "seating"],
    "loneliness": ["seating", "activity"], "groupAffinity": ["activity", "seating"],
    "taskProgress": ["work"], "curiosity": ["activity"], "status": ["activity", "service"],
}
# the affordance a zone offers for a given need — lets the fallback skip a zone whose affordance a
# patch DISABLED (e.g. "restroom is closed" disables relieve -> bladder agents must not go there).
NEED_ACTION = {
    "thirst": "drink", "hunger": "eat", "bladder": "relieve", "energy": "sit",
    "loneliness": "talk", "groupAffinity": "talk", "curiosity": "observe",
    "status": "observe", "taskProgress": "work",
}


@dataclass
class Decision:
    action: str                       # walk | drink | eat | sit | work | clean | talk | idle
    target_zone: Optional[str] = None
    target_object: Optional[str] = None
    need: str = ""
    reason: str = ""
    score: float = 0.0


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _agent_center(agent, scene) -> tuple:
    z = scene.zones.get(agent.current_zone)
    return z.center if z else (0.0, 0.0)


def _crowding(scene, zone_id: str) -> float:
    return sum(1 for a in scene.agents.values() if a.current_zone == zone_id)


def _nearest_task(agent, scene):
    if not scene.tasks:
        return None
    ac = _agent_center(agent, scene)
    return min(scene.tasks, key=lambda t: _dist(ac, scene.zones[t["zone"]].center)
              if t["zone"] in scene.zones else 1e9)


def choose_action(agent, scene) -> Decision:
    needs = agent.needs
    ac = _agent_center(agent, scene)

    # GLOBAL LEAVE — the HIGHEST priority, overrides staff duty, directives and needs: EVERY agent heads
    # for the exit and leaves the building. Covers BOTH emergency evacuation (fire) and calm end_of_day
    # closing. Deterministic, so no one is left behind.
    if scene.is_global_leave():
        ez = scene.exit_zone_id()
        if ez:
            if agent.current_zone != ez:
                return Decision("walk", target_zone=ez, need="evacuate",
                                reason=f"{agent.name} evacuates to {ez}")
            return Decision("leave", target_zone=ez, need="evacuate",
                            reason=f"{agent.name} leaves through {ez}")

    # USER DIRECTIVE (social / "Morgan talk to Sam", "Drew go to the bar") — overrides normal behaviour
    # for this NAMED agent: walk to the directed zone, or settle once there. Highest priority.
    dz = scene.agent_directive(agent)
    if dz and dz in scene.zones:
        if agent.current_zone != dz:
            return Decision("walk", target_zone=dz, need="directive",
                            reason=f"{agent.name} follows the director's instruction toward {dz}")
        return Decision("idle", target_zone=dz, need="directive",
                        reason=f"{agent.name} is at {dz} as directed")

    # Staff respond to pending object events (dirty table, empty cup, spill...) first,
    # and otherwise stand by at their station rather than wandering on personal needs.
    if agent.role in STAFF_ROLES:
        if scene.tasks:
            task = _nearest_task(agent, scene)
            if task:
                zone, oid = task["zone"], task["object"]
                if agent.current_zone != zone:
                    return Decision("walk", target_zone=zone, need="duty",
                                    reason=f"{agent.name} (staff) heads to {zone} to service {oid}")
                return Decision("clean", target_object=oid, target_zone=zone, need="duty",
                                reason=f"{agent.name} (staff) services {oid}")
        return Decision("idle", target_zone=agent.current_zone, need="duty",
                        reason=f"{agent.name} (staff) on standby")

    field, pressure = top_need(needs)
    best: Optional[Decision] = None
    best_score = -1e9

    # 1) objects whose affordance reduces the top need (a concrete affordance always beats
    #    merely standing in a zone -> +COMPLETE_BONUS when the agent can act right now).
    COMPLETE_BONUS = 100.0
    for obj in scene.objects.values():
        zc = scene.zones[obj.zone_id].center if obj.zone_id in scene.zones else ac
        for aff in obj.available_affordances():
            if scene.action_disabled(aff.action, obj.zone_id):   # an event switched this off
                continue
            # benefit = how much the effect REDUCES this need's pressure. energy is inverted
            # (higher = better), so a positive energy effect is beneficial; others invert.
            raw = float(aff.need_effects.get(field, 0))
            benefit = raw if field == "energy" else -raw
            if benefit <= 0:
                continue
            in_zone = agent.current_zone == obj.zone_id
            score = benefit - 0.3 * _dist(ac, zc) - 1.0 * _crowding(scene, obj.zone_id) \
                + (COMPLETE_BONUS if in_zone else 0.0) + scene.zone_bias(agent, obj.zone_id)
            if score > best_score:
                best_score = score
                if in_zone:
                    best = Decision(aff.action, target_object=obj.id, target_zone=obj.zone_id,
                                    need=field, score=score,
                                    reason=f"{field}={getattr(needs, field):.0f} -> {aff.action} {obj.id}")
                else:
                    best = Decision("walk", target_zone=obj.zone_id, target_object=obj.id,
                                    need=field, score=score,
                                    reason=f"{field}={getattr(needs, field):.0f} -> go to {obj.zone_id} to {aff.action}")

    # 2) fall back to a zone whose function matches the need (only wins when no object does)
    for zone in scene.zones.values():
        if not (set(zone.zone_function) & set(NEED_ZONEFUNC.get(field, []))):
            continue
        act = NEED_ACTION.get(field)
        if act and scene.action_disabled(act, zone.id):    # a patch closed this zone for this need
            continue
        # + zone_bias so a patch's NEGATIVE attraction (repulsion) pushes agents away from a
        #   hazard/closed zone even when its function still nominally matches the need.
        score = 20.0 - 0.3 * _dist(ac, zone.center) - 1.0 * _crowding(scene, zone.id) \
            + scene.zone_bias(agent, zone.id)
        if score > best_score:
            best_score = score
            if agent.current_zone == zone.id:
                best = Decision("idle", target_zone=zone.id, need=field, score=score,
                                reason=f"{field}: settled in {zone.label}")
            else:
                best = Decision("walk", target_zone=zone.id, need=field, score=score,
                                reason=f"{field}={getattr(needs, field):.0f} -> {zone.label}")

    # 2b) EVENT ATTRACTION: a user patch can pull agents to a zone regardless of need (this is how
    #     "there's a party at the pool" redirects the crowd). A genuinely urgent physiological need
    #     still outscores it (the object pass above already added COMPLETE_BONUS for those).
    for zone in scene.zones.values():
        bias = scene.zone_bias(agent, zone.id)
        if bias <= 0:
            continue
        score = bias - 0.3 * _dist(ac, zone.center) - 0.5 * _crowding(scene, zone.id)
        if score > best_score:
            best_score = score
            if agent.current_zone == zone.id:
                best = Decision("idle", target_zone=zone.id, need="event", score=score,
                                reason=f"{agent.name} joins the event at {zone.label}")
            else:
                best = Decision("walk", target_zone=zone.id, need="event", score=score,
                                reason=f"{agent.name} heads to {zone.label} (event)")

    # 3) nothing afforded -> explore
    if best is None:
        zid = next(iter(scene.zones), "")
        best = Decision("walk", target_zone=zid, need="curiosity", reason="exploring")
    return best
