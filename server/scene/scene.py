"""
SceneModel — the headless simulation engine that runs one tick of the whole loop:
update needs -> pick affordance-grounded actions -> apply agent<->object interactions ->
co-tick smart objects -> route emitted events into staff tasks -> log traces.

This is engine-agnostic: the WebSocket server drives it and Unity renders the result.
"""

from dataclasses import dataclass, field
from .world import AgentInstance, ZoneInstance
from .smart_object import SmartObject, Event
from .behavior import choose_action
from .trace import TraceLog
from . import patch as patchmod

# per-tick need drift (customers). Positive = grows more urgent; energy falls.
# Paced against the DEMO venue's real throughput: 28 customers vs 2 food + 2 drink sources and a
# 2-capacity toilet. At the old rates (thirst 6/tick) an agent needed a drink every ~minute, the venue
# could not keep up, every need saturated at 100 and STAYED there — which the (now server-synced) UI
# bars displayed honestly as an all-empty wall, and which kept half the crowd permanently urgent
# (urgency bypasses the decision stagger, so saturated needs also meant constant re-deciding).
NEED_RATES = {
    "thirst": 1.5, "hunger": 1.0, "bladder": 1.2, "energy": -0.8,
    "loneliness": 0.8, "curiosity": 0.8,
}
# events that turn into a staff service task, keyed on the emitting object
TASK_EVENTS = {"needs_cleaning", "needs_clearing", "needs_refill", "spill"}


class SceneModel:
    def __init__(self):
        self.zones: dict[str, ZoneInstance] = {}
        self.agents: dict[str, AgentInstance] = {}
        self.objects: dict[str, SmartObject] = {}
        self.tasks: list[dict] = []          # [{zone, object, event}]
        self.tick_no: int = 0
        self.trace = TraceLog()
        self.active_patches: list = []       # live ScenePatch list (user interruptions in effect)

    # ── construction ─────────────────────────────────────────────────────────
    def add_zone(self, z: ZoneInstance):     self.zones[z.id] = z
    def add_agent(self, a: AgentInstance):   self.agents[a.id] = a
    def add_object(self, o: SmartObject):    self.objects[o.id] = o

    # ── user-interruption patches (the redirection mechanism) ─────────────────
    def apply_patch(self, patch):
        """Register a ScenePatch and fire its one-shot need_shift ops immediately. The crowd then
        redirects by reading the patched world through choose_action (or the GNN) each tick."""
        self.active_patches.append(patch)
        # LEAVE-LIFECYCLE (item C): if this patch carries agent_leave ops, freeze WHO leaves right now,
        # against the CURRENT agent set — never re-resolved later against a shrinking/changing list (that
        # was the 'resolved to NOBODY' bug: role='family' stopped matching once members had already left).
        if any(op.op == "agent_leave" for op in patch.ops):
            patchmod.lock_leavers(patch, self)
        for op in patch.ops:
            if op.op == "need_shift" and op.need:
                for a in self.agents.values():
                    if patchmod.role_matches(op, a) and hasattr(a.needs, op.need):
                        v = max(0.0, min(100.0, getattr(a.needs, op.need) + op.delta))
                        setattr(a.needs, op.need, v)

    def zone_bias(self, agent, zone_id: str) -> float:
        z = self.zones.get(zone_id)
        return patchmod.zone_bias(self.active_patches, agent, z) if z else 0.0

    def action_disabled(self, action: str, zone_id: str) -> bool:
        z = self.zones.get(zone_id)
        return bool(z) and patchmod.action_disabled(self.active_patches, action, z, list(self.zones.values()))

    def agent_directive(self, agent) -> str:
        """Zone id this specific agent has been directed to (social/"X go to Y"), or None."""
        return patchmod.agent_directive_target(self.active_patches, agent, self)

    def is_emergency(self) -> bool:
        """A hard EMERGENCY evacuation is active (fire/gas) — panic, immediate leave."""
        return patchmod.is_evacuation(self.active_patches)

    def is_global_leave(self) -> bool:
        """The whole crowd should leave the building — emergency evacuation OR calm end_of_day closing."""
        return patchmod.is_global_leave(self.active_patches)

    def exit_zone_id(self) -> str:
        """The zone agents leave through: the 'access' zone (entrance/exit), by function then keyword."""
        for z in self.zones.values():
            if "access" in z.zone_function:
                return z.id
        for z in self.zones.values():
            blob = f"{z.id} {z.zone_type} {getattr(z, 'label', '')}".lower()
            if any(k in blob for k in ("entrance", "exit", "gate", "lobby", "door", "foyer")):
                return z.id
        return next(iter(self.zones), None)

    # ── one simulation tick ──────────────────────────────────────────────────
    def tick(self) -> list[Event]:
        self.tick_no += 1
        for p in self.active_patches:        # patches fade so the world relaxes after an event
            p.ttl -= 1
        self.active_patches = [p for p in self.active_patches if p.ttl > 0]
        self._update_needs()

        for agent in self.agents.values():
            d = choose_action(agent, self)
            self._apply(agent, d)
            self.trace.record(self.tick_no, agent, d)

        events: list[Event] = []
        for obj in self.objects.values():
            events += obj.tick()
        for e in events:
            self._route_event(e)
        return events

    # ── internals ────────────────────────────────────────────────────────────
    def _update_needs(self):
        from .behavior import STAFF_ROLES
        for a in self.agents.values():
            if a.role in STAFF_ROLES:
                continue                      # staff act on duty, not personal need drift
            for f, r in NEED_RATES.items():
                r += patchmod.need_rate_delta(self.active_patches, a, f)   # patches reshape drift
                v = max(0.0, min(100.0, getattr(a.needs, f) + r))
                setattr(a.needs, f, v)

    def _apply(self, agent: AgentInstance, d):
        agent.last_action = d.action
        if d.action == "walk":
            if d.target_zone:
                agent.current_zone = d.target_zone
        elif d.action == "clean" and d.target_object in self.objects:
            self.objects[d.target_object].reset()
            self.tasks = [t for t in self.tasks if t["object"] != d.target_object]
        elif d.target_object in self.objects:
            obj = self.objects[d.target_object]
            aff = obj.begin_use(agent.id, d.action)
            if aff:
                self._apply_effects(agent, obj.finish_use(aff))
        # idle -> no-op

    @staticmethod
    def _apply_effects(agent: AgentInstance, effects: dict):
        for f, dv in effects.items():
            if hasattr(agent.needs, f):
                setattr(agent.needs, f, max(0.0, min(100.0, getattr(agent.needs, f) + dv)))

    def _route_event(self, e: Event):
        if e.type in TASK_EVENTS:
            oid = e.source_id
            if not any(t["object"] == oid for t in self.tasks):
                self.tasks.append({"zone": e.data.get("zone", ""), "object": oid, "event": e.type})
