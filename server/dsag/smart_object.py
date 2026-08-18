"""
Smart objects — the core novelty.

An object is a lightweight need/affordance entity (Gibson affordances + "smart objects"):
it advertises what actions it affords, what those actions do to an agent's needs, how its
own state changes, and what events it emits back into the simulation. Agents and objects
co-simulate on the same tick, so object state produces emergent, traceable cascades
(cup emptied -> emits needs_clearing -> staff dispatched -> congestion -> stress).

Policies (state machines) are LLM-GENERATED ONCE at scene creation and then run purely
by this rule engine per tick — no per-tick LLM calls. Conditions use a small, safe,
key-based DSL (no eval) so LLM output is constrained and auditable.
"""

from dataclasses import dataclass, field
from typing import Optional, ClassVar


# ── Events emitted by objects into the tick loop ─────────────────────────────
@dataclass
class Event:
    type: str                 # a vocabulary.EVENT_TYPES value, or a script/need signal
    source_id: str            # object (or agent) that emitted it
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"type": self.type, "source_id": self.source_id, "data": self.data}


# ── Safe condition DSL ───────────────────────────────────────────────────────
# A condition is a dict of key->value, all ANDed. Supported keys (LLM constrained to these):
#   state / not_state            : current object state equals / not equals
#   usage_count_gte              : total times any affordance was used >= n
#   idle_ticks_gte               : ticks since last interaction >= n
#   state_ticks_gte              : ticks spent in the current state >= n
#   occupied / not_occupied      : someone is / isn't using the object
def _check(cond: dict, ctx: dict) -> bool:
    for key, val in cond.items():
        if key == "state" and ctx["state"] != val:                     return False
        elif key == "not_state" and ctx["state"] == val:               return False
        elif key == "usage_count_gte" and ctx["usage_count"] < val:    return False
        elif key == "idle_ticks_gte" and ctx["idle_ticks"] < val:      return False
        elif key == "state_ticks_gte" and ctx["state_ticks"] < val:    return False
        elif key == "occupied" and not ctx["occupied"]:                return False
        elif key == "not_occupied" and ctx["occupied"]:                return False
    return True


# ── Affordance: an action the object offers an agent ─────────────────────────
@dataclass
class Affordance:
    action: str                              # the SUPPORTED ECGP semantic action the policy reasons over
    requires_state: Optional[str] = None     # object must be in this state to offer it
    changes_state_to: Optional[str] = None   # object transitions to this after use
    need_effects: dict = field(default_factory=dict)  # deltas applied to the AGENT's needs
    duration_ticks: int = 1
    variant: Optional[str] = None            # Unity execution VARIANT of the semantic action (e.g. the
                                             # catalog verb "pay"/"buy"/"wash" whose semantic action is
                                             # eat/relieve). The MODEL never sees it — it is only carried to
                                             # Unity for rendering, so we never claim the policy predicts it.

    def available_in(self, state: str) -> bool:
        return self.requires_state is None or self.requires_state == state

    @classmethod
    def from_dict(cls, d: dict) -> "Affordance":
        return cls(
            action=d["action"],
            requires_state=d.get("requires_state"),
            changes_state_to=d.get("changes_state_to"),
            need_effects=dict(d.get("need_effects", {})),
            duration_ticks=int(d.get("duration_ticks", 1)),
        )


# ── PolicyRule: an object's own state-machine transition (LLM-generated) ──────
@dataclass
class PolicyRule:
    when: dict                               # condition DSL over the object's context
    set_state: Optional[str] = None          # transition the object to this state
    emit: Optional[str] = None               # emit an Event of this type
    once: bool = True                        # fire at most once per entry into the matching state

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyRule":
        return cls(
            when=dict(d.get("when", {})),
            set_state=d.get("set_state"),
            emit=d.get("emit"),
            once=bool(d.get("once", True)),
        )


# ── SmartObject ──────────────────────────────────────────────────────────────
@dataclass
class SmartObject:
    id: str
    object_type: str                         # asset key, e.g. "coffee_cup"
    zone_id: str
    states: list[str]
    state: str
    affordances: list[Affordance] = field(default_factory=list)
    policy: list[PolicyRule] = field(default_factory=list)
    pos: tuple[float, float] = (0.0, 0.0)
    # What a HUMAN calls this object ("Coffee Machine"). `object_type` is the ASSET key the level was built
    # from, and in a hand-built LimeZu level that is a file name ("Kitchen_Singles_48x48_186") — which is what
    # used to leak into the UI. Authored per object in the scene config; "" => callers prettify object_type.
    display_name: str = ""

    # runtime bookkeeping
    usage_count: int = 0
    idle_ticks: int = 0
    state_ticks: int = 0
    occupied_by: Optional[str] = None
    _fired: set = field(default_factory=set)   # policy rules already fired this state-entry

    # ── ECGP capacity / reservation (SCHEMA_ECGP.md §9) ──────────────────────
    # Backward-compatible: default capacity=1, occupancy=0 → the legacy single-occupant path
    # (occupied_by + begin_use/finish_use) is unchanged. The lifecycle below is ADDITIVE, used by
    # the ECGP encoder (capacity_ok feature) and the rollout teacher's transition model.
    capacity: int = 1
    occupancy: int = 0
    reservations: dict = field(default_factory=dict)   # token -> [agent_id, ttl_left]
    _resv_seq: int = 0
    RESV_TTL: ClassVar[int] = 8                         # ticks an unused reservation is held

    # ── reversible removed/unavailable lifecycle (graph-EDIT remove_object) ───
    # A broken glass / cordoned exhibit is taken OUT of service, NOT deleted: the node keeps its stable id
    # and provenance, all affordances go dark, reservations/occupancy are released, and it is excluded from
    # candidate options — then restore() puts it back when the owning patch clears. See dsag.patch.
    removed: bool = False
    removed_by: Optional[str] = None                   # provenance: display_name of the patch that removed it
    _state_before_removal: Optional[str] = None

    def mark_removed(self, source: Optional[str] = None) -> None:
        """Reversibly take the object out of service. Idempotent. Preserves id, records provenance, cancels
        every reservation and frees occupancy, and (via `removed`) disables all affordances."""
        if self.removed:
            return
        self._state_before_removal = self.state
        self.removed = True
        self.removed_by = source
        self.reservations.clear()                       # cancel/release held slots — nobody holds a dead object
        self.occupancy = 0
        self.occupied_by = None
        if "removed" not in self.states:
            self.states.append("removed")
        self.state = "removed"
        self.state_ticks = 0

    def restore(self) -> None:
        """Put a removed object back in service (owning patch cleared). Returns to its pre-removal state."""
        if not self.removed:
            return
        self.removed = False
        self.removed_by = None
        self.state = self._state_before_removal or (self.states[0] if self.states else "default")
        self._state_before_removal = None
        self.state_ticks = 0
        self._fired.clear()

    def slots_free(self) -> int:
        if self.removed:
            return 0
        return self.capacity - self.occupancy - len(self.reservations)

    def capacity_ok(self) -> bool:
        """True if a new agent could occupy/reserve a slot right now (§4 `available` gate)."""
        return self.slots_free() > 0

    def reserve(self, agent_id: str) -> Optional[str]:
        """Hold a slot for an agent walking toward the object. Returns a token, or None if full."""
        if self.slots_free() <= 0:
            return None
        self._resv_seq += 1
        token = f"{self.id}#r{self._resv_seq}"
        self.reservations[token] = [agent_id, self.RESV_TTL]
        return token

    def claim(self, agent_id: str, token: Optional[str] = None) -> bool:
        """Begin using a slot (SCHEMA §9 `begin_use(agent, token)`). Consumes a matching reservation,
        else takes a free slot. Returns False if invalid token or no slot free."""
        if token is not None:
            r = self.reservations.get(token)
            if not r or r[0] != agent_id:
                return False
            del self.reservations[token]
            self.occupancy += 1
            return True
        if self.slots_free() <= 0:
            return False
        self.occupancy += 1
        return True

    def complete(self, agent_id: str, aff: "Affordance" = None) -> dict:
        """Finish an in-progress use; apply need_effects. Slot stays occupied until release()."""
        self.usage_count += 1
        self.idle_ticks = 0
        if aff is not None and aff.changes_state_to and aff.changes_state_to != self.state:
            self._transition(aff.changes_state_to)
        return dict(aff.need_effects) if aff is not None else {}

    def release(self, agent_id: str) -> bool:
        """Free a slot after use. Returns False if the object wasn't occupied."""
        if self.occupancy <= 0:
            return False
        self.occupancy -= 1
        return True

    def cancel(self, token: str) -> bool:
        """Drop a reservation without using it. Returns False for an unknown token."""
        return self.reservations.pop(token, None) is not None

    def expire_reservations(self) -> int:
        """Decrement reservation TTLs; drop any that hit zero (abandoned). Returns # dropped."""
        dropped = 0
        for tok in list(self.reservations):
            self.reservations[tok][1] -= 1
            if self.reservations[tok][1] <= 0:
                del self.reservations[tok]
                dropped += 1
        return dropped

    # ── queries ──────────────────────────────────────────────────────────────
    def available_affordances(self) -> list[Affordance]:
        """Affordances valid given the current state (and not currently occupied). A removed/unavailable
        object offers nothing until it is restored."""
        if self.removed or self.occupied_by is not None:
            return []
        return [a for a in self.affordances if a.available_in(self.state)]

    def affords(self, action: str) -> bool:
        return any(a.action == action for a in self.available_affordances())

    # ── interaction ──────────────────────────────────────────────────────────
    def begin_use(self, agent_id: str, action: str) -> Optional[Affordance]:
        """Reserve the object for an agent's action; returns the Affordance or None if invalid."""
        for a in self.available_affordances():
            if a.action == action:
                self.occupied_by = agent_id
                return a
        return None

    def finish_use(self, aff: Affordance) -> dict:
        """Apply the affordance's outcome (state change + usage); return need_effects for the agent."""
        self.usage_count += 1
        self.idle_ticks = 0
        self.occupied_by = None
        if aff.changes_state_to and aff.changes_state_to != self.state:
            self._transition(aff.changes_state_to)
        return dict(aff.need_effects)

    # ── per-tick policy engine ───────────────────────────────────────────────
    def tick(self) -> list[Event]:
        """Advance the object's own state machine; return any events it emits this tick."""
        self.state_ticks += 1
        if self.occupied_by is None:
            self.idle_ticks += 1

        ctx = {
            "state": self.state, "usage_count": self.usage_count,
            "idle_ticks": self.idle_ticks, "state_ticks": self.state_ticks,
            "occupied": self.occupied_by is not None,
        }
        events: list[Event] = []
        for i, rule in enumerate(self.policy):
            # Already in the rule's target state -> nothing to do (prevents a transition
            # rule whose condition stays true from re-firing every tick after it fires).
            if rule.set_state and self.state == rule.set_state:
                continue
            if rule.once and i in self._fired:
                continue
            if _check(rule.when, ctx):
                if rule.emit:
                    events.append(Event(type=rule.emit, source_id=self.id,
                                        data={"zone": self.zone_id, "object_type": self.object_type}))
                self._fired.add(i)
                if rule.set_state and rule.set_state != self.state:
                    self._transition(rule.set_state)
                    ctx["state"] = self.state   # reflect for subsequent rules this tick
        return events

    def reset(self):
        """Return the object to its initial state (e.g. staff refills a cup / cleans a table)."""
        self.occupied_by = None
        self.usage_count = 0
        self._transition(self.states[0])

    def _transition(self, new_state: str):
        if new_state not in self.states:
            self.states.append(new_state)   # tolerate LLM-introduced states
        self.state = new_state
        self.state_ticks = 0
        self._fired.clear()                 # `once` rules may fire again in the new state

    # ── serialization ────────────────────────────────────────────────────────
    def render_state(self) -> dict:
        """Compact state the engine needs to render/sync this object. `available`/`removed` drive Unity's
        reversible hide/show (a removed object is hidden, not destroyed, so it can reappear on restore)."""
        return {"id": self.id, "object_type": self.object_type, "zone": self.zone_id,
                "display_name": self.display_name,     # "" => Unity prettifies object_type itself
                "state": self.state, "occupied": self.occupied_by is not None,
                "available": not self.removed, "removed": self.removed,
                "removed_by": self.removed_by,
                "stock": getattr(self, "stock", -1)}   # -1 = not a stocked source (Unity draws N portions)

    @classmethod
    def from_dict(cls, d: dict) -> "SmartObject":
        states = list(d.get("states", []))
        so = d.get("smart_object", d)
        states = list(so.get("states", states)) or ["default"]
        return cls(
            id=d["id"],
            object_type=d.get("object_type", d.get("prop_type", "prop")),
            zone_id=d.get("zone_id", d.get("zone", "")),
            states=states,
            state=d.get("state", states[0]),
            affordances=[Affordance.from_dict(a) for a in so.get("affordances", [])],
            policy=[PolicyRule.from_dict(p) for p in so.get("policy", [])],
            pos=tuple(d.get("pos", (0.0, 0.0))),
            display_name=str(d.get("display_name", "") or ""),
        )
