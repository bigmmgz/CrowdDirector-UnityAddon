"""
overlays.py — patch provenance + reversible overlays (SCHEMA_ECGP.md rev. 3.1 §6).

effective_state = base_world_state ⊕ Σ(overlays owned by ACTIVE patches).

Each reversible effect stores the set of patch_ids that own it (reference counting), so clearing ONE
patch leaves a co-owner's effect active — there is NO blind snapshot restore. Durable mutations
(emotion_delta one-shots, relationship_set/delta) are NOT overlays: they are logged and applied to base
state and never reverted. Combination rules per effect (rev. 3.1 §6.2):
  hazard / blocked_edge / disabled_affordance : union + refcount (hazard severity = max owner)
  attraction / need_drift                     : sum over active owners
  object_state / directive                    : newest active owner wins
"""
from dataclasses import dataclass, field

# op kinds that are DURABLE base mutations (never reverted by event_clear)
DURABLE_OPS = ("emotion_delta", "relationship_set", "relationship_delta")


def _edge_key(a, b):
    return frozenset((a, b))


@dataclass
class OverlaySet:
    patches: dict = field(default_factory=dict)        # patch_id -> {"ttl", "evac"}
    _hazard: dict = field(default_factory=dict)        # zone_id -> {patch_id: severity}
    _blocked: dict = field(default_factory=dict)       # frozenset(a,b) -> set(patch_id)
    _disabled: dict = field(default_factory=dict)      # (target_id, action) -> set(patch_id)
    # Which of a disabled key's owning patches marked it permanent — "out of order indefinitely" as
    # against a "back soon" restock timer. Opt-in: a caller that omits `permanent` never gets an entry
    # here and is_permanently_disabled stays False.
    _disabled_permanent: dict = field(default_factory=dict)   # (target_id, action) -> bool
    _attraction: dict = field(default_factory=dict)    # target_id -> {patch_id: delta}
    _need_drift: dict = field(default_factory=dict)    # (scope, need) -> {patch_id: delta}
    _object_state: dict = field(default_factory=dict)  # object_id -> [(seq, patch_id, state)]
    _directive: dict = field(default_factory=dict)     # subject_id -> [(seq, patch_id, target)]
    durable_log: list = field(default_factory=list)    # applied one-shot / relationship mutations
    _seq: int = 0

    # ── application ──────────────────────────────────────────────────────────
    def apply_patch(self, patch_id: str, ops: list, ttl: int = 25, evac: bool = False) -> list:
        """Register a patch's ops as overlays (reversible) or durable mutations. Returns the durable
        mutations (caller applies them to base state)."""
        self.patches[patch_id] = {"ttl": int(ttl), "evac": bool(evac)}
        durables = []
        for op in ops:
            kind = op.get("op")
            if kind in DURABLE_OPS:
                mut = {"patch_id": patch_id, **op}
                self.durable_log.append(mut)
                durables.append(mut)
            else:
                self._apply_overlay(patch_id, op)
        return durables

    def _apply_overlay(self, patch_id, op):
        kind = op.get("op")
        if kind == "zone_hazard":
            sev = float(op.get("severity", op.get("delta", 0.8)) or 0.8)
            self._hazard.setdefault(op["zone"], {})[patch_id] = sev
        elif kind == "block_edge":
            self._blocked.setdefault(_edge_key(op["zone"], op.get("zone_b")), set()).add(patch_id)
        elif kind == "disable_affordance":
            key = (op.get("zone") or op.get("object"), op["action"])
            self._disabled.setdefault(key, set()).add(patch_id)
            if op.get("permanent"):
                self._disabled_permanent[key] = True
        elif kind in ("zone_attraction", "object_attraction"):
            tgt = op.get("zone") or op.get("object")
            self._attraction.setdefault(tgt, {})[patch_id] = float(op.get("delta", 0.0) or 0.0)
        elif kind == "need_rate":
            key = (op.get("role") or op.get("agent") or "all", op["need"])
            self._need_drift.setdefault(key, {})[patch_id] = float(op.get("delta", 0.0) or 0.0)
        elif kind == "object_state":
            self._seq += 1
            self._object_state.setdefault(op["object"], []).append((self._seq, patch_id, op["state"]))
        elif kind in ("agent_directive", "group_directive"):
            self._seq += 1
            subj = op.get("agent") or op.get("group")
            tgt = op.get("zone") or op.get("target_agent") or op.get("mode")
            self._directive.setdefault(subj, []).append((self._seq, patch_id, tgt))
        # event_clear is handled by the caller via clear_one/clear_all

    # ── queries over EFFECTIVE state (active patches only) ────────────────────
    def _active(self, pid):
        return pid in self.patches

    def is_hazard(self, zone_id) -> bool:
        return any(self._active(p) for p in self._hazard.get(zone_id, {}))

    def hazard_severity(self, zone_id) -> float:
        vals = [s for p, s in self._hazard.get(zone_id, {}).items() if self._active(p)]
        return max(vals) if vals else 0.0

    def is_blocked_edge(self, a, b) -> bool:
        return any(self._active(p) for p in self._blocked.get(_edge_key(a, b), set()))

    def is_disabled(self, target_id, action) -> bool:
        return any(self._active(p) for p in self._disabled.get((target_id, action), set()))

    def is_permanently_disabled(self, target_id, action) -> bool:
        """True for a disable_affordance patch marked `permanent`, as against an ordinary timed one that
        clears on its own TTL. Only meaningful while the disable is active: false once the owning
        patches clear, the same as is_disabled."""
        return self.is_disabled(target_id, action) and self._disabled_permanent.get((target_id, action), False)

    def attraction(self, target_id) -> float:
        return sum(d for p, d in self._attraction.get(target_id, {}).items() if self._active(p))

    def need_drift(self, scope, need) -> float:
        return sum(d for p, d in self._need_drift.get((scope, need), {}).items() if self._active(p))

    def object_state(self, object_id):
        active = [(seq, st) for seq, p, st in self._object_state.get(object_id, []) if self._active(p)]
        return max(active, key=lambda t: t[0])[1] if active else None   # newest active wins

    def directive(self, subject_id):
        active = [(seq, tgt) for seq, p, tgt in self._directive.get(subject_id, []) if self._active(p)]
        return max(active, key=lambda t: t[0])[1] if active else None

    def owners(self, effect: str, key) -> set:
        """Owner patch_ids for a refcounted effect (for tests/inspection)."""
        table = {"hazard": self._hazard, "blocked": self._blocked, "disabled": self._disabled}[effect]
        v = table.get(key)
        if v is None:
            return set()
        pids = set(v.keys()) if isinstance(v, dict) else set(v)
        return {p for p in pids if self._active(p)}

    # ── clearing (no blind restoration) ──────────────────────────────────────
    def clear_one(self, patch_id: str):
        """Remove one patch's ownership from every effect. Co-owned effects stay active."""
        self.patches.pop(patch_id, None)
        self._prune(patch_id)

    def clear_all(self):
        """All-clear: drop every active patch's overlays. Durable log is preserved."""
        for pid in list(self.patches):
            self.clear_one(pid)

    def _prune(self, patch_id):
        for tbl in (self._hazard, self._attraction, self._need_drift):
            for key in list(tbl):
                tbl[key].pop(patch_id, None)
                if not tbl[key]:
                    del tbl[key]
        for tbl in (self._blocked, self._disabled):
            for key in list(tbl):
                tbl[key].discard(patch_id)
                if not tbl[key]:
                    del tbl[key]
        for tbl in (self._object_state, self._directive):
            for key in list(tbl):
                tbl[key] = [t for t in tbl[key] if t[1] != patch_id]
                if not tbl[key]:
                    del tbl[key]

    def tick(self) -> list:
        """Decrement every active patch's ttl; clear_one any that expire. Returns expired patch_ids."""
        expired = []
        for pid, meta in list(self.patches.items()):
            meta["ttl"] -= 1
            if meta["ttl"] <= 0:
                expired.append(pid)
        for pid in expired:
            self.clear_one(pid)
        return expired

    def active_patch_ids(self) -> list:
        return list(self.patches)
