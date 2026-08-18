"""
chain_state.py — V2.1 section 6/8 (Gate 2): explicit macro-chain state, NOT inferred primarily from
object reservations. A single shared vocabulary (`ChainStage`) and record (`ChainState`) that every real
executor of a cluster-style macro-chain ("eat@dining_cluster_root": approach provider -> acquire ->
approach seat -> seated -> consume -> complete) constructs and advances as it actually runs the chain:

  - `ecgp.teacher.simulator` (the offline rollout stepper) sets/advances `agent.chain_state` as it
    executes a fixed or rule-chosen cluster-root intent tick by tick.
  - `ecgp.runtime.live_bridge` (the live per-tick server loop) does the same for the real running scene.

`ecgp.teacher.reward._chain_stage_and_distance` reads `agent.chain_state` as its PRIMARY signal (falling
back to the older reservation-based inference only when no chain_state is present for this cluster — e.g.
a hand-built unit-test world that never runs a real executor). Eval logs and the Unity wire protocol
(`DirectorAction.stage`/`failure_reason` in SceneSpec.cs) both carry the same field names, so a single
struct is the one source of truth end to end.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChainStage(str, Enum):
    NONE = "NONE"                          # no active macro-chain
    APPROACH_PROVIDER = "APPROACH_PROVIDER"  # walking to / ordering at the provider
    ACQUIRED = "ACQUIRED"                    # got the item from the provider
    APPROACH_SEAT = "APPROACH_SEAT"          # walking to the reserved seat
    SEATED = "SEATED"                        # arrived, not yet consuming
    CONSUMING = "CONSUMING"                  # dwelling at the seat, need-effect not yet applied
    COMPLETED = "COMPLETED"                  # need effect applied, seat released — success
    FAILED = "FAILED"                        # aborted by something outside the agent's control
    CANCELLED = "CANCELLED"                  # aborted by an external override (emergency, directive)


TERMINAL_STAGES = frozenset({ChainStage.COMPLETED, ChainStage.FAILED, ChainStage.CANCELLED})


@dataclass
class ChainState:
    """One agent's progress through one macro-chain instance. `stage_started_tick`/`stage_completed_tick`
    describe the CURRENT `stage` value's own lifecycle: started_tick is when the agent entered `stage`;
    completed_tick is set the instant `advance()` moves AWAY from it (so it also directly answers "when did
    the chain conclude" once `stage` is a terminal value)."""
    chain_id: str
    semantic_goal: str
    semantic_target: str
    cluster_id: Optional[str] = None
    stage: ChainStage = ChainStage.NONE
    provider_id: Optional[str] = None
    acquired_item_id: Optional[str] = None
    seat_id: Optional[str] = None
    slot_id: Optional[str] = None            # the reservation/claim token for the held seat slot
    stage_started_tick: Optional[int] = None
    stage_completed_tick: Optional[int] = None
    failure_reason: Optional[str] = None

    def advance(self, stage: ChainStage, tick: int, failure_reason: Optional[str] = None) -> None:
        """Move to `stage` at `tick`. No-op if already there, so a per-tick re-check with an unchanged
        stage doesn't reset stage_started_tick every call."""
        if stage == self.stage:
            return
        self.stage_completed_tick = tick
        self.stage = stage
        self.stage_started_tick = tick
        if failure_reason is not None:
            self.failure_reason = failure_reason

    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES
