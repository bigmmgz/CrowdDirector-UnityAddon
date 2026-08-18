"""
Decision provenance logging.

Every per-tick decision is recorded with its causal chain (agent -> need -> action ->
target), which is (a) the "traceable" contribution shown in the paper, (b) the source of
evaluation metrics, and (c) the training data for the Tier-1 GNN distillation.
"""

import json
from dataclasses import dataclass, asdict


@dataclass
class TraceRecord:
    tick: int
    agent_id: str
    agent_name: str
    action: str
    target: str
    need: str
    reason: str
    score: float


class TraceLog:
    def __init__(self):
        self.records: list[TraceRecord] = []

    def record(self, tick: int, agent, decision):
        self.records.append(TraceRecord(
            tick=tick, agent_id=agent.id, agent_name=agent.name,
            action=decision.action,
            target=decision.target_object or decision.target_zone or "",
            need=decision.need, reason=decision.reason, score=round(decision.score, 2),
        ))

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(r)) for r in self.records)

    def write(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())
