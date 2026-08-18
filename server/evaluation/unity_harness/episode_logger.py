"""
episode_logger.py — deterministic per-episode JSON logging for the Unity closed-loop harness (E3) and
the café smoke test. Append-only JSONL under evaluation/results/raw/episodes/; one file per episode,
one record per logged event, schema below. Never raises into the server path.

Episode lifecycle: begin_episode(...) → log_instruction / log_tick / log_event ... → end_episode(...).

Record schema (every record: {"t": <server tick or -1>, "kind": <str>, ...}):
  kind=episode_begin  {episode_id, scene_id, condition, n_agents, seed, config}
  kind=instruction    {instruction, support_status, gate, compiler_ops, unresolved, reason}
  kind=patch          {patch_ops, display_name, source}
  kind=tick           {actions: [{agent_id, action, target_id, semantic_origin, reason}], n_affected}
  kind=reinvoke       {agent_id, reason, prev_commitment}                (mirrors [ecgp/reinvoke])
  kind=conflict       {agent_id, object_id, kind}                        (capacity/reservation)
  kind=stuck          {agent_id, zone, ticks_stuck}
  kind=completion     {what, agent_id?, object_id?, latency_ticks}
  kind=episode_end    {outcome, metrics: {affected_response_rate, unaffected_stability,
                       completion_rate, response_latency_ticks, invalid_target_count,
                       capacity_conflicts, stuck_agents, alternative_or_exit_correct}}
"""
import json
import os
import time

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "episodes")
_state = {"fh": None, "episode_id": None, "tick": -1}


def _write(rec):
    try:
        if _state["fh"] is None:
            _open_default()
        rec.setdefault("t", _state["tick"])
        rec.setdefault("wall", round(time.time(), 3))
        _state["fh"].write(json.dumps(rec) + "\n")
        _state["fh"].flush()
    except Exception:
        pass                                                # logging must never break the server


def _open_default():
    os.makedirs(_DIR, exist_ok=True)
    eid = _state["episode_id"] or f"adhoc_{int(time.time())}"
    _state["episode_id"] = eid
    _state["fh"] = open(os.path.join(_DIR, eid + ".jsonl"), "a", encoding="utf-8")


def begin_episode(episode_id, scene_id, condition, n_agents, seed, config=None):
    end_episode(outcome="superseded") if _state["fh"] else None
    _state["episode_id"] = episode_id
    _state["tick"] = 0
    os.makedirs(_DIR, exist_ok=True)
    _state["fh"] = open(os.path.join(_DIR, f"{episode_id}.jsonl"), "w", encoding="utf-8")
    _write({"kind": "episode_begin", "episode_id": episode_id, "scene_id": scene_id,
            "condition": condition, "n_agents": n_agents, "seed": seed, "config": config or {}})


def set_tick(t):
    _state["tick"] = t


def log_instruction(instruction, verdict):
    _write({"kind": "instruction", "instruction": instruction,
            "support_status": verdict.get("support_status"), "gate": verdict.get("gate"),
            "compiler_ops": verdict.get("operations", []),
            "unresolved": verdict.get("unresolved", []), "reason": verdict.get("reason", "")})


def log_patch(patch_ops, display_name, source):
    _write({"kind": "patch", "patch_ops": patch_ops, "display_name": display_name, "source": source})


def log_tick(actions):
    compact = [{"agent_id": a.get("agent_id"), "action": a.get("action"),
                "target_id": a.get("target_id") or a.get("zone_id") or a.get("object_id"),
                "semantic_origin": a.get("semantic_origin"), "reason": a.get("reason")}
               for a in actions]
    _write({"kind": "tick", "actions": compact, "n_actions": len(compact)})


def log_reinvoke(agent_id, reason, prev_commitment):
    _write({"kind": "reinvoke", "agent_id": agent_id, "reason": reason,
            "prev_commitment": prev_commitment})


def log_conflict(agent_id, object_id, kind):
    _write({"kind": "conflict", "agent_id": agent_id, "object_id": object_id, "conflict": kind})


def log_stuck(agent_id, zone, ticks_stuck):
    _write({"kind": "stuck", "agent_id": agent_id, "zone": zone, "ticks_stuck": ticks_stuck})


def log_completion(what, agent_id=None, object_id=None, latency_ticks=None):
    _write({"kind": "completion", "what": what, "agent_id": agent_id, "object_id": object_id,
            "latency_ticks": latency_ticks})


def end_episode(outcome="done", metrics=None):
    if _state["fh"] is None:
        return
    _write({"kind": "episode_end", "outcome": outcome, "metrics": metrics or {}})
    try:
        _state["fh"].close()
    except Exception:
        pass
    _state.update({"fh": None, "episode_id": None, "tick": -1})
