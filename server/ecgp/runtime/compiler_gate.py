"""
compiler_gate.py — server-side wiring of the capability-constrained compiler (evaluation engineering;
frozen scope). A PRE-PASS on every free-text director instruction in dsag mode:

    instruction → live EcgpWorld snapshot (live_bridge.build_world) → capability_compiler
        → {support_status, grounded ops, scope, resolved/unresolved, reason}

Contract (the capability gate): UNSUPPORTED / AMBIGUOUS / INVALID_REFERENCE instructions are REJECTED
here — no patch is applied, the reason is reported back. EXECUTABLE / PARTIALLY_EXECUTABLE proceed to
the existing patch-synthesis path (PARTIALLY with its missing-capability report attached; nothing is
silently pretended to be fully realized). Set ECGP_COMPILER_GATE=0 to disable rejection (pre-pass still
logs its verdict for the episode trace).
"""
import logging
import os

log = logging.getLogger("ecgp.compiler_gate")

GATE_ENABLED = os.environ.get("ECGP_COMPILER_GATE", "1") != "0"
_REJECT_STATUSES = ("UNSUPPORTED", "AMBIGUOUS", "INVALID_REFERENCE")


def precheck(description, scene, zone_dicts):
    """Run the capability compiler against the LIVE scene. Returns the compiler verdict dict, plus
    'gate': 'pass'|'reject' per the contract above. Never raises — a pre-pass failure must not take
    down the director path (returns gate='pass', status='PRECHECK_ERROR' so the trace shows it)."""
    try:
        from . import live_bridge
        from .capability_compiler import compile_instruction
        world = live_bridge.build_world(scene, zone_dicts)
        res = compile_instruction(description, world)
        verdict = res.as_dict()
        verdict["gate"] = ("reject" if GATE_ENABLED and res.support_status in _REJECT_STATUSES
                           else "pass")
        log.info(f"[compiler_gate] '{description[:60]}' -> {res.support_status} gate={verdict['gate']} "
                 f"ops={len(res.operations)} unresolved={len(res.unresolved)} reason={res.reason[:80]}")
        return verdict
    except Exception as e:                                   # noqa: BLE001 — deliberate never-break guard
        log.warning(f"[compiler_gate] precheck failed ({e}); passing through")
        return {"support_status": "PRECHECK_ERROR", "gate": "pass", "operations": [],
                "affected_scope": {}, "resolved_targets": [], "unresolved": [], "reason": str(e)}
