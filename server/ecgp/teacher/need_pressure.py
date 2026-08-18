"""
need_pressure.py — correct need-pressure directions (Stage-3 item A).

A need's raw 0..100 value is not the same as its behavioral PRESSURE (urgency to act):
  - thirst / hunger / bladder / stress:  pressure = value / 100        (high value ⇒ high pressure)
  - energy:                              pressure = 1 - value / 100    (LOW energy ⇒ high pressure)
Everything that ranks urgency or feeds the encoder's need channels must use pressure, not the raw value,
so a well-rested (high-energy) agent is not treated as urgently needing rest, and a tired (low-energy)
agent is. Non-drive needs (loneliness/groupAffinity/status/curiosity) use the direct value/100 form.
"""

# needs whose pressure is INVERTED (low value = high pressure)
_INVERTED = {"energy"}


def pressure(need: str, value: float) -> float:
    """Behavioral pressure in [0,1] for a raw 0..100 need value, with the correct per-need direction."""
    p = value / 100.0
    if need in _INVERTED:
        p = 1.0 - p
    return max(0.0, min(1.0, p))


def pressures(needs: dict) -> dict:
    return {n: pressure(n, v) for n, v in needs.items()}


def top_pressure_needs(needs: dict, k=2):
    """The k needs with the highest pressure (correct direction), most-urgent first."""
    return [n for n, _ in sorted(needs.items(), key=lambda kv: -pressure(kv[0], kv[1]))[:k]]
