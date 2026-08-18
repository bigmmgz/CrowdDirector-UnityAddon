"""
ecgp — Event-Conditioned Graph Policy: the CrowdSim learned crowd-behavior model.

Realizes the FROZEN schema in ecgp/SCHEMA_ECGP.md (rev. 3.1). Subpackages: graph/ (world, overlays,
options, encoder), teacher/ (complete headless simulator + short-horizon rollout teacher + reward +
grounded rule policy + need-pressure + social), data/ (generator, splits, serializer, validator,
dataset loader, collate), training/, evaluation/, runtime/ (inference + the live CrowdSim bridge).
The trained policy (64x3) lives in outputs/final_64x3/best.pt and drives agents via runtime/live_bridge.
"""

SCHEMA_VERSION = "ecgp-1.1"
VOCAB_VERSION = "ecgp-vocab-1.2"
REL_VERSION = "ecgp-rel-1.0"
