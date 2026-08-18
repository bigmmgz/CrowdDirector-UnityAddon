"""
schema.py — frozen schema/vocab/relation versions + the dataset-record schema (single source of truth).

The full human-readable spec lives in ecgp/SCHEMA_ECGP.md (rev. 3.1, FROZEN). This module exposes
the version strings the dataset/records/checkpoints stamp, and the canonical record field list + a light
validator used by the data pipeline. Bump only alongside a schema change (there is none in this pass).
"""
class SchemaVersionError(AssertionError):
    """Raised SPECIFICALLY when a record's stamped (schema, vocab, rel) triple doesn't match what a
    validator expects — a distinct type from the generic AssertionError used for missing-field checks, so
    a test can assert exactly THIS failure mode fired (via isinstance), not "some AssertionError happened,
    possibly the test's own diagnostic one". Subclasses AssertionError so any existing caller that catches
    AssertionError broadly is unaffected. Carries `expected`/`actual` as real dict attributes (not just
    baked into the message string) so a caller/test can compare the exact stamps precisely."""
    def __init__(self, expected: dict, actual: dict):
        self.expected = dict(expected)
        self.actual = dict(actual)
        super().__init__(f"version mismatch: expected {self.expected}, got {self.actual}")


SCHEMA_VERSION = "ecgp-1.1"
VOCAB_VERSION = "ecgp-vocab-1.2"
REL_VERSION = "ecgp-rel-1.0"

# V2.1 (hierarchy-aware) stamped versions — a DISTINCT, incompatible triple from the frozen v2 above. A
# record/checkpoint stamped with these consumes OBJECT_ROLE_V2 (not OBJECT_CLASS_V1), RELATIONS_V2 (33
# relations, not 23), and the wider V2.1 option-feature tensor (see vocab.py OPTION_FEAT_DIM_V2). Nothing
# in the live system stamps these yet — no V2.1 checkpoint exists (see ecgp-v2.1-hierarchy-and-diagnostics
# memory) — this is the schema plumbing a future dataset/training pass will stamp its records with.
SCHEMA_VERSION_V21 = "ecgp-1.2"
VOCAB_VERSION_V21 = "ecgp-vocab-1.3"
REL_VERSION_V21 = "ecgp-rel-1.1"
VERSIONS_V21 = {"schema": SCHEMA_VERSION_V21, "vocab": VOCAB_VERSION_V21, "rel": REL_VERSION_V21}


def stamp_v21():
    return dict(VERSIONS_V21)


def validate_record_shape_v21(rec: dict):
    """The V2.1 analogue of validate_record_shape — same structural check, but raises SchemaVersionError
    (not a bare AssertionError) against the OLD v2 versions too (a V2 record must never be silently
    accepted as V2.1, and vice versa) — see SchemaVersionError's docstring for why the distinct type
    matters for testing this precisely."""
    for k in ("schema", "vocab", "rel", "split", "agent_id", "options", "teacher_label", "opt_feasible"):
        assert k in rec, f"missing field {k}"
    actual = {"schema": rec.get("schema"), "vocab": rec.get("vocab"), "rel": rec.get("rel")}
    if actual != VERSIONS_V21:
        raise SchemaVersionError(VERSIONS_V21, actual)
    n = len(rec["options"])
    assert len(rec["teacher_label"]) == n == len(rec["opt_feasible"])
    return True

# one JSONL line per decision (see REPO_PLAN.md §4 + SCHEMA_ECGP.md §10-§11)
RECORD_FIELDS = [
    "schema", "vocab", "rel", "teacher_config",
    "split", "scenario_family", "layout", "event_combo", "crowd_size", "seed",
    "scenario_id", "agent_id", "timestep",
    "n_nodes", "n_events", "n_operations", "active_patches",
    "options", "opt_feasible", "opt_returns", "teacher_label",
    "n_options", "n_feasible", "scene_scale",
]

VERSIONS = {"schema": SCHEMA_VERSION, "vocab": VOCAB_VERSION, "rel": REL_VERSION}


def stamp():
    """The version block every dataset/record/checkpoint carries."""
    return dict(VERSIONS)


def validate_record_shape(rec: dict):
    """Cheap structural check (full numeric validation lives in data/validator.py). Raises
    SchemaVersionError (not a bare AssertionError) specifically for a version mismatch — e.g. a V2.1
    record must never be silently accepted here as V2."""
    for k in ("schema", "vocab", "rel", "split", "agent_id", "options", "teacher_label", "opt_feasible"):
        assert k in rec, f"missing field {k}"
    actual = {"schema": rec.get("schema"), "vocab": rec.get("vocab"), "rel": rec.get("rel")}
    if actual != VERSIONS:
        raise SchemaVersionError(VERSIONS, actual)
    n = len(rec["options"])
    assert len(rec["teacher_label"]) == n == len(rec["opt_feasible"])
    return True
