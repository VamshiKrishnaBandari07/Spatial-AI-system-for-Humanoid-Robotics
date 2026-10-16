"""Classify how real local-VLM outputs failed, using the committed raw outputs.

The fixtures are the actual model narratives saved in
reports/local_structured_visual_validation.json (no answer key). These tests pin
failure *diagnosis*; they say nothing about model quality beyond those two runs.
"""

import json
from pathlib import Path

import pytest

from spatialmemory.local_visual import validate_local_output

REPORT = Path(__file__).resolve().parents[1] / "reports" / "local_structured_visual_validation.json"
EVIDENCE = [{"id": f"{role}:{t}", "role": role} for role in ("before", "after") for t in (0, 2761, 5500, 8200)]


def narrative(model_fragment):
    runs = json.loads(REPORT.read_text())["runs"]
    return next(r["narrative"] for r in runs if r["model"].endswith(model_fragment))


@pytest.mark.skipif(not REPORT.exists(), reason="committed report unavailable")
def test_500m_header_echo_is_classified_as_not_json_and_yields_no_proposals():
    result = validate_local_output(narrative("SmolVLM-500M-Instruct"), EVIDENCE, False)
    assert result["status"] == "invalid_structured_output" and result["changes"] == []
    assert result["failure_category"] == "not_json"


@pytest.mark.skipif(not REPORT.exists(), reason="committed report unavailable")
def test_2b_invented_keys_and_ids_are_classified_as_schema_violation():
    result = validate_local_output(narrative("SmolVLM-Instruct"), EVIDENCE, False)
    assert result["status"] == "invalid_structured_output" and result["changes"] == []
    assert result["failure_category"] == "schema_violation"
    assert {"image_id", "frame_id"} <= set(result["validation_error_fields"])
    assert "1234567890" not in json.dumps(result["changes"])  # invented id never accepted


def valid(before="before:0", after="after:0", change="ADDED"):
    return json.dumps({"changes": [{
        "label": "white bottle", "change": change, "before_frames": [before], "after_frames": [after],
        "observation": "visible in AFTER only", "visibility_evidence": "empty shelf in BEFORE",
        "uncertainty": "single view"}], "limitations": "visual proposal only"})


def test_well_formed_response_with_supplied_frames_is_accepted_as_unverified_proposal():
    result = validate_local_output(valid(), EVIDENCE, False)
    assert result["status"] == "visual_proposals" and len(result["changes"]) == 1
    assert result["physical_3d_verified"] is False and result["calibrated"] is False


@pytest.mark.parametrize("before,after", [("before:9999", "after:0"), ("after:0", "after:0"), ("before:0", "before:0")])
def test_invented_or_wrong_role_frames_are_rejected_and_categorised(before, after):
    result = validate_local_output(valid(before, after), EVIDENCE, False)
    assert result["status"] == "invalid_structured_output" and result["changes"] == []
    assert result["failure_category"] == "invented_frame_reference"


def test_truncated_output_is_never_parsed_even_if_it_looks_valid():
    result = validate_local_output(valid(), EVIDENCE, True)
    assert result["status"] == "invalid_structured_output" and result["changes"] == []
