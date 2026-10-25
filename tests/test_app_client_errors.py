from fastapi import HTTPException
from spatialmemory.app import client_error, prune_job_map, MAX_JOB_MAP


def test_client_error_redacts_unexpected_exceptions():
    assert client_error(RuntimeError("/secret/path boom")) == "Processing failed; see server logs"
    assert "missing video" in client_error(ValueError("missing video")).lower()
    assert client_error(HTTPException(400, "Empty video")) == "Empty video"


def test_prune_job_map_keeps_active_jobs():
    mapping = {f"{i:032x}": {"status": "complete"} for i in range(MAX_JOB_MAP + 5)}
    mapping["a" * 32] = {"status": "running"}
    prune_job_map(mapping)
    assert len(mapping) <= MAX_JOB_MAP
    assert mapping.get("a" * 32, {}).get("status") == "running"
