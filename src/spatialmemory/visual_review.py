"""Manual visual findings, kept separate from automated inference and 3D proof."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VisualFinding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = Field(min_length=1)
    change: Literal['ADDED', 'REMOVED', 'MOVED']
    observation: str = Field(min_length=1)
    before_frames: list[str] = Field(min_length=1)
    after_frames: list[str] = Field(min_length=1)

    @field_validator('before_frames', 'after_frames')
    @classmethod
    def role_matches(cls, values, info):
        role = 'before:' if info.field_name == 'before_frames' else 'after:'
        if any(not value.startswith(role) for value in values):
            raise ValueError('Evidence frame has incorrect recording role')
        return values


class VisualReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: Literal['manual_visual_review'] = 'manual_visual_review'
    reviewer: str = Field(min_length=1)
    automated_prediction: Literal[False] = False
    geometry_verified: Literal[False] = False
    findings: list[VisualFinding]


def read_review(run: Path) -> dict | None:
    path = run / 'reports/manual_visual_review.json'
    if not path.exists():
        return None
    if not path.resolve().is_relative_to(run.resolve()):
        raise ValueError('Visual review escapes run directory')
    return VisualReview.model_validate_json(path.read_text()).model_dump()
