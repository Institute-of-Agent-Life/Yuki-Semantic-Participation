"""Validated runtime tuning for non-request autonomous opportunities.

Values are deliberately separate from persisted controller state. Replacing a
profile changes future sampling while leaving accepted Work and receipts intact.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutonomyParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intrinsic_interval_seconds: float = Field(default=600, ge=1, le=86400)
    human_activity_decay_seconds: float = Field(default=172800, gt=0, le=604800)
    human_activity_half_saturation: float = Field(default=80, gt=0, le=100000)
    silence_rise_seconds: float = Field(default=600, gt=0, le=604800)
    silence_decay_seconds: float = Field(default=64800, gt=0, le=604800)

    pressure_bias: float = Field(default=0.2, ge=-20, le=20)
    tendency_weight: float = Field(default=0.8, ge=0, le=20)
    activity_weight: float = Field(default=0.5, ge=0, le=20)
    compute_weight: float = Field(default=0.3, ge=0, le=20)
    no_reply_weight: float = Field(default=0.8, ge=0, le=20)
    work_fast_weight: float = Field(default=1.8, ge=0, le=20)
    work_slow_weight: float = Field(default=0.25, ge=0, le=20)
    exposure_weight: float = Field(default=0.55, ge=0, le=20)
    reception_weight: float = Field(default=0.5, ge=0, le=20)

    source_interval_seconds: float = Field(default=45, ge=1, le=86400)
    source_participation_minimum: float = Field(default=0.15, ge=0, le=1)
    source_familiarity_weight: float = Field(default=0.15, ge=0, le=20)
    source_invitation_weight: float = Field(default=0.35, ge=0, le=20)
    source_extension_weight: float = Field(default=0.25, ge=0, le=20)
    source_self_weight: float = Field(default=0.15, ge=0, le=20)
    source_closed_weight: float = Field(default=0.70, ge=0, le=20)
    source_social_weight: float = Field(default=2.0, ge=0, le=20)
    source_opening_seconds: float = Field(default=4, gt=0, le=86400)
    source_opening_floor: float = Field(default=0.1, ge=0, le=1)
    continuation_review_span: int = Field(default=4, ge=1)

    activity_decay_seconds: float = Field(default=60, gt=0, le=604800)
    activity_increment: float = Field(default=0.1, ge=0, le=1)
    social_context_decay_seconds: float = Field(default=14400, gt=0, le=604800)
    social_context_increment: float = Field(default=0.1, ge=0, le=1)
    compute_decay_seconds: float = Field(default=180, gt=0, le=604800)
    compute_increment: float = Field(default=0.25, ge=0, le=1)
    no_reply_decay_seconds: float = Field(default=3600, gt=0, le=604800)
    no_reply_increment: float = Field(default=0.45, ge=0, le=1)
    work_fast_decay_seconds: float = Field(default=1800, gt=0, le=86400)
    work_slow_decay_seconds: float = Field(default=21600, gt=0, le=86400)
    exposure_rise_seconds: float = Field(default=1200, gt=0, le=604800)
    exposure_decay_seconds: float = Field(default=43200, gt=0, le=86400)
    reception_decay_seconds: float = Field(default=21600, gt=0, le=604800)
    willingness_decay_seconds: float = Field(default=1800, gt=0, le=604800)
    willingness_magnitude: float = Field(default=0.7, ge=0, le=1)
    conversation_source_decay_seconds: float = Field(default=90, gt=4, le=604800)
    unanchored_reception_weight: float = Field(default=0.02, ge=0, le=1)
    unanchored_reception_decay_seconds: float = Field(default=1800, gt=0, le=604800)

    @model_validator(mode="before")
    @classmethod
    def retire_seed_parameters(cls, value: object) -> object:
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if key
                not in {
                    "independent_source_factor",
                    "recall_source_decay_seconds",
                    "contact_source_decay_seconds",
                }
            }
        return value


DEFAULT_AUTONOMY_PARAMETERS = AutonomyParameters()
