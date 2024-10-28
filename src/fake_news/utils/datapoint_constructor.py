from typing import Optional

from pydantic import BaseModel


class Datapoint(BaseModel):
    id: Optional[str]
    statement_json: Optional[str]
    label: Optional[bool]
    statement: str
    subject: Optional[str]
    speaker: Optional[str]
    speaker_title: Optional[str]
    state_info: Optional[str]
    party_affiliation: Optional[str]
    barely_true_count: float
    false_count: float
    half_true_count: float
    mostly_true_count: float
    pants_fire_count: float
    context: Optional[str]
    justification: Optional[str]
