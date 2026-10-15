"""Core data contracts shared by every phase.

The `Incident` object is the single source of truth: the dataset factory samples
it first and renders text/audio from it, and the NLP pipeline (Phase 3) tries to
recover it from a transcript. Keeping one schema for both directions is what
makes evaluation possible.
"""

from __future__ import annotations

from enum import Enum, IntEnum

from pydantic import BaseModel, Field


class Language(str, Enum):
    EN = "en"
    HI = "hi"
    TA = "ta"
    ML = "ml"


class CallType(str, Enum):
    MAYDAY = "MAYDAY"  # grave and imminent danger
    PANPAN = "PAN-PAN"  # urgent, not immediately life-threatening


class IncidentType(str, Enum):
    FIRE = "fire"
    FLOODING = "flooding"
    ENGINE_FAILURE = "engine_failure"
    CAPSIZING = "capsizing"
    MAN_OVERBOARD = "man_overboard"
    MEDICAL = "medical"
    COLLISION = "collision"
    GROUNDING = "grounding"


class Severity(IntEnum):
    """Ordinal severity used by the Phase 4 CORAL model."""

    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class VesselType(str, Enum):
    FISHING_BOAT = "fishing_boat"
    TRAWLER = "trawler"
    CARGO = "cargo"
    TANKER = "tanker"
    FERRY = "ferry"
    YACHT = "yacht"
    TUG = "tug"


class PositionFormat(str, Enum):
    DDM = "ddm"  # degrees + decimal minutes, the SMCP default
    DMS = "dms"
    DECIMAL = "decimal"
    RELATIVE = "relative"  # bearing/range from a landmark


class Position(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Vessel(BaseModel):
    name: str
    vessel_type: VesselType
    mmsi: str | None = Field(default=None, pattern=r"^\d{9}$")
    call_sign: str | None = None


# Fields a coast guard operator needs before launching a response.
CRITICAL_FIELDS = ("vessel_name", "position", "incident_types", "pob")


class Incident(BaseModel):
    """Structured ground truth (factory) or prediction (pipeline)."""

    call_type: CallType
    vessel: Vessel
    position: Position
    position_format: PositionFormat
    incident_types: list[IncidentType] = Field(min_length=1)
    pob: int = Field(ge=1, description="Persons on board")
    injured: int = Field(default=0, ge=0)
    severity: Severity
    nearest_landmark: str


class EntityLabel(str, Enum):
    VESSEL = "VESSEL"
    VESSEL_TYPE = "VESSEL_TYPE"
    MMSI = "MMSI"
    POSITION = "POSITION"
    POB = "POB"
    INCIDENT = "INCIDENT"


class Span(BaseModel):
    start: int
    end: int
    label: EntityLabel
    text: str


class Turn(BaseModel):
    speaker: str  # "CALLER" | "MRCC"
    text: str
    lang: Language


class Dialogue(BaseModel):
    """One annotated training example produced by the dataset factory."""

    id: str
    split: str
    language: Language
    style: str
    unseen_phrasing: bool  # uses a phrasing held out from train (eval splits only)
    unseen_vessel: bool  # vessel name held out from train (eval splits only)
    incident: Incident
    turns: list[Turn]
    transcript: str  # turns joined as "SPEAKER: text" lines; spans index into this
    spans: list[Span]
    missing_fields: list[str]  # critical fields never stated by the caller
    contradiction: bool  # caller gives conflicting values for a field
