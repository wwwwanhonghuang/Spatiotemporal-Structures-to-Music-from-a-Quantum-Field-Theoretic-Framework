from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Set, Optional

@dataclass
class FoamFace:
    fid: int
    face_type: str  # "note_component" | "glue_worldsheet"

    # note_component fields
    owner_edge: Optional[int] = None
    channel: Optional[str] = None  # harmonic|spectral|energy|phase
    j: Optional[float] = None
    m: Optional[float] = None
    flux: Optional[Tuple[float, float, float]] = None
    meta: Dict = field(default_factory=dict)

    # glue_worldsheet fields
    edge_a: Optional[int] = None
    edge_b: Optional[int] = None
    v_start: Optional[int] = None
    v_end: Optional[int] = None