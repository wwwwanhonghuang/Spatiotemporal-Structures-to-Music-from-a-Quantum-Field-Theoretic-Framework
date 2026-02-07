from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FoamEdge:
    eid: int
    note: Dict[str, int]  # pitch, velocity, channel

    v_start: int
    v_end: Optional[int]  # open if None

    component_faces: List[int] = field(default_factory=list)

    intertwiner: Dict = field(default_factory=dict)

    # optional embedding (not intrinsic)
    embedding: Dict = field(default_factory=dict)