
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class FoliationSlice:
    sid: int
    embedding_time_beats: float
    gamma_nodes: List[int]              # active note-edges
    gamma_links: List[Tuple[int, int]]  # glue links (spin-network edges)