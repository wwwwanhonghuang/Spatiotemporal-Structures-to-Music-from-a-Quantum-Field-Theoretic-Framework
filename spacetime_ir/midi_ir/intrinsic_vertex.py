from dataclasses import dataclass
from typing import List,  Tuple

@dataclass(frozen=True)
class SliceSignature:
    active_edges: Tuple[int, ...]
    glue_links: Tuple[Tuple[int, int], ...]

def signature_of_slice(active_eids: List[int], links: List[Tuple[int, int]]) -> SliceSignature:
    return SliceSignature(
        active_edges=tuple(sorted(active_eids)),
        glue_links=tuple(sorted((min(a,b), max(a,b)) for a,b in links))
    )