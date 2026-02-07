from dataclasses import dataclass


@dataclass
class FoamVertex:
    vid: int
    label: str = ""  # intrinsic label only (no time)
