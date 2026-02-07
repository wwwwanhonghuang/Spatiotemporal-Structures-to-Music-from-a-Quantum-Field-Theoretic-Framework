#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# Semantic Field (Reconstruction-grade)
# =============================================================================

@dataclass
class SemanticField:
    version: str = "semantic-2.0"

    tempo_bpm: Optional[float] = None
    ticks_per_beat: Optional[int] = None

    edge_fields: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "version": self.version,
            "tempo_bpm": self.tempo_bpm,
            "ticks_per_beat": self.ticks_per_beat,
            "edge_fields": self.edge_fields,
        }

    def serialize(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"✓ Saved SemanticField: {path}")


# =============================================================================
# General SpinFoam IR
# =============================================================================

@dataclass
class GeneralFoamVertex:
    vid: int
    label: str = ""


@dataclass
class GeneralFoamEdge:
    eid: int
    v_start: int
    v_end: Optional[int]
    incident_faces: List[int] = field(default_factory=list)


@dataclass
class GeneralFoamFace:
    fid: int
    boundary_edges: List[int] = field(default_factory=list)
    j: Optional[float] = None
    m: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneralSpinFoamIR:
    version: str = "general-2.0"

    vertices: List[GeneralFoamVertex] = field(default_factory=list)
    edges: List[GeneralFoamEdge] = field(default_factory=list)
    faces: List[GeneralFoamFace] = field(default_factory=list)

    def to_dict(self):
        return {
            "version": self.version,
            "num_vertices": len(self.vertices),
            "num_edges": len(self.edges),
            "num_faces": len(self.faces),
            "vertices": [asdict(v) for v in self.vertices],
            "edges": [asdict(e) for e in self.edges],
            "faces": [asdict(f) for f in self.faces],
        }

    def serialize(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"✓ Saved General SpinFoam IR: {path}")


# =============================================================================
# Helpers
# =============================================================================

def _load_ir_payload(ir_or_json: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:

    if isinstance(ir_or_json, str):
        with open(ir_or_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if isinstance(ir_or_json, dict):
        return ir_or_json

    if hasattr(ir_or_json, "to_dict"):
        try:
            return ir_or_json.to_dict(include_foliation=True)
        except TypeError:
            return ir_or_json.to_dict()

    if all(hasattr(ir_or_json, k) for k in ("vertices", "edges", "faces")):
        return {
            "vertices": [vars(v) for v in ir_or_json.vertices],
            "edges": [vars(e) for e in ir_or_json.edges],
            "faces": [vars(f) for f in ir_or_json.faces],
        }

    raise TypeError("Unsupported IR input.")


def _extract_id(x: Dict[str, Any], key: str) -> int:
    if key in x:
        return int(x[key])
    for alt in ("id", "idx"):
        if alt in x:
            return int(x[alt])
    raise KeyError(f"Missing id field '{key}'")


# =============================================================================
# Compiler
# =============================================================================

def compile_to_general_spinfoam_ir_with_field(
    ir_or_json,
    *,
    keep_m=True,
    keep_meta=False,
    reindex_ids=True,
):

    payload = _load_ir_payload(ir_or_json)

    semantic = SemanticField(
        tempo_bpm=payload.get("tempo_bpm"),
        ticks_per_beat=payload.get("ticks_per_beat"),
    )

    src_vertices = payload.get("vertices", [])
    src_edges = payload.get("edges", [])
    src_faces = payload.get("faces", [])

    # ---------- ID maps ----------
    v_ids = [_extract_id(v, "vid") for v in src_vertices]
    e_ids = [_extract_id(e, "eid") for e in src_edges]
    f_ids = [_extract_id(f, "fid") for f in src_faces]

    if reindex_ids:
        v_map = {old: i for i, old in enumerate(sorted(v_ids))}
        e_map = {old: i for i, old in enumerate(sorted(e_ids))}
        f_map = {old: i for i, old in enumerate(sorted(f_ids))}
    else:
        v_map = {old: old for old in v_ids}
        e_map = {old: old for old in e_ids}
        f_map = {old: old for old in f_ids}

    # ---------- vertices ----------
    g_vertices = [
        GeneralFoamVertex(
            vid=v_map[_extract_id(v, "vid")],
            label=str(v.get("label", "")),
        )
        for v in src_vertices
    ]

    # ---------- faces ----------
    g_faces = []

    for face in src_faces:

        new_fid = f_map[_extract_id(face, "fid")]

        boundary = []

        for key, value in face.items():
            if value is None:
                continue
            if key == "owner_edge" or key.startswith("edge_"):
                boundary.append(e_map[int(value)])

        boundary = sorted(set(boundary))

        g_faces.append(
            GeneralFoamFace(
                fid=new_fid,
                boundary_edges=boundary,
                j=float(face["j"]) if face.get("j") is not None else None,
                m=float(face["m"]) if (keep_m and face.get("m") is not None) else None,
                meta=face.get("meta", {}) if keep_meta else {},
            )
        )

    # ---------- incident map ----------
    incident = {e_map[e]: [] for e in e_ids}

    for f in g_faces:
        for be in f.boundary_edges:
            incident.setdefault(be, []).append(f.fid)

    # ---------- edges ----------
    g_edges = []

    for edge in src_edges:

        old_eid = _extract_id(edge, "eid")
        new_eid = e_map[old_eid]

        v_start = v_map[int(edge["v_start"])]
        v_end = v_map[int(edge["v_end"])] if edge.get("v_end") is not None else None

        g_edges.append(
            GeneralFoamEdge(
                eid=new_eid,
                v_start=v_start,
                v_end=v_end,
                incident_faces=sorted(set(incident.get(new_eid, []))),
            )
        )

        # =====================================================
        # 🔥 Reconstruction-grade semantic extraction
        # =====================================================

        note = edge.get("note", {})

        embedding = edge.get("embedding", {})

        start = embedding.get("start_beats")
        end = embedding.get("end_beats")

        if start is None:
            start = note.get("start_beats")

        if end is None:
            end = note.get("end_beats")

        if start is not None and end is not None:
            duration = end - start
        else:
            duration = None

        semantic.edge_fields[new_eid] = {

            "pitch": note.get("pitch"),
            "velocity": note.get("velocity"),
            "channel": note.get("channel", 0),

            "start_beats": start,
            "end_beats": end,
            "duration_beats": duration,
        }

    general = GeneralSpinFoamIR(
        vertices=sorted(g_vertices, key=lambda x: x.vid),
        edges=sorted(g_edges, key=lambda x: x.eid),
        faces=sorted(g_faces, key=lambda x: x.fid),
    )

    return general, semantic


# =============================================================================
# File wrapper
# =============================================================================

def compile_file_to_general_with_field(
    in_json,
    out_general,
    *,
    out_semantic=None,
):

    g, s = compile_to_general_spinfoam_ir_with_field(in_json)

    g.serialize(out_general)

    if out_semantic is None:
        out_semantic = out_general.replace(".json", "_semantic.json")

    s.serialize(out_semantic)

    return out_general, out_semantic

