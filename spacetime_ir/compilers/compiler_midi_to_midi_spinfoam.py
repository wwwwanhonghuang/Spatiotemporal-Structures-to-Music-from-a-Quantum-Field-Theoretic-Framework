from typing import Optional, List, Tuple
from spacetime_ir.midi_ir.midi_ir import MIDISpinFoamIR
from spacetime_ir.spinfoam_ir.foam_edge import FoamEdge
from spacetime_ir.spinfoam_ir.foam_face import FoamFace
from spacetime_ir.spinfoam_ir.foam_vertex import FoamVertex
from spacetime_ir.spinfoam_ir.foliation_slice import FoliationSlice
from spacetime_ir.midi_ir.intrinsic_vertex import SliceSignature, signature_of_slice
from spacetime_ir.midi_ir.gluing import *
from spacetime_ir.midi_ir.note_key import NoteKey
from spacetime_ir.midi_ir.spin_physics import *
from spacetime_ir.midi_ir.intertwiner import *
from utils.music import parse_midi_file, detect_state_change_times


# =============================================================================
# Compiler
# =============================================================================

def compile_midi_to_spinfoam_ir(
    midi_path: str,
    output_json: Optional[str] = None,
    include_foliation: bool = True,
    glue_top_k: int = 2,
    intertwiner_mode: str = "hybrid",
    closure_tol: float = 0.75,
) -> MIDISpinFoamIR:
    """
    Full compiler:
      - edges: one per MIDI note instance
      - note-component faces: 4 per edge with (j,m,flux)
      - edge intertwiner: hybrid gate + fusion + vector closure + m-sum
      - optional foliation: Γ slices
      - intrinsic vertices: rewrite regions (compressed identical signatures)
      - glue-worldsheet faces between rewrite regions
    """
    notes, bpm, tpb = parse_midi_file(midi_path)
    times = detect_state_change_times(notes)

    ir = MIDISpinFoamIR(
        title=midi_path,
        tempo_bpm=bpm,
        ticks_per_beat=tpb
    )

    # Build edges (note worldlines): one edge per MIDINote instance
    for eid, n in enumerate(notes):
        ir.edges.append(FoamEdge(
            eid=eid,
            note={"pitch": n.pitch, "velocity": n.velocity, "channel": n.channel},
            v_start=-1,
            v_end=None,
            component_faces=[],
            intertwiner={},
            embedding={"start_beats": n.start_beats, "end_beats": n.end_beats}
        ))

    # Slice -> active edges
    def active_edges_in_slice(sid: int) -> List[int]:
        t = times[sid]
        out = []
        for eid, n in enumerate(notes):
            if n.is_active_at(t):
                out.append(eid)
        return sorted(out)

    # Build optional foliation and slice signatures
    slice_links: List[List[Tuple[int, int]]] = []
    slice_signs: List[SliceSignature] = []
    for sid, t in enumerate(times):
        active_eids = active_edges_in_slice(sid)
        links = build_glue_links(active_eids, ir.edges, top_k=glue_top_k)
        slice_links.append(links)
        slice_signs.append(signature_of_slice(active_eids, links))
        if include_foliation:
            ir.foliation.append(FoliationSlice(
                sid=sid,
                embedding_time_beats=t,
                gamma_nodes=active_eids,
                gamma_links=links
            ))

    # Intrinsic vertices as rewrite regions (compress identical consecutive slice signatures)
    region_starts = [0]
    for sid in range(1, len(slice_signs)):
        if slice_signs[sid] != slice_signs[sid - 1]:
            region_starts.append(sid)

    # Create vertices (one per region)
    for rid in range(len(region_starts)):
        ir.vertices.append(FoamVertex(vid=rid, label=f"rewrite_region#{rid}"))

    # Map each slice sid -> region id
    region_of_slice: Dict[int, int] = {}
    rid = 0
    next_idx = 1
    next_boundary = region_starts[next_idx] if next_idx < len(region_starts) else None
    for sid in range(len(times)):
        if next_boundary is not None and sid >= next_boundary:
            rid += 1
            next_idx += 1
            next_boundary = region_starts[next_idx] if next_idx < len(region_starts) else None
        region_of_slice[sid] = rid

    # Assign edge endpoints in region vertex ids (intrinsic)
    for e in ir.edges:
        t0 = e.embedding["start_beats"]
        t1 = e.embedding["end_beats"]
        s_start = min(range(len(times)), key=lambda i: abs(times[i] - t0))
        s_end = min(range(len(times)), key=lambda i: abs(times[i] - t1))

        e.v_start = region_of_slice[s_start]
        e.v_end = region_of_slice[s_end] if s_end < len(times) - 1 else None

    # Build note-component faces (4 per edge) + intertwiner
    fid = 0
    for e in ir.edges:
        nk = NoteKey(e.note["pitch"], e.note["velocity"], e.note["channel"])
        onset = e.embedding["start_beats"]

        face_data = make_note_component_faces(nk, onset)
        # write component faces to IR
        for ch in CHANNELS:
            j = face_data[ch]["j"]
            m = face_data[ch]["m"]
            flux = face_data[ch]["flux"]
            meta = face_data[ch]["meta"]
            ir.faces.append(FoamFace(
                fid=fid,
                face_type="note_component",
                owner_edge=e.eid,
                channel=ch,
                j=j,
                m=m,
                flux=flux,
                meta=meta
            ))
            e.component_faces.append(fid)
            fid += 1

        # build intertwiner (hybrid by default)
        # allow user to set vector closure tolerance
        inter = build_intertwiner(face_data, mode=intertwiner_mode)
        # overwrite closure tol if using vector closure in out
        if "vector_closure" in inter:
            inter["vector_closure"]["tol"] = closure_tol
            inter["vector_closure"]["closed"] = inter["vector_closure"]["closure_error"] <= closure_tol
            # recompute closed for hybrid
            if intertwiner_mode == "hybrid":
                inter["closed"] = (
                    inter["gate"]["admissible"]
                    and inter["fusion"]["exists"]
                    and inter["vector_closure"]["closed"]
                    and inter["m_ok"]
                )
            elif intertwiner_mode == "vector_closure":
                inter["closed"] = inter["vector_closure"]["closed"] and inter["m_ok"]
        e.intertwiner = inter

    # Build region representative glue links (first slice in region)
    region_rep_slice: Dict[int, int] = {}
    for sid in range(len(times)):
        r = region_of_slice[sid]
        if r not in region_rep_slice:
            region_rep_slice[r] = sid

    region_links: Dict[int, Set[Tuple[int, int]]] = {}
    for r, sid in region_rep_slice.items():
        region_links[r] = set(slice_links[sid])

    # Glue-worldsheet faces between consecutive regions: persistent links sweep faces
    for r in range(len(ir.vertices) - 1):
        common = region_links.get(r, set()).intersection(region_links.get(r + 1, set()))
        for (a, b) in sorted(common):
            lbl = glue_face_label(ir.edges, a, b)
            ir.faces.append(FoamFace(
                fid=fid,
                face_type="glue_worldsheet",
                edge_a=a,
                edge_b=b,
                v_start=r,
                v_end=r + 1,
                j=lbl["j"],
                m=lbl["m"],
                flux=lbl["flux"],
                meta=lbl["meta"]
            ))
            fid += 1

    if output_json:
        ir.serialize(output_json, include_foliation=include_foliation)

    return ir