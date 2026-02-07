#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpinFoam / SpinNetwork Visualizer for MIDI SpinFoam IR JSON (v4 style)

Two modes:
  (A) foam: visualize 2-complex via incidence graph (V/E/F as nodes)
  (B) gamma: visualize a spin network Γ (from foliation slice sid), or from an external Γ JSON

Usage:
  python visualizer.py foam  --ir test_example.spinfoam.v4.json --out foam.png
  python visualizer.py gamma --ir test_example.spinfoam.v4.json --sid 0 --out gamma0.png
  python visualizer.py gamma --gamma gamma.json --out gamma.png

Notes:
- This is a robust visualization: it does not assume faces are planar polygons.
- For foam, we draw an incidence graph: V-nodes connect to E-nodes; E-nodes connect to F-nodes (faces incident to that edge).
"""

import argparse
import json
import math
from typing import Dict, Any, List, Tuple, Optional

import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# Helpers: parse & labels
# ----------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def pitch_to_name(pitch: int) -> str:
    octave = (pitch // 12) - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Build Γ (spin network) graph
# ----------------------------

def build_gamma_graph_from_ir(ir: Dict[str, Any], sid: int) -> nx.Graph:
    fol = ir.get("foliation", [])
    if not fol:
        raise ValueError("IR has no foliation[]; cannot build Γ from slice. Re-export with include_foliation=True.")

    if sid < 0 or sid >= len(fol):
        raise ValueError(f"sid out of range: {sid}, available [0..{len(fol)-1}]")

    slice_obj = fol[sid]
    nodes = slice_obj.get("gamma_nodes", [])
    links = slice_obj.get("gamma_links", [])

    edges = ir.get("edges", [])

    G = nx.Graph()
    for eid in nodes:
        note = safe_get(edges[eid], ["note"], {}) if eid < len(edges) else {}
        pitch = note.get("pitch", None)
        vel = note.get("velocity", None)
        ch = note.get("channel", None)
        label = pitch_to_name(pitch) if isinstance(pitch, int) else f"e{eid}"
        G.add_node(eid, label=label, pitch=pitch, velocity=vel, channel=ch)

    for a, b in links:
        if a in G.nodes and b in G.nodes:
            # optional edge metadata: interval
            pa = G.nodes[a].get("pitch")
            pb = G.nodes[b].get("pitch")
            interval = None
            if isinstance(pa, int) and isinstance(pb, int):
                interval = abs(pa - pb) % 12
            G.add_edge(a, b, interval=interval)

    return G

def build_gamma_graph_from_external_gamma(gamma: Dict[str, Any]) -> nx.Graph:
    """
    Expected schema:
      {
        "nodes": [{"id": 0, "label": "...", ...}, ...] OR [0,1,2,...]
        "links": [[0,1],[1,2],...]
      }
    """
    G = nx.Graph()
    nodes = gamma.get("nodes", [])
    links = gamma.get("links", [])

    if nodes and isinstance(nodes[0], dict):
        for n in nodes:
            nid = n["id"]
            G.add_node(nid, **{k:v for k,v in n.items() if k != "id"})
    else:
        for nid in nodes:
            G.add_node(nid, label=str(nid))

    for a, b in links:
        G.add_edge(a, b)

    return G


# ----------------------------
# Build SpinFoam incidence graph
# ----------------------------

def build_foam_incidence_graph(ir: Dict[str, Any], include_note_component_faces: bool = False) -> nx.Graph:
    """
    Construct incidence graph:
      - V nodes: v{vid}
      - E nodes: e{eid}
      - F nodes: f{fid}

    Add edges:
      v -- e if edge endpoint matches vertex
      e -- f if face incident to edge:
         - note_component: owner_edge = eid  (optional; can be many, often clutters)
         - glue_worldsheet: edge_a/edge_b incident to eid
    """
    G = nx.Graph()

    vertices = ir.get("vertices", [])
    edges = ir.get("edges", [])
    faces = ir.get("faces", [])

    # Add V nodes
    for v in vertices:
        vid = v.get("vid")
        name = f"v{vid}"
        G.add_node(name, kind="V", label=name)

    # Add E nodes
    for e in edges:
        eid = e.get("eid")
        note = e.get("note", {})
        pitch = note.get("pitch")
        label = f"e{eid}:{pitch_to_name(pitch)}" if isinstance(pitch, int) else f"e{eid}"
        G.add_node(f"e{eid}", kind="E", label=label)

        vs = e.get("v_start", None)
        ve = e.get("v_end", None)
        if vs is not None:
            G.add_edge(f"v{vs}", f"e{eid}")
        if ve is not None:
            G.add_edge(f"v{ve}", f"e{eid}")

    # Add F nodes + incidence to E nodes
    for f in faces:
        fid = f.get("fid")
        ftype = f.get("face_type")
        fname = f"f{fid}"
        if ftype == "note_component" and (not include_note_component_faces):
            continue

        # label: keep short
        if ftype == "note_component":
            ch = f.get("channel", "")
            j = f.get("j", None)
            m = f.get("m", None)
            flabel = f"f{fid}:{ch} (j={j},m={m})"
        else:
            j = f.get("j", None)
            m = f.get("m", None)
            ea = f.get("edge_a")
            eb = f.get("edge_b")
            flabel = f"f{fid}:glue e{ea}-e{eb} (j={j},m={m})"

        G.add_node(fname, kind="F", label=flabel, face_type=ftype)

        if ftype == "note_component":
            owner = f.get("owner_edge")
            if owner is not None:
                G.add_edge(f"e{owner}", fname)

        elif ftype == "glue_worldsheet":
            ea = f.get("edge_a")
            eb = f.get("edge_b")
            if ea is not None:
                G.add_edge(f"e{ea}", fname)
            if eb is not None:
                G.add_edge(f"e{eb}", fname)

            # also connect to the region vertices if provided
            vs = f.get("v_start")
            ve = f.get("v_end")
            if vs is not None:
                G.add_edge(f"v{vs}", fname)
            if ve is not None:
                G.add_edge(f"v{ve}", fname)

    return G


# ----------------------------
# Drawing utilities
# ----------------------------

def draw_graph(
    G: nx.Graph,
    out_path: str,
    title: str = "",
    layout: str = "spring",
    seed: int = 7,
    node_size: int = 700,
    font_size: int = 8,
    with_labels: bool = True,
    label_key: str = "label",
    k: Optional[float] = None,
):
    plt.figure(figsize=(14, 9))
    print("nx setting layout...")

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=k)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "sfdp":
        pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
    else:
        pos = nx.spring_layout(G, seed=seed, k=k)
    print("nx setting layout finished")

    print("nx getting node attributes...")

    # Group nodes by kind if present
    kinds = nx.get_node_attributes(G, "kind")
    print("nx getting node attributes finished.")

    # We avoid hard-coding colors; matplotlib will choose defaults if not specified.
    # But to distinguish types clearly, we *do* set minimal styling via shapes only.
    V_nodes = [n for n in G.nodes if kinds.get(n) == "V"]
    E_nodes = [n for n in G.nodes if kinds.get(n) == "E"]
    F_nodes = [n for n in G.nodes if kinds.get(n) == "F"]
    other = [n for n in G.nodes if n not in set(V_nodes+E_nodes+F_nodes)]

    print("nx drawing network edges...")
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.2)
    print("nx drawing network edges finished")

    if V_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=V_nodes, node_shape="o", node_size=node_size, alpha=0.95)
    if E_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=E_nodes, node_shape="s", node_size=node_size, alpha=0.95)
    if F_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=F_nodes, node_shape="^", node_size=node_size, alpha=0.95)
    if other:
        nx.draw_networkx_nodes(G, pos, nodelist=other, node_shape="d", node_size=node_size, alpha=0.95)

    if with_labels:
        labels = {n: G.nodes[n].get(label_key, str(n)) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"✓ Saved: {out_path}")


def draw_gamma_edges_with_edge_labels(G: nx.Graph, out_path: str, title: str = "", seed: int = 7):
    """
    Specialized: draw Γ with interval labels on links (if present).
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.5)
    nx.draw_networkx_nodes(G, pos, node_size=850, alpha=0.95)

    labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # edge labels
    edge_labels = {}
    for a, b, data in G.edges(data=True):
        itv = data.get("interval")
        if itv is not None:
            edge_labels[(a, b)] = str(itv)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"✓ Saved: {out_path}")

