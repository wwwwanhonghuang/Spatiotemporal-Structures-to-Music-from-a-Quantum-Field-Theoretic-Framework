#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpinFoam Geometry Statistics

Produces observables instead of layouts.

Usage example:

from analysis.spinfoam_stats import *
ir = load_json("test_example.spinfoam.v4.json")

vertex_face_heatmap(ir)
edge_coupling_heatmap(ir)
vertex_degree_histogram(ir)
"""

import json
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Loader
# -----------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Vertex Face Density
# -----------------------------

def vertex_face_counts(ir):
    counts = {v["vid"]: 0 for v in ir["vertices"]}

    for f in ir["faces"]:
        vs = f.get("v_start")
        ve = f.get("v_end")

        if vs is not None:
            counts[vs] += 1
        if ve is not None:
            counts[ve] += 1

    return counts


def vertex_face_heatmap(ir):
    counts = vertex_face_counts(ir)

    vids = list(counts.keys())
    vals = list(counts.values())

    plt.figure()
    plt.bar(vids, vals)
    plt.title("Face Density per Vertex")
    plt.xlabel("Vertex")
    plt.ylabel("# Faces")
    plt.show()


# -----------------------------
# Vertex Degree
# -----------------------------

def vertex_degree_histogram(ir):
    degrees = []

    for v in ir["vertices"]:
        vid = v["vid"]
        deg = 0

        for e in ir["edges"]:
            if e.get("v_start") == vid:
                deg += 1
            if e.get("v_end") == vid:
                deg += 1

        degrees.append(deg)

    plt.figure()
    plt.hist(degrees, bins=10)
    plt.title("Vertex Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()


# -----------------------------
# Edge Coupling (VERY important)
# -----------------------------

def build_edge_coupling_matrix(ir):
    edges = ir["edges"]
    faces = ir["faces"]

    n = len(edges)
    M = np.zeros((n, n))

    for f in faces:
        if f.get("face_type") != "glue_worldsheet":
            continue

        a = f.get("edge_a")
        b = f.get("edge_b")

        if a is None or b is None:
            continue

        j = f.get("j", 1)

        M[a, b] += j
        M[b, a] += j

    return M


def edge_coupling_heatmap(ir):
    M = build_edge_coupling_matrix(ir)

    plt.figure()
    plt.imshow(M)
    plt.colorbar(label="Glue strength (sum j)")
    plt.title("Edge Coupling Heatmap")
    plt.xlabel("Edge")
    plt.ylabel("Edge")
    plt.show()


# -----------------------------
# Face Spin Distribution
# -----------------------------

def face_spin_histogram(ir):
    js = []

    for f in ir["faces"]:
        j = f.get("j")
        if j is not None:
            js.append(j)

    plt.figure()
    plt.hist(js, bins=20)
    plt.title("Face Spin Distribution")
    plt.xlabel("j")
    plt.ylabel("Count")
    plt.show()
