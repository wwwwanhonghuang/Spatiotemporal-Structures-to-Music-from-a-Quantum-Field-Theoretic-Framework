import argparse
from spacetime_ir.midi_ir.visualizers.midi_spinfoam_visualizer import load_json, build_foam_incidence_graph, draw_graph, build_gamma_graph_from_ir, safe_get
from spacetime_ir.midi_ir.visualizers.midi_spinfoam_visualizer import build_gamma_graph_from_external_gamma, draw_gamma_edges_with_edge_labels
# ----------------------------
# CLI
# ----------------------------


if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_foam = sub.add_parser("foam", help="Visualize spin foam 2-complex (incidence graph)")
    p_foam.add_argument("--ir", required=True, help="SpinFoam IR JSON path")
    p_foam.add_argument("--out", required=True, help="Output image path (png/pdf/svg)")
    p_foam.add_argument("--layout", default="spring", choices=["spring", "kamada_kawai", "sfdp"], help="Layout engine")
    p_foam.add_argument("--include_note_component_faces", action="store_true",
                        help="Include note_component faces (usually very dense)")
    p_foam.add_argument("--k", type=float, default=None, help="spring_layout k parameter (spacing)")

    p_gamma = sub.add_parser("gamma", help="Visualize spin network Γ (from foliation slice or external file)")
    src = p_gamma.add_mutually_exclusive_group(required=True)
    src.add_argument("--ir", help="SpinFoam IR JSON path")
    src.add_argument("--gamma", help="External Γ JSON path")
    p_gamma.add_argument("--sid", type=int, default=0, help="Slice id for foliation Γ (if --ir)")
    p_gamma.add_argument("--out", required=True, help="Output image path")
    p_gamma.add_argument("--edge_labels", action="store_true", help="Show interval labels on Γ edges")

    args = p.parse_args()

    if args.cmd == "foam":
        ir = load_json(args.ir)
        G = build_foam_incidence_graph(ir, include_note_component_faces=args.include_note_component_faces)
        title = f"SpinFoam incidence graph | V={ir.get('num_vertices')} E={ir.get('num_edges')} F={ir.get('num_faces')}"
        print("Begin Grawing Graph...")
        draw_graph(
            G, args.out,
            title=title,
            layout=args.layout,
            with_labels=True,
            font_size=7,
            node_size=620,
            k=args.k
        )

    elif args.cmd == "gamma":
        if args.ir:
            ir = load_json(args.ir)
            G = build_gamma_graph_from_ir(ir, args.sid)
            t = safe_get(ir, ["foliation", args.sid, "embedding_time_beats"], None)
            title = f"Γ slice sid={args.sid}" + (f" (beats={t:.2f})" if isinstance(t, (int, float)) else "")
        else:
            gamma = load_json(args.gamma)
            G = build_gamma_graph_from_external_gamma(gamma)
            title = "Γ (external)"

        if args.edge_labels:
            draw_gamma_edges_with_edge_labels(G, args.out, title=title)
        else:
            # plain draw
            # attach kind so it uses a single node style
            for n in G.nodes:
                G.nodes[n]["label"] = G.nodes[n].get("label", str(n))
            draw_graph(G, args.out, title=title, with_labels=True, font_size=10, node_size=850)

