
from spacetime_ir.compilers.compiler_midi_to_midi_spinfoam import compile_midi_to_spinfoam_ir
from spacetime_ir.midi_ir.midi_ir import print_summary, faces_of_edge, glue_faces_between
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--midi_file_path", default="assets/first_rabbit.mid")
parser.add_argument("--ir_output", default="test_example.spinfoam.v4.json")

parser.add_argument("-v", action="store_true")

args = parser.parse_args()

midi_file_path = args.midi_file_path

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    midi_path = midi_file_path #create_test_midi()

    ir = compile_midi_to_spinfoam_ir(
        midi_path,
        output_json=args.ir_output,
        include_foliation=True,     # set False to make ontology-only output
        glue_top_k=2,               # sparse Γ links
        intertwiner_mode="hybrid",  # "vector_closure" | "fusion_4valent" | "constraint_gate" | "hybrid"
        closure_tol=0.75
    )

    print_summary(ir)

    # Example search:
    # - find (j,m) of note-edge 0:
    e0_faces = faces_of_edge(ir, 0)
    # - find glue-worldsheet faces between note-edge 0 and 1:
    g01 = glue_faces_between(ir, 0, 1)
    print(f"\nQuery demo: edge0 component faces={len(e0_faces)} ; glue faces between e0-e1={len(g01)}")
