from spacetime_ir.compilers.compiler_midi_spinfoam_to_general_spinfoam import compile_file_to_general_with_field
# compile_file_to_general("test_example.spinfoam.v4.json", "general_spinfoam.json", keep_m=True, keep_meta=False)

import argparse
# =============================================================================
# CLI
# =============================================================================

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_json")
    parser.add_argument("output_general")

    args = parser.parse_args()

    compile_file_to_general_with_field(
        args.input_json,
        args.output_general,
    )
