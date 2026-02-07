from spacetime_ir.midi_ir.midi_ir_decoder import decode_spinfoam_to_midi
# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ir", required=True, help="Path to SpinFoam IR json")
    p.add_argument("--out", required=True, help="Output MIDI path")
    p.add_argument("--allow_missing_embedding", action="store_true")
    args = p.parse_args()

    decode_spinfoam_to_midi(
        args.ir,
        args.out,
        allow_missing_embedding=args.allow_missing_embedding,
    )
