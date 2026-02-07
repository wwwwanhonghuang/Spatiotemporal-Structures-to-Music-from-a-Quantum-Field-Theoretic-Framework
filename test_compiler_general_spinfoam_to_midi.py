import argparse
from spacetime_ir.compilers.compiler_general_spinfoam_to_midi import decode_general_ir_to_midi, load_general_ir, load_semantic_field
# =============================================================================
# CLI
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description="Decode General SpinFoam IR to MIDI (optional semantic field).")
    ap.add_argument("general_ir", help="General SpinFoam IR JSON")
    ap.add_argument("out_midi", help="Output MIDI file")
    ap.add_argument("--semantic", default=None, help="Semantic field JSON (optional)")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--tpb", type=int, default=480)
    ap.add_argument("--beat-unit", type=float, default=1.0)
    ap.add_argument("--base-pitch", type=int, default=60)
    ap.add_argument("--pitch-span", type=int, default=6)
    ap.add_argument("--channel", type=int, default=0)
    args = ap.parse_args()

    ir = load_general_ir(args.general_ir)
    sf = load_semantic_field(args.semantic) if args.semantic else None

    decode_general_ir_to_midi(
        ir,
        args.out_midi,
        semantic_field=sf,
        default_tpb=args.tpb,
        default_bpm=args.bpm,
        #beat_unit=args.beat_unit,
        base_pitch=args.base_pitch,
        pitch_span=args.pitch_span,
        default_channel=args.channel,
    )


if __name__ == "__main__":
    main()
