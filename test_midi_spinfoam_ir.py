
from spacetime_ir.midi_ir.midi_ir import compile_midi_to_spinfoam, create_test_midi, print_spinfoam_summary


if __name__ == "__main__":
    # Create and compile test MIDI
    test_midi = "assets/first_rabbit.mid"#create_test_midi()
    
    try:
        # Compile to spin foam
        foam = compile_midi_to_spinfoam(
            midi_file_path=test_midi,
            output_json_path="philosophy_spinfoam.json"
        )
        
        # Print summary
        print_spinfoam_summary(foam)
        
        # Demonstrate the causal structure
        print(f"\n{'='*80}")
        print(f"TIME EMERGES FROM THIS CAUSAL CHAIN:")
        print(f"{'='*80}")
        
        chain = foam.get_causal_chain()
        for i, event in enumerate(chain):
            print(f"Step {i}: At E{event.event_id}, after {event.time:.2f} beats of musical time")
            if event.duration_to_next:
                print(f"       Music persists unchanged for {event.duration_to_next:.2f} beats")
                print(f"       Then at E{event.next_event_id}, a change occurs")
            else:
                print(f"       Music ends here")
            print()
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install mido: pip install mido")
    except Exception as e:
        print(f"Error: {e}")