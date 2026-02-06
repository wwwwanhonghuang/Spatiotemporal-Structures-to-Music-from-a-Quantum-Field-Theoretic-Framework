
from music_spacetime.midi_to_gft import process_midi_with_robust_triangulation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mid_input", default="assets/first_rabbit.mid")
args = parser.parse_args()

if __name__ == "__main__":
    # 处理MIDI文件
    result = process_midi_with_robust_triangulation(args.mid_input)
    
    if result:
        print(f"\nFinal result for GFT research:")
        print(f"  • Points: {result['metadata']['num_points']}")
        print(f"  • Triangles: {result['metadata']['num_triangles']}")
        print(f"  • Ready for GFT simulation!")
        
        if result['metadata']['num_triangles'] == 0:
            print("\nWARNING: No triangles generated!")
            print("Consider these alternatives:")
            print("1. Reduce grid resolution to get more dense points")
            print("2. Use different scaling factors for time/pitch/velocity")
            print("3. Try a different triangulation algorithm")
            print("4. Manually create a base triangulation for your GFT")