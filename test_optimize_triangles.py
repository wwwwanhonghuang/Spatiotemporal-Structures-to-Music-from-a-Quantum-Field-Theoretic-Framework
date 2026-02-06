
from filters.optimize_triangles import optimize_for_gft
if __name__ == "__main__":
    # 运行优化
    result = optimize_for_gft("./outputs/json/first_rabbit_music_spacetime.json")
    
    if result:
        print("\n" + "="*70)
        print("NEXT STEPS FOR GFT SIMULATION")
        print("="*70)
        print("\n1. Load the GFT-ready JSON file into your simulator:")
        print("   • Use 'field_config' as initial field values")
        print("   • Use 'triangles' as the simplicial complex")
        print("   • Use 'points' as vertex positions")
        
        print("\n2. Run GFT dynamics:")
        print("   • Apply MCMC sampling with the given parameters")
        print("   • Evolve the field configuration")
        print("   • Observe emergent spacetime structures")
        
        print("\n3. Map back to music:")
        print("   • Extract evolved field patterns")
        print("   • Map triangle properties to musical parameters")
        print("   • Generate new MIDI from the GFT-evolved geometry")