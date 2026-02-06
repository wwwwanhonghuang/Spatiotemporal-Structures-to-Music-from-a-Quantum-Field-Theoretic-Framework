import os
import argparse

from music_spacetime.visualization.visualize_music_spacetime import visualize_complete_triangulation, create_enhanced_network_visualization

parser = argparse.ArgumentParser()
parser.add_argument("--json_input", default="outputs/json/first_rabbit_music_spacetime.json")
args = parser.parse_args()

if __name__ == "__main__":
    json_file = args.json_input
    
    if os.path.exists(json_file):
        print("="*70)
        print("COMPLETE TRIANGULATION VISUALIZATION")
        print("="*70)
        
        # 1. 完整可视化
        analysis = visualize_complete_triangulation(json_file)
        
        print(f"\nKey findings:")
        print(f"  1. {analysis['isolated_points']} isolated points found")
        print(f"  2. Regional distribution:")
        for region, count in analysis['region_triangle_counts'].items():
            print(f"     • {region}: {count} triangles")
        
        # 2. 增强网络可视化
        print("\n" + "="*70)
        print("ENHANCED NETWORK VISUALIZATION")
        print("="*70)
        
        network_analysis = create_enhanced_network_visualization(json_file)
        
        print(f"\nNetwork analysis:")
        print(f"  • Connected components: {len(network_analysis['components'])}")
        print(f"  • Largest component: {max(network_analysis['component_sizes']) if network_analysis['component_sizes'] else 0} points")
        
    else:
        print(f"File not found: {json_file}")
        print("\nAvailable JSON files:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"  • {file}")