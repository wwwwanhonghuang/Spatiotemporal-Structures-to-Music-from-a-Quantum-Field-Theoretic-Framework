import numpy as np
from scipy.spatial import KDTree, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from filters.gft_connectivity_optimizer import GFTConnectivityOptimizer

def visualize_connectivity_comparison(original_data: Dict, connected_data: Dict, 
                                     save_path: str = "outputs/images/connectivity_comparison.png"):
    """可视化连通性对比"""
    
    fig = plt.figure(figsize=(20, 10))
    
    # 原始几何
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("Original Geometry", fontsize=12, fontweight='bold')
    
    original_points = original_data['original_points']
    original_triangles = original_data['original_triangles']
    
    # 绘制原始三角形
    for tri in original_triangles[:500]:  # 限制数量
        if len(tri) == 3:
            tri_points = original_points[list(tri) + [tri[0]]]
            ax1.plot(tri_points[:, 0], tri_points[:, 1], tri_points[:, 2],
                    'b-', alpha=0.1, linewidth=0.5)
    
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
               c='red', s=10, alpha=0.5)
    
    # 分析孤立点
    original_analysis = original_data['analysis']
    isolated = original_analysis['isolated_nodes']
    if isolated:
        isolated_points = original_points[isolated]
        ax1.scatter(isolated_points[:, 0], isolated_points[:, 1], isolated_points[:, 2],
                   c='black', s=50, marker='x', label=f'Isolated ({len(isolated)})')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Pitch')
    ax1.set_zlabel('Velocity')
    ax1.legend()
    
    # 连接后几何
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Fully Connected Geometry", fontsize=12, fontweight='bold')
    
    connected_points = connected_data['points']
    connected_triangles = connected_data['triangles']
    
    # 绘制所有三角形
    for tri in connected_triangles[:]:
        if len(tri) == 3:
            tri_points = connected_points[list(tri) + [tri[0]]]
            ax2.plot(tri_points[:, 0], tri_points[:, 1], tri_points[:, 2],
                    'b-', alpha=0.1, linewidth=0.5)
    
    ax2.scatter(connected_points[:, 0], connected_points[:, 1], connected_points[:, 2],
               c='green', s=10, alpha=0.5)
    
    # 突出显示新增的边
    added_edges = connected_data.get('added_edges', [])
    for edge in added_edges[:]: 
        if len(edge) == 2:
            p1, p2 = connected_points[edge[0]], connected_points[edge[1]]
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'r-', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Pitch')
    ax2.set_zlabel('Velocity')
    
    # 连通性统计
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    # 计算统计
    original_stats = original_analysis
    connected_analysis = GFTConnectivityOptimizer()._analyze_connectivity(connected_points, connected_triangles)
    connected_stats = connected_analysis
    
    info_text = f"""
    Connectivity Comparison
    {'='*40}
    
    Original Geometry:
    • Points: {len(original_points)}
    • Triangles: {len(original_triangles)}
    • Components: {original_stats['num_components']}
    • Isolated Points: {original_stats['num_isolated']}
    • Avg Degree: {original_stats['avg_degree']:.2f}
    
    Connected Geometry:
    • Points: {len(connected_points)}
    • Triangles: {len(connected_triangles)}
    • Components: {connected_stats['num_components']}
    • Isolated Points: {connected_stats['num_isolated']}
    • Avg Degree: {connected_stats['avg_degree']:.2f}
    
    Improvements:
    • Added Edges: {len(added_edges)}
    • Added Triangles: {len(connected_data.get('added_triangles', []))}
    • Strategy: {connected_data.get('strategy', 'unknown')}
    
    GFT Readiness:
    • Fully Connected: {connected_stats['num_components'] == 1}
    • No Isolated Points: {connected_stats['num_isolated'] == 0}
    • Triangle Density: {len(connected_triangles)*3/len(connected_points):.2f} triangles/point
    """
    
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle("GFT Connectivity Optimization", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Connectivity comparison saved to: {save_path}")


def optimize_for_gft(json_file: str = "./outputs/json/first_rabbit_music_spacetime.json"):
    """为GFT优化三角化"""
    
    print("="*70)
    print("GFT CONNECTIVITY OPTIMIZATION")
    print("="*70)
    
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return
    
    # 1. 初始化优化器
    optimizer = GFTConnectivityOptimizer(connectivity_threshold=0.15)
    
    # 2. 加载和分析原始数据
    print("\n1. Loading and analyzing original data...")
    original_data = optimizer.load_and_analyze(json_file)
    
    original_stats = original_data['analysis']
    print(f"   • Original points: {len(original_data['original_points'])}")
    print(f"   • Original triangles: {len(original_data['original_triangles'])}")
    print(f"   • Connected components: {original_stats['num_components']}")
    print(f"   • Isolated points: {original_stats['num_isolated']}")
    
    # 3. 创建全联通几何
    print("\n2. Creating fully connected geometry...")
    
    # 尝试不同的策略
    strategies = ['gft_optimized', 'minimal_spanning', 'delaunay_complete']
    
    best_result = None
    best_score = -1
    
    for strategy in strategies:
        print(f"\n   Trying strategy: {strategy}")
        try:
            result = optimizer.create_fully_connected_geometry(
                original_data['original_points'],
                original_data['original_triangles'],
                strategy=strategy
            )
            
            # 评分：基于连通性和三角形密度
            analysis = optimizer._analyze_connectivity(result['points'], result['triangles'])
            
            score = 0
            if analysis['num_components'] == 1:
                score += 50
            if analysis['num_isolated'] == 0:
                score += 30
            score += min(20, analysis['avg_degree'])  # 平均度
            
            print(f"     • Score: {score}, Components: {analysis['num_components']}, "
                  f"Isolated: {analysis['num_isolated']}")
            
            if score > best_score:
                best_score = score
                best_result = result
                best_result['strategy'] = strategy
        
        except Exception as e:
            print(f"     • Strategy failed: {e}")
    
    if not best_result:
        print("\nERROR: All strategies failed!")
        return
    
    print(f"\n3. Best strategy: {best_result['strategy']} (score: {best_score})")
    
    # 4. 可视化对比
    print("\n4. Creating visualization...")
    midi_name = os.path.splitext(json_file)[0]
    visualize_connectivity_comparison(
        original_data, best_result,
        f"{midi_name}_connectivity_comparison.png"
    )
    
    # 5. 保存GFT就绪数据
    print("\n5. Saving GFT-ready data...")
    output_file = f"{midi_name}_gft_fully_connected.json"
    optimizer.save_gft_ready_data(best_result, output_file)
    
    # 6. 验证结果
    final_analysis = optimizer._analyze_connectivity(best_result['points'], best_result['triangles'])
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nOriginal Geometry:")
    print(f"  • Components: {original_stats['num_components']}")
    print(f"  • Isolated Points: {original_stats['num_isolated']}")
    print(f"  • Avg Degree: {original_stats['avg_degree']:.2f}")
    
    print(f"\nOptimized Geometry:")
    print(f"  • Components: {final_analysis['num_components']} "
          f"{'✓' if final_analysis['num_components'] == 1 else '✗'}")
    print(f"  • Isolated Points: {final_analysis['num_isolated']} "
          f"{'✓' if final_analysis['num_isolated'] == 0 else '✗'}")
    print(f"  • Avg Degree: {final_analysis['avg_degree']:.2f}")
    print(f"  • Total Triangles: {len(best_result['triangles'])}")
    print(f"  • Strategy Used: {best_result['strategy']}")
    
    print(f"\nFiles Created:")
    print(f"  1. {midi_name}_connectivity_comparison.png")
    print(f"  2. {output_file} (GFT-ready data)")
    
    if final_analysis['num_components'] == 1 and final_analysis['num_isolated'] == 0:
        print("\n✓ SUCCESS: Geometry is fully connected and ready for GFT!")
    else:
        print("\n⚠ WARNING: Geometry is not fully connected.")
        print("Consider manual adjustments or different parameters.")
    
    return best_result

