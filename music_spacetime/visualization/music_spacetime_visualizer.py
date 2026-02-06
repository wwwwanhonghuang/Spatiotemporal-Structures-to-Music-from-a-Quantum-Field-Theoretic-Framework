
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido
from typing import Dict, List, Tuple, Any, Set
import networkx as nx
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido
from typing import Dict, List, Tuple, Any, Set
import networkx as nx
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MusicSpacetimeVisualizer:
    """音乐时空可视化器"""
    
    @staticmethod
    def visualize_music_triangulation(points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                    features: List[Dict], analysis: Dict[str, Any],
                                    midi_name: str, save_path: str = None):
        """可视化音乐三角化"""
        if save_path is None:
            save_path = f"outputs/images/{midi_name}_music_spacetime.png"
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 3D时空结构
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.set_title("3D Music Spacetime", fontsize=11, fontweight='bold')
        
        # 按音高着色
        colors = []
        for feat in features:
            hue = (feat['pitch'] % 12) / 12.0
            colors.append(plt.cm.hsv(hue))
        
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=15, alpha=0.8, depthshade=True)
        
        # 绘制三角形
        if triangles:
            for tri in triangles[:]:
                if len(tri) == 3:
                    tri_points = points[list(tri) + [tri[0]]]
                    ax1.plot(tri_points[:, 0], tri_points[:, 1], tri_points[:, 2],
                            'b-', alpha=0.2, linewidth=0.8)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pitch')
        ax1.set_zlabel('Intensity')
        ax1.view_init(elev=20, azim=45)
        
        # 2. 2D投影
        ax2 = fig.add_subplot(232)
        ax2.set_title("Time-Pitch Projection", fontsize=11, fontweight='bold')
        
        times = [f['time'] for f in features]
        pitches = [f['pitch'] for f in features]
        velocities = [f['velocity_norm'] for f in features]
        
        scatter = ax2.scatter(times, pitches, c=velocities, cmap='viridis',
                            s=20, alpha=0.7, edgecolors='k', linewidth=0.5)
        
        # 添加三角形投影
        if triangles:
            for tri in triangles[:min(100, len(triangles))]:
                tri_times = [features[i]['time'] for i in tri]
                tri_pitches = [features[i]['pitch'] for i in tri]
                tri_times.append(tri_times[0])
                tri_pitches.append(tri_pitches[0])
                ax2.plot(tri_times, tri_pitches, 'gray', alpha=0.15, linewidth=0.5)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pitch (MIDI)')
        plt.colorbar(scatter, ax=ax2, label='Velocity')
        ax2.grid(True, alpha=0.3)
        
        # 3. 网络拓扑
        ax3 = fig.add_subplot(233)
        ax3.set_title("Network Topology", fontsize=11, fontweight='bold')
        
        if triangles:
            # 创建简化的网络
            G = nx.Graph()
            for i in range(len(points)):
                G.add_node(i, pos=(points[i, 0], points[i, 1]))
            
            # 添加部分边
            edges = set()
            for tri in triangles[:]:
                for i in range(3):
                    for j in range(i+1, 3):
                        edges.add((tri[i], tri[j]))
            
            for edge in list(edges)[:]:
                G.add_edge(edge[0], edge[1])
            
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, ax=ax3, node_size=20, width=0.5,
                    node_color='lightblue', edge_color='gray', alpha=0.7)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Pitch')
        
        # 4. 和弦分布
        ax4 = fig.add_subplot(234)
        ax4.set_title("Chord Patterns", fontsize=11, fontweight='bold')
        
        if 'chord_patterns' in analysis:
            chord_data = analysis['chord_patterns']
            if chord_data:
                chords = list(chord_data.keys())
                counts = list(chord_data.values())
                
                bars = ax4.bar(range(len(chords)), counts, color='skyblue', edgecolor='black')
                ax4.set_xticks(range(len(chords)))
                ax4.set_xticklabels(chords, rotation=45, ha='right')
                ax4.set_ylabel('Count')
                
                # 添加数值标签
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{count}', ha='center', va='bottom', fontsize=8)
        
        # 5. 连通性分析
        ax5 = fig.add_subplot(235)
        ax5.set_title("Connectivity Analysis", fontsize=11, fontweight='bold')
        
        if 'connectivity' in analysis:
            conn = analysis['connectivity']
            metrics = ['Components', 'Avg Degree']
            values = [conn.get('num_components', 0), conn.get('avg_degree', 0)]
            
            bars = ax5.bar(metrics, values, color=['lightgreen', 'salmon'], edgecolor='black')
            ax5.set_ylabel('Value')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 6. 统计信息
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        info_text = f"""
        Music Spacetime Analysis
        {'='*40}
        
        MIDI: {midi_name}
        
        Geometry:
        • Points: {len(points)}
        • Triangles: {len(triangles)}
        • Density: {len(triangles)/max(1, len(points)):.1f} triangles/point
        
        Music Features:
        • Time: [{min(times):.1f}, {max(times):.1f}] s
        • Pitch: [{min(pitches)}, {max(pitches)}]
        • Velocity: [{min(velocities):.3f}, {max(velocities):.3f}]
        
        Structure:
        • Chord Types: {len(analysis.get('chord_patterns', {}))}
        • Components: {analysis.get('connectivity', {}).get('num_components', 0)}
        """
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.suptitle(f"Music Spacetime Triangulation: {midi_name}", 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Visualization saved to {save_path}")
