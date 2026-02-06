import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mido
from typing import Dict, List, Tuple, Any, Set
import networkx as nx
from dataclasses import dataclass
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RobustTriangulationConfig:
    """稳健三角化配置"""
    # 网格化参数
    grid_resolution: int = 50             # 网格分辨率
    min_cell_density: float = 0.01        # 最小单元密度
    
    # 三角化参数
    max_edge_length: float = 0.15         # 最大边长
    min_triangle_angle: float = 15.0      # 最小角度（度）
    connect_neighbors: int = 6            # 连接最近邻数量
    
    # 音乐特征参数
    time_scale: float = 1.0              # 时间缩放
    pitch_scale: float = 0.1             # 音高缩放
    velocity_scale: float = 0.05         # 力度缩放

class MusicSpacetimeTriangulator:
    """音乐时空三角化器 - 专门为音乐数据设计"""
    
    def __init__(self, config: RobustTriangulationConfig = None):
        self.config = config or RobustTriangulationConfig()
        
    def load_midi_to_structured_grid(self, midi_path: str) -> Dict[str, Any]:
        """将MIDI加载到结构化网格"""
        print(f"Loading MIDI: {os.path.basename(midi_path)}")
        
        # 1. 解析MIDI
        notes = self._parse_midi(midi_path)
        
        # 2. 创建时空网格
        grid_data = self._create_structured_grid(notes)
        
        # 3. 从网格中提取点
        points = self._extract_grid_points(grid_data)
        features = self._extract_grid_features(grid_data)
        
        return {
            'notes': notes,
            'grid': grid_data,
            'points': points,
            'features': features
        }
    
    def _parse_midi(self, midi_path: str) -> List[Dict]:
        """解析MIDI文件"""
        midi = mido.MidiFile(midi_path)
        notes = []
        current_time = 0
        tempo = 500000
        
        for track in midi.tracks:
            track_time = 0
            active_notes = {}
            
            for msg in track:
                track_time += msg.time
                delta_seconds = mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
                current_time += delta_seconds
                
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
                elif msg.type == 'note_on' and msg.velocity > 0:
                    note_key = (msg.note, msg.channel)
                    active_notes[note_key] = {
                        'start_time': current_time,
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'channel': msg.channel
                    }
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    note_key = (msg.note, msg.channel)
                    if note_key in active_notes:
                        note_start = active_notes[note_key]
                        duration = current_time - note_start['start_time']
                        
                        notes.append({
                            'time': note_start['start_time'],
                            'note': note_start['note'],
                            'velocity': note_start['velocity'],
                            'duration': duration,
                            'channel': note_start['channel']
                        })
                        
                        del active_notes[note_key]
        
        return notes
    
    def _create_structured_grid(self, notes: List[Dict]) -> Dict[str, Any]:
        """创建结构化时空网格"""
        if not notes:
            return {}
        
        # 确定网格范围
        times = [note['time'] for note in notes]
        pitches = [note['note'] for note in notes]
        durations = [note['duration'] for note in notes]
        
        time_min, time_max = min(times), max(times)
        pitch_min, pitch_max = min(pitches), max(pitches)
        duration_max = max(durations) if durations else 0
        
        print(f"  • Time range: [{time_min:.1f}, {time_max:.1f}]")
        print(f"  • Pitch range: [{pitch_min}, {pitch_max}]")
        print(f"  • Duration max: {duration_max:.2f}")
        
        # 创建网格
        grid_res = self.config.grid_resolution
        grid = np.zeros((grid_res, grid_res, 3))  # 时间, 音高, 密度
        
        # 填充网格
        for note in notes:
            # 计算网格位置
            t_norm = (note['time'] - time_min) / max(1, time_max - time_min)
            p_norm = (note['note'] - pitch_min) / max(1, pitch_max - pitch_min)
            
            t_idx = min(int(t_norm * grid_res), grid_res - 1)
            p_idx = min(int(p_norm * grid_res), grid_res - 1)
            
            # 更新网格值
            grid[t_idx, p_idx, 0] += 1  # 计数
            grid[t_idx, p_idx, 1] = max(grid[t_idx, p_idx, 1], note['velocity'] / 127.0)  # 最大力度
            grid[t_idx, p_idx, 2] = max(grid[t_idx, p_idx, 2], 
                                      min(1.0, note['duration'] / duration_max))  # 持续时间
        
        return {
            'grid': grid,
            'time_range': (time_min, time_max),
            'pitch_range': (pitch_min, pitch_max),
            'resolution': grid_res
        }
    
    def _extract_grid_points(self, grid_data: Dict[str, Any]) -> np.ndarray:
        """从网格中提取点"""
        grid = grid_data['grid']
        grid_res = grid_data['resolution']
        
        points = []
        
        # 遍历网格，为每个非空单元创建点
        for i in range(grid_res):
            for j in range(grid_res):
                if grid[i, j, 0] > 0:  # 有音符的单元
                    # 3D坐标：时间, 音高, 力度
                    point = np.array([
                        i / grid_res * self.config.time_scale,
                        j / grid_res * self.config.pitch_scale,
                        grid[i, j, 1] * self.config.velocity_scale  # 力度
                    ])
                    points.append(point)
        
        if points:
            points_array = np.vstack(points)
            # 归一化
            for dim in range(3):
                min_val = points_array[:, dim].min()
                max_val = points_array[:, dim].max()
                if max_val > min_val:
                    points_array[:, dim] = (points_array[:, dim] - min_val) / (max_val - min_val)
            return points_array
        else:
            return np.array([])
    
    def _extract_grid_features(self, grid_data: Dict[str, Any]) -> List[Dict]:
        """提取特征"""
        grid = grid_data['grid']
        grid_res = grid_data['resolution']
        time_min, time_max = grid_data['time_range']
        pitch_min, pitch_max = grid_data['pitch_range']
        
        features = []
        
        for i in range(grid_res):
            for j in range(grid_res):
                if grid[i, j, 0] > 0:
                    # 计算实际时间和音高
                    time_val = time_min + (i / grid_res) * (time_max - time_min)
                    pitch_val = pitch_min + (j / grid_res) * (pitch_max - pitch_min)
                    
                    features.append({
                        'time': time_val,
                        'pitch': int(pitch_val),
                        'velocity_norm': grid[i, j, 1],
                        'duration_norm': grid[i, j, 2],
                        'note_count': grid[i, j, 0],
                        'grid_position': (i, j)
                    })
        
        return features
    
    def create_music_triangulation(self, points: np.ndarray) -> List[Tuple[int, int, int]]:
        """为音乐数据创建三角化"""
        if len(points) < 3:
            return []
        
        print(f"\nCreating music triangulation for {len(points)} points...")
        
        # 方法1：尝试Delaunay
        triangles_delaunay = self._try_delaunay(points)
        if triangles_delaunay:
            print(f"  • Delaunay created {len(triangles_delaunay)} triangles")
            return triangles_delaunay
        
        # 方法2：凸包三角化
        print("  • Delaunay failed, trying convex hull triangulation...")
        triangles_hull = self._convex_hull_triangulation(points)
        if triangles_hull:
            print(f"  • Convex hull created {len(triangles_hull)} triangles")
            return triangles_hull
        
        # 方法3：最近邻连接
        print("  • Hull failed, trying nearest neighbor connections...")
        triangles_nn = self._nearest_neighbor_triangulation(points)
        print(f"  • Nearest neighbor created {len(triangles_nn)} triangles")
        return triangles_nn
    
    def _try_delaunay(self, points: np.ndarray) -> List[Tuple[int, int, int]]:
        """尝试Delaunay三角化"""
        try:
            # 添加随机扰动避免共面
            if len(points) > 0:
                noise = np.random.normal(0, 1e-6, points.shape)
                points_perturbed = points + noise
            else:
                points_perturbed = points
            
            tri = Delaunay(points_perturbed)
            triangles = []
            
            for simplex in tri.simplices:
                if len(simplex) == 3:
                    # 检查三角形质量
                    p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                    if self._is_valid_triangle(p1, p2, p3):
                        triangles.append(tuple(sorted(simplex)))
            
            return triangles
        except:
            return []
    
    def _convex_hull_triangulation(self, points: np.ndarray) -> List[Tuple[int, int, int]]:
        """凸包三角化"""
        try:
            if len(points) < 4:
                # 对于少量点，直接连接
                triangles = []
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        for k in range(j+1, len(points)):
                            if self._is_valid_triangle(points[i], points[j], points[k]):
                                triangles.append((i, j, k))
                return triangles
            
            # 计算凸包
            hull = ConvexHull(points)
            hull_points = hull.vertices
            
            # 凸包三角化（耳朵剪裁法简化版）
            triangles = []
            
            # 简单方法：从凸包顶点创建三角形扇形
            center = points.mean(axis=0)
            center_idx = len(points)  # 临时中心点
            
            # 对凸包顶点排序
            hull_points_sorted = list(hull_points)
            hull_points_sorted.sort(key=lambda i: np.arctan2(points[i,1]-center[1], 
                                                           points[i,0]-center[0]))
            
            # 创建三角形
            n = len(hull_points_sorted)
            for i in range(n):
                j = (i + 1) % n
                k = (i + 2) % n
                
                if j != k:
                    tri = (hull_points_sorted[i], hull_points_sorted[j], hull_points_sorted[k])
                    p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
                    
                    if self._is_valid_triangle(p1, p2, p3):
                        triangles.append(tri)
            
            return triangles
            
        except:
            return []
    
    def _nearest_neighbor_triangulation(self, points: np.ndarray) -> List[Tuple[int, int, int]]:
        """最近邻三角化"""
        if len(points) < 3:
            return []
        
        triangles = []
        kdtree = KDTree(points)
        
        # 为每个点连接其最近邻形成三角形
        for i in range(len(points)):
            # 寻找最近邻
            distances, indices = kdtree.query(points[i], 
                                            k=min(self.config.connect_neighbors + 1, 
                                                  len(points)))
            neighbors = indices[1:]  # 排除自身
            
            # 与最近邻形成三角形
            if len(neighbors) >= 2:
                for j in range(len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        tri = tuple(sorted([i, neighbors[j], neighbors[k]]))
                        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
                        
                        if self._is_valid_triangle(p1, p2, p3):
                            triangles.append(tri)
        
        # 去重
        unique_triangles = []
        seen = set()
        for tri in triangles:
            if tri not in seen:
                unique_triangles.append(tri)
                seen.add(tri)
        
        return unique_triangles
    
    def _is_valid_triangle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """检查三角形是否有效"""
        # 1. 检查边长
        edges = [
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p3 - p1)
        ]
        
        if max(edges) > self.config.max_edge_length:
            return False
        
        # 2. 检查最小角度
        # 计算角度
        a, b, c = edges[0], edges[1], edges[2]
        
        # 使用余弦定理
        if a > 0 and b > 0 and c > 0:
            cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
            cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
            
            # 转换为角度
            angle_A = np.degrees(np.arccos(max(-1, min(1, cos_A))))
            angle_B = np.degrees(np.arccos(max(-1, min(1, cos_B))))
            angle_C = np.degrees(np.arccos(max(-1, min(1, cos_C))))
            
            min_angle = min(angle_A, angle_B, angle_C)
            if min_angle < self.config.min_triangle_angle:
                return False
        
        # 3. 检查是否退化（三点共线）
        # 计算面积
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        if area < 1e-10:
            return False
        
        return True
    
    def create_mesh_graph(self, points: np.ndarray, triangles: List[Tuple[int, int, int]]) -> nx.Graph:
        """创建网格图"""
        G = nx.Graph()
        
        # 添加节点
        for i, point in enumerate(points):
            G.add_node(i, pos=tuple(point))
        
        # 添加边
        for tri in triangles:
            for i in range(3):
                for j in range(i+1, 3):
                    G.add_edge(tri[i], tri[j])
        
        return G
    
    def analyze_music_structure(self, points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                               features: List[Dict]) -> Dict[str, Any]:
        """分析音乐结构"""
        analysis = {
            'num_points': len(points),
            'num_triangles': len(triangles),
            'connectivity': {},
            'chord_patterns': {},
            'temporal_structure': {}
        }
        
        if len(triangles) == 0:
            return analysis
        
        # 1. 计算连通性
        G = self.create_mesh_graph(points, triangles)
        components = list(nx.connected_components(G))
        analysis['connectivity']['num_components'] = len(components)
        analysis['connectivity']['component_sizes'] = [len(c) for c in components]
        analysis['connectivity']['avg_degree'] = sum(dict(G.degree()).values()) / len(G)
        
        # 2. 分析和弦模式
        chord_counts = defaultdict(int)
        for tri in triangles[:1000]:  # 限制数量
            if len(tri) == 3:
                pitches = [features[i]['pitch'] % 12 for i in tri]
                chord_type = self._classify_chord(pitches)
                chord_counts[chord_type] += 1
        
        analysis['chord_patterns'] = dict(chord_counts)
        
        # 3. 时间结构分析
        if triangles:
            triangle_times = []
            for tri in triangles[:1000]:
                if len(tri) == 3:
                    avg_time = np.mean([features[i]['time'] for i in tri])
                    triangle_times.append(avg_time)
            
            if triangle_times:
                analysis['temporal_structure']['time_range'] = (min(triangle_times), max(triangle_times))
                analysis['temporal_structure']['time_distribution'] = np.histogram(triangle_times, bins=10)[0].tolist()
        
        return analysis
    
    def _classify_chord(self, pitches: List[int]) -> str:
        """分类和弦"""
        if len(pitches) < 3:
            return "incomplete"
        
        pitches = sorted(set(pitches))
        
        if len(pitches) < 3:
            return "dyad"
        
        # 计算音程
        intervals = []
        for i in range(len(pitches)-1):
            intervals.append((pitches[i+1] - pitches[i]) % 12)
        
        # 常见和弦
        if intervals == [4, 3] or intervals == [3, 4]:  # 大三和弦
            return "major_triad"
        elif intervals == [3, 3] or intervals == [3, 3]:  # 小三和弦
            return "minor_triad"
        elif intervals == [4, 4]:  # 增三和弦
            return "augmented_triad"
        elif intervals == [3, 4] or intervals == [4, 3]:  # 属七和弦（部分）
            return "dominant_partial"
        elif max(intervals) <= 2:  # 密集音群
            return "cluster"
        elif min(intervals) >= 5:  # 开放音程
            return "open_voicing"
        else:
            return "other"


class MusicSpacetimeVisualizer:
    """音乐时空可视化器"""
    
    @staticmethod
    def visualize_music_triangulation(points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                    features: List[Dict], analysis: Dict[str, Any],
                                    midi_name: str, save_path: str = None):
        """可视化音乐三角化"""
        if save_path is None:
            save_path = f"{midi_name}_music_spacetime.png"
        
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


# ==================== 主程序 ====================

def process_midi_with_robust_triangulation(midi_path: str = "first_rabbit.midi"):
    """使用稳健三角化处理MIDI"""
    print("="*70)
    print("ROBUST MUSIC SPACETIME TRIANGULATION")
    print("="*70)
    
    try:
        # 1. 初始化
        config = RobustTriangulationConfig(
            grid_resolution=80,
            max_edge_length=0.2,
            min_triangle_angle=10.0,
            connect_neighbors=8,
            time_scale=1.0,
            pitch_scale=0.08,
            velocity_scale=0.1
        )
        
        triangulator = MusicSpacetimeTriangulator(config)
        
        # 2. 加载MIDI到结构化网格
        data = triangulator.load_midi_to_structured_grid(midi_path)
        
        if len(data['points']) < 3:
            print("Error: Not enough points for triangulation")
            return None
        
        print(f"\nGrid analysis:")
        print(f"  • Grid points: {len(data['points'])}")
        print(f"  • Grid features: {len(data['features'])}")
        
        # 3. 创建三角化
        triangles = triangulator.create_music_triangulation(data['points'])
        
        if not triangles:
            print("\nWARNING: Still no triangles generated!")
            print("Trying alternative methods...")
            
            # 尝试强制生成一些三角形
            triangles = triangulator._force_triangle_generation(data['points'])
        
        print(f"\nTriangulation result:")
        print(f"  • Triangles generated: {len(triangles)}")
        
        # 4. 分析音乐结构
        analysis = triangulator.analyze_music_structure(data['points'], triangles, data['features'])
        
        # 5. 可视化
        midi_name = os.path.splitext(os.path.basename(midi_path))[0]
        visualizer = MusicSpacetimeVisualizer()
        visualizer.visualize_music_triangulation(
            data['points'], triangles, data['features'], analysis,
            midi_name, f"{midi_name}_robust_triangulation.png"
        )
        
        # 6. 保存数据
        output_data = {
            'metadata': {
                'midi_file': midi_path,
                'num_points': len(data['points']),
                'num_triangles': len(triangles),
                'grid_resolution': config.grid_resolution
            },
            'points': data['points'].tolist(),
            'triangles': triangles,
            'features': data['features'],
            'analysis': analysis
        }
        
        output_file = f"./outputs/json/{midi_name}_music_spacetime.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nData saved to: {output_file}")
        
        # 7. GFT-ready格式
        if triangles:
            gft_data = {
                'vertices': list(range(len(data['points']))),
                'vertex_positions': {str(i): list(data['points'][i]) for i in range(len(data['points']))},
                'triangles': triangles,
                'vertex_weights': {str(i): data['features'][i]['velocity_norm'] 
                                 for i in range(len(data['points']))}
            }
            
            gft_file = f"./outputs/json/{midi_name}_gft_input.json"
            with open(gft_file, 'w') as f:
                json.dump(gft_data, f, indent=2)
            
            print(f"GFT input saved to: {gft_file}")
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        
        return output_data
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def force_triangle_generation_method(points: np.ndarray, min_triangles: int = 100) -> List[Tuple[int, int, int]]:
    """强制生成三角形的方法"""
    triangles = []
    
    if len(points) < 3:
        return triangles
    
    # 方法1：完全连接所有点（对少量点）
    if len(points) <= 20:
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    triangles.append((i, j, k))
        return triangles[:min_triangles]
    
    # 方法2：基于K-means聚类
    from sklearn.cluster import KMeans
    
    # 聚类
    n_clusters = min(10, len(points) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(points)
    
    # 在每个簇内创建三角形
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) >= 3:
            # 取簇内点
            cluster_points = points[cluster_indices]
            
            # 简单三角化：连接每个点与两个最近邻
            for i in range(len(cluster_points)):
                # 计算到其他点的距离
                distances = []
                for j in range(len(cluster_points)):
                    if i != j:
                        dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                        distances.append((j, dist))
                
                distances.sort(key=lambda x: x[1])
                
                # 与前两个最近邻形成三角形
                if len(distances) >= 2:
                    j_idx = cluster_indices[distances[0][0]]
                    k_idx = cluster_indices[distances[1][0]]
                    tri = tuple(sorted([cluster_indices[i], j_idx, k_idx]))
                    triangles.append(tri)
    
    # 方法3：连接簇中心
    if n_clusters >= 3:
        centers = kmeans.cluster_centers_
        
        # 为每个中心点找最近的两个中心
        for i in range(len(centers)):
            distances = []
            for j in range(len(centers)):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append((j, dist))
            
            distances.sort(key=lambda x: x[1])
            
            if len(distances) >= 2:
                # 找到对应的原始点
                i_points = np.where(labels == i)[0]
                j_points = np.where(labels == distances[0][0])[0]
                k_points = np.where(labels == distances[1][0])[0]
                
                if i_points.size > 0 and j_points.size > 0 and k_points.size > 0:
                    tri = (i_points[0], j_points[0], k_points[0])
                    triangles.append(tri)
    
    # 去重
    unique_triangles = []
    seen = set()
    for tri in triangles:
        if tri not in seen:
            unique_triangles.append(tri)
            seen.add(tri)
    
    return unique_triangles[:min_triangles]


# 添加到MusicSpacetimeTriangulator类中
MusicSpacetimeTriangulator._force_triangle_generation = lambda self, points: force_triangle_generation_method(points)


if __name__ == "__main__":
    # 处理MIDI文件
    result = process_midi_with_robust_triangulation("asserts/first_rabbit.mid")
    
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