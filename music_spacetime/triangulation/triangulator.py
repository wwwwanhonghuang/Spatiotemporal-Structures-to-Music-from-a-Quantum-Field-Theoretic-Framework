

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
from music_spacetime.configuration import RobustTriangulationConfig

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
    def ensure_vertex_coverage(self, points, triangles):
    
        used = set(i for tri in triangles for i in tri)
        unused = set(range(len(points))) - used

        if not unused:
            return triangles

        print(f"  • Repairing {len(unused)} isolated vertices")

        kdt = KDTree(points)

        for u in unused:

            _, idx = kdt.query(points[u], k=min(3, len(points)))

            tri = tuple(sorted(idx))
            triangles.append(tri)

        return triangles

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
        
        grid = grid_data['grid']
        grid_res = grid_data['resolution']

        time_min, time_max = grid_data['time_range']
        pitch_min, pitch_max = grid_data['pitch_range']

        points = []

        for i in range(grid_res):
            for j in range(grid_res):

                if grid[i, j, 0] > 0:

                    time_val = time_min + (i / grid_res) * (time_max - time_min)
                    pitch_val = pitch_min + (j / grid_res) * (pitch_max - pitch_min)

                    point = np.array([
                        time_val * self.config.time_scale,
                        pitch_val * self.config.pitch_scale,
                        grid[i, j, 1] * self.config.velocity_scale
                    ])

                    points.append(point)

        if points:
            return np.vstack(points)

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
    
    def create_music_triangulation(self, points):
    
        if len(points) < 3:
            return []

        print(f"\nCreating musical simplicial complex for {len(points)} points...")

        triangles = self._try_delaunay(points)

        if not triangles:
            print("Delaunay failed — fallback to nearest neighbor.")
            triangles = self._nearest_neighbor_triangulation(points)

        # --- repair coverage ---
        triangles = self.ensure_vertex_coverage(points, triangles)

        # --- verify connectivity ---
        G = self.create_mesh_graph(points, triangles)

        if not nx.is_connected(G):

            print("Graph still disconnected — performing secondary bridging.")

            components = list(nx.connected_components(G))
            triangles += self._bridge_components(points, components)

            G = self.create_mesh_graph(points, triangles)

            if not nx.is_connected(G):
                raise RuntimeError("Failed to construct connected spacetime.")

        # --- diagnostics ---
        print("  • Connected:", nx.is_connected(G))
        print("  • Avg degree:", np.mean([d for _, d in G.degree()]))
        print("  • Triangle count:", len(triangles))

        return triangles

    
    def _try_delaunay(self, points: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Robust 2D Delaunay triangulation with enforced connectivity.
        Uses time-pitch plane as base manifold.
        """

        if len(points) < 3:
            return []

        try:
            # Use only time and pitch
            pts2d = points[:, :2]

            # tiny jitter prevents degeneracy
            pts2d = pts2d + np.random.normal(0, 1e-9, pts2d.shape)

            tri = Delaunay(pts2d)
            triangles = [
                tuple(sorted(simplex))
                for simplex in tri.simplices
                if self._is_valid_triangle(
                    points[simplex[0]],
                    points[simplex[1]],
                    points[simplex[2]]
                )
            ]


            # --- enforce connectivity ---
            G = self.create_mesh_graph(points, triangles)
            components = list(nx.connected_components(G))

            if len(components) > 1:
                print(f"  • Found {len(components)} disconnected regions — bridging...")

                triangles += self._bridge_components(points, components)

            return triangles

        except Exception as e:
            print("Delaunay failed:", e)
            return []
    def _bridge_components(self, points, components):
        """
        Connect components by shortest-distance edges
        and convert them into triangles.
        """

        bridges = []

        comps = [list(c) for c in components]

        while len(comps) > 1:

            best_pair = None
            best_dist = float("inf")

            for i in range(len(comps)):
                for j in range(i+1, len(comps)):

                    A = np.array(comps[i])
                    B = np.array(comps[j])

                    dists = cdist(points[A], points[B])
                    idx = np.unravel_index(np.argmin(dists), dists.shape)

                    if dists[idx] < best_dist:
                        best_dist = dists[idx]
                        best_pair = (i, j, A[idx[0]], B[idx[1]])

            i, j, a, b = best_pair

            # find a third nearby point to form triangle
            kdt = KDTree(points)
            _, neigh = kdt.query(points[a], k=3)

            for c in neigh:
                if c != a and c != b:
                    bridges.append(tuple(sorted((a, b, c))))
                    break

            comps[i] = list(set(comps[i]) | set(comps[j]))
            comps.pop(j)

        return bridges

    
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
    

    def _is_valid_triangle(self, p1, p2, p3):
        
        # compute edges
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)

        if min(a, b, c) < 1e-6:
            return False

        # area check
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

        if area < 1e-10:
            return False

        # VERY gentle angle constraint
        cos_A = (b*b + c*c - a*a) / (2*b*c)
        cos_A = np.clip(cos_A, -1, 1)
        angle = np.degrees(np.arccos(cos_A))

        if angle < 2.0:
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
        
        analysis['topology'] = {
            "connected": nx.is_connected(G),
            "avg_degree": np.mean([d for _, d in G.degree()]),
            "num_vertices": len(points),
            "num_triangles": len(triangles)
        }

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