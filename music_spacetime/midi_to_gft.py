

from music_spacetime.configuration import RobustTriangulationConfig
from music_spacetime.triangulation.triangulator import MusicSpacetimeTriangulator
from music_spacetime.visualization.music_spacetime_visualizer import MusicSpacetimeVisualizer
import os, json
from typing import List, Tuple
import numpy as np
from io_helpers.json_encoders import NumpyEncoder


def process_midi_with_robust_triangulation(midi_path: str = "assets/first_rabbit.mid", json_output_file=""):
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
        
        if json_output_file == "":
            output_file = f"./outputs/json/{midi_name}_music_spacetime.json"
        else:
            output_file = json_output_file
            
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
                json.dump(gft_data, f, indent=2, cls=NumpyEncoder)
            
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


