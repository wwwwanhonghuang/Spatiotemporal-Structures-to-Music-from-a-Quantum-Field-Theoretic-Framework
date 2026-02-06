import numpy as np
from scipy.spatial import KDTree, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class GFTConnectivityOptimizer:
    """GFT连通性优化器 - 确保时空几何全联通"""
    
    def __init__(self, connectivity_threshold: float = 0.1):
        self.connectivity_threshold = connectivity_threshold
        
    def load_and_analyze(self, json_path: str) -> Dict:
        """加载并分析三角化数据"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        points = np.array(data['points'])
        triangles = [[int(p) for p in triangle] for triangle in data['triangles']]
        
        # 分析连通性
        analysis = self._analyze_connectivity(points, triangles)
        
        return {
            'original_points': points,
            'original_triangles': triangles,
            'original_features': data.get('features', []),
            'analysis': analysis
        }
    
    def _analyze_connectivity(self, points: np.ndarray, triangles: List[Tuple[int, int, int]]) -> Dict:
        """分析连通性"""
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for i in range(len(points)):
            G.add_node(i)
        
        # 添加边（来自三角形）
        edges = set()
        for tri in triangles:
            if len(tri) == 3:
                for i in range(3):
                    for j in range(i+1, 3):
                        edges.add(tuple(sorted([tri[i], tri[j]])))
        
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        
        # 找出连通组件
        components = list(nx.connected_components(G))
        component_sizes = [len(c) for c in components]
        
        # 找出孤立点
        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
        
        return {
            'num_components': len(components),
            'component_sizes': component_sizes,
            'largest_component': max(component_sizes) if component_sizes else 0,
            'isolated_nodes': isolated_nodes,
            'num_isolated': len(isolated_nodes),
            'total_edges': len(edges),
            'avg_degree': sum(dict(G.degree()).values()) / max(1, len(G.nodes())),
            'graph': G
        }
    
    def create_fully_connected_geometry(self, points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                       strategy: str = 'minimal_spanning') -> Dict:
        """创建全联通几何"""
        
        print(f"\nCreating fully connected geometry...")
        print(f"  • Original points: {len(points)}")
        print(f"  • Original triangles: {len(triangles)}")
        
        # 分析原始连通性
        analysis = self._analyze_connectivity(points, triangles)
        print(f"  • Original components: {analysis['num_components']}")
        print(f"  • Isolated points: {analysis['num_isolated']}")
        
        if analysis['num_components'] == 1 and analysis['num_isolated'] == 0:
            print("  • Already fully connected!")
            return {
                'points': points,
                'triangles': triangles,
                'added_edges': [],
                'added_triangles': []
            }
        
        # 根据策略进行全联通化
        if strategy == 'minimal_spanning':
            return self._minimal_spanning_connection(points, triangles, analysis)
        elif strategy == 'delaunay_complete':
            return self._delaunay_complete_connection(points, triangles)
        elif strategy == 'gft_optimized':
            return self._gft_optimized_connection(points, triangles, analysis)
        else:
            return self._minimal_spanning_connection(points, triangles, analysis)
    
    def _minimal_spanning_connection(self, points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                    analysis: Dict) -> Dict:
        """最小生成树连接 - 最轻量的全联通"""
        
        G = analysis['graph']
        components = list(nx.connected_components(G))
        
        # 如果只有一个组件但有孤立点
        if len(components) == 1:
            # 连接孤立点到最近的非孤立点
            isolated_nodes = analysis['isolated_nodes']
            connected_nodes = [node for node in G.nodes() if node not in isolated_nodes]
            
            if not connected_nodes:
                # 所有点都是孤立的，创建完全图的最小生成树
                return self._connect_all_isolated(points)
            
            added_edges = []
            added_triangles = []
            
            for isolated in isolated_nodes:
                # 找到最近的连接点
                distances = [np.linalg.norm(points[isolated] - points[other]) 
                           for other in connected_nodes]
                nearest_idx = np.argmin(distances)
                nearest = connected_nodes[nearest_idx]
                
                # 添加边
                added_edges.append((isolated, nearest))
                G.add_edge(isolated, nearest)
                
                # 为新边创建三角形（如果需要）
                # 找到第三个点形成三角形
                for potential in connected_nodes:
                    if potential != nearest:
                        # 检查是否形成有效三角形
                        if self._is_valid_triangle(points[isolated], points[nearest], points[potential]):
                            tri = tuple(sorted([isolated, nearest, potential]))
                            if tri not in triangles:
                                added_triangles.append(tri)
                                triangles.append(tri)
                                break
            
            return {
                'points': points,
                'triangles': triangles,
                'added_edges': added_edges,
                'added_triangles': added_triangles,
                'strategy': 'minimal_spanning'
            }
        
        else:
            # 多个组件，连接组件间
            return self._connect_components(points, triangles, components)
    
    def _connect_all_isolated(self, points: np.ndarray) -> Dict:
        """连接所有孤立点"""
        print("  • All points are isolated, creating minimal spanning tree...")
        
        # 创建完全图的最小生成树
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import pdist, squareform
        
        # 计算距离矩阵
        dist_matrix = squareform(pdist(points))
        
        # 最小生成树
        mst = minimum_spanning_tree(dist_matrix)
        
        # 转换为边
        added_edges = []
        rows, cols = mst.nonzero()
        for i, j in zip(rows, cols):
            if i < j:  # 避免重复
                added_edges.append((int(i), int(j)))
        
        # 基于MST边创建三角形
        added_triangles = []
        triangles = []
        
        # 对每个边，找到最近的点形成三角形
        for edge in added_edges:
            i, j = edge
            
            # 找到离这条边中点最近的其他点
            midpoint = (points[i] + points[j]) / 2
            distances = []
            
            for k in range(len(points)):
                if k != i and k != j:
                    dist = np.linalg.norm(points[k] - midpoint)
                    distances.append((k, dist))
            
            if distances:
                distances.sort(key=lambda x: x[1])
                nearest = distances[0][0]
                
                # 创建三角形
                tri = tuple(sorted([i, j, nearest]))
                if self._is_valid_triangle(points[i], points[j], points[nearest]):
                    added_triangles.append(tri)
                    triangles.append(tri)
        
        return {
            'points': points,
            'triangles': triangles,
            'added_edges': added_edges,
            'added_triangles': added_triangles,
            'strategy': 'minimal_spanning_tree'
        }
    
    def _connect_components(self, points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                           components: List[Set[int]]) -> Dict:
        """连接多个组件"""
        print(f"  • Connecting {len(components)} components...")
        
        added_edges = []
        added_triangles = triangles.copy()
        
        # 计算每个组件的中心
        component_centers = []
        for comp in components:
            comp_points = points[list(comp)]
            center = comp_points.mean(axis=0)
            component_centers.append(center)
        
        # 创建组件间的完全连接（最小生成树）
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        # 组件间距离矩阵
        n_components = len(components)
        comp_dist_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(i+1, n_components):
                # 找到两个组件间最近的点对
                min_dist = float('inf')
                best_pair = None
                
                for point_i in components[i]:
                    for point_j in components[j]:
                        dist = np.linalg.norm(points[point_i] - points[point_j])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (point_i, point_j)
                
                comp_dist_matrix[i, j] = min_dist
                comp_dist_matrix[j, i] = min_dist
        
        # 组件间最小生成树
        mst = minimum_spanning_tree(comp_dist_matrix)
        
        # 添加连接边
        rows, cols = mst.nonzero()
        for i, j in zip(rows, cols):
            if i < j:
                # 找到这两个组件间最近的点对
                comp_i = list(components[int(i)])
                comp_j = list(components[int(j)])
                
                min_dist = float('inf')
                best_pair = None
                
                for point_i in comp_i:
                    for point_j in comp_j:
                        dist = np.linalg.norm(points[point_i] - points[point_j])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (point_i, point_j)
                
                if best_pair:
                    added_edges.append(best_pair)
                    
                    # 为新边创建三角形
                    edge_i, edge_j = best_pair
                    
                    # 在各自组件内找最近的点形成三角形
                    for comp in [comp_i, comp_j]:
                        for point_k in comp:
                            if point_k != edge_i and point_k != edge_j:
                                tri = tuple(sorted([edge_i, edge_j, point_k]))
                                if self._is_valid_triangle(points[edge_i], points[edge_j], points[point_k]):
                                    if tri not in added_triangles:
                                        added_triangles.append(tri)
                                    break
        
        return {
            'points': points,
            'triangles': added_triangles,
            'added_edges': added_edges,
            'added_triangles': [t for t in added_triangles if t not in triangles],
            'strategy': 'component_connection'
        }
    
    def _delaunay_complete_connection(self, points: np.ndarray, triangles: List[Tuple[int, int, int]]) -> Dict:
        """Delaunay完全连接"""
        print("  • Using Delaunay for complete triangulation...")
        
        try:
            # 执行Delaunay三角剖分
            tri = Delaunay(points)
            
            # 获取所有三角形
            all_triangles = []
            for simplex in tri.simplices:
                if len(simplex) == 3:
                    tri_tuple = tuple(sorted(simplex))
                    if self._is_valid_triangle(points[simplex[0]], points[simplex[1]], points[simplex[2]]):
                        all_triangles.append(tri_tuple)
            
            # 找出新增的三角形
            original_set = set(triangles)
            new_triangles = [t for t in all_triangles if t not in original_set]
            
            # 找出新增的边
            original_edges = set()
            for t in triangles:
                for i in range(3):
                    for j in range(i+1, 3):
                        original_edges.add(tuple(sorted([t[i], t[j]])))
            
            new_edges = set()
            for t in new_triangles:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = tuple(sorted([t[i], t[j]]))
                        if edge not in original_edges:
                            new_edges.add(edge)
            
            return {
                'points': points,
                'triangles': all_triangles,
                'added_edges': list(new_edges),
                'added_triangles': new_triangles,
                'strategy': 'delaunay_complete'
            }
            
        except Exception as e:
            print(f"  • Delaunay failed: {e}, falling back to minimal spanning")
            return self._minimal_spanning_connection(points, triangles, 
                                                   self._analyze_connectivity(points, triangles))
    
    def _gft_optimized_connection(self, points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                 analysis: Dict) -> Dict:
        """GFT优化连接 - 考虑物理约束"""
        print("  • Creating GFT-optimized fully connected geometry...")
        
        # GFT-specific 考虑因素：
        # 1. 避免过长连接（保持局部性）
        # 2. 确保足够的三角形密度
        # 3. 保持几何平滑
        
        G = analysis['graph']
        components = list(nx.connected_components(G))
        
        # 如果只有一个组件
        if len(components) == 1:
            return self._optimize_single_component(points, triangles, G)
        else:
            # 先连接组件
            result = self._connect_components(points, triangles, components)
            # 然后优化
            return self._optimize_single_component(result['points'], result['triangles'], 
                                                  self._analyze_connectivity(result['points'], 
                                                                            result['triangles'])['graph'])
    
    def _optimize_single_component(self, points: np.ndarray, triangles: List[Tuple[int, int, int]], 
                                  G: nx.Graph) -> Dict:
        """优化单个组件"""
        
        # 检查孤立点
        isolated = [node for node in G.nodes() if G.degree(node) == 0]
        
        if not isolated:
            # 已经全连接，但可能需要增加三角形密度
            return self._increase_triangle_density(points, triangles)
        
        # 连接孤立点
        added_edges = []
        added_triangles = triangles.copy()
        
        for iso in isolated:
            # 找到k个最近邻
            kdtree = KDTree(points)
            distances, indices = kdtree.query(points[iso], k=min(10, len(points)))
            
            # 连接到最近的有效点
            for idx in indices[1:]:  # 跳过自身
                if G.degree(idx) > 0:  # 连接到已有连接的点
                    edge = tuple(sorted([iso, idx]))
                    added_edges.append(edge)
                    G.add_edge(iso, idx)
                    
                    # 创建三角形
                    for other in G.neighbors(idx):
                        if other != iso:
                            tri = tuple(sorted([iso, idx, other]))
                            if self._is_valid_triangle(points[iso], points[idx], points[other]):
                                if tri not in added_triangles:
                                    added_triangles.append(tri)
                                break
                    break
        
        return {
            'points': points,
            'triangles': added_triangles,
            'added_edges': added_edges,
            'added_triangles': [t for t in added_triangles if t not in triangles],
            'strategy': 'gft_optimized'
        }
    
    def _increase_triangle_density(self, points: np.ndarray, triangles: List[Tuple[int, int, int]]) -> Dict:
        """增加三角形密度"""
        
        # 检查是否需要更多三角形
        target_triangle_ratio = 2.0  # 目标：平均每个点属于2个三角形
        current_ratio = len(triangles) * 3 / max(1, len(points))
        
        if current_ratio >= target_triangle_ratio:
            return {
                'points': points,
                'triangles': triangles,
                'added_edges': [],
                'added_triangles': [],
                'strategy': 'already_dense'
            }
        
        print(f"  • Increasing triangle density (current: {current_ratio:.2f}, target: {target_triangle_ratio})")
        
        added_triangles = []
        
        # 使用KDTree寻找潜在的三角形
        kdtree = KDTree(points)
        
        for i in range(len(points)):
            # 找到最近的邻居
            distances, indices = kdtree.query(points[i], k=min(15, len(points)))
            
            # 尝试创建新三角形
            neighbors = indices[1:]  # 跳过自身
            
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    idx_j, idx_k = neighbors[j], neighbors[k]
                    
                    # 检查三角形是否已存在
                    tri = tuple(sorted([i, idx_j, idx_k]))
                    if tri in triangles or tri in added_triangles:
                        continue
                    
                    # 检查三角形有效性
                    if self._is_valid_triangle(points[i], points[idx_j], points[idx_k]):
                        added_triangles.append(tri)
                        
                        # 如果达到目标，停止
                        new_ratio = (len(triangles) + len(added_triangles)) * 3 / len(points)
                        if new_ratio >= target_triangle_ratio:
                            break
                
                if len(added_triangles) * 3 / len(points) >= target_triangle_ratio:
                    break
        
        # 找出新增的边
        original_edges = set()
        for t in triangles:
            for i in range(3):
                for j in range(i+1, 3):
                    original_edges.add(tuple(sorted([t[i], t[j]])))
        
        new_edges = set()
        for t in added_triangles:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([t[i], t[j]]))
                    if edge not in original_edges:
                        new_edges.add(edge)
        
        return {
            'points': points,
            'triangles': triangles + added_triangles,
            'added_edges': list(new_edges),
            'added_triangles': added_triangles,
            'strategy': 'density_increase'
        }
    
    def _is_valid_triangle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """检查三角形是否有效"""
        # 基本检查
        edges = [
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p3 - p1)
        ]
        
        # 避免退化
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        if area < 1e-10:
            return False
        
        # 边长限制
        if max(edges) > self.connectivity_threshold * 3:  # 允许稍长的连接
            return False
        
        return True
    
    def save_gft_ready_data(self, result: Dict, output_path: str):
        """保存GFT就绪数据"""
        
        points = result['points']
        triangles = result['triangles']
        
        # 为GFT创建场配置
        # 每个三角形分配一个场值
        field_config = {}
        for i, tri in enumerate(triangles):
            # 计算三角形特征作为初始场值
            if len(tri) == 3:
                v1, v2, v3 = points[tri[0]], points[tri[1]], points[tri[2]]
                center = (v1 + v2 + v3) / 3
                # 使用中心位置的特征作为场值
                field_value = np.linalg.norm(center) * 0.5 + 0.3  # 0.3-0.8范围
                field_config[tri] = complex(field_value, 0)
        
        # 创建顶点特征
        vertex_features = {}
        for i, point in enumerate(points):
            vertex_features[i] = {
                'position': point.tolist(),
                'degree': 0  # 将在下面计算
            }
        
        # 计算顶点度
        for tri in triangles:
            for vertex in tri:
                if vertex in vertex_features:
                    vertex_features[vertex]['degree'] += 1
        
        gft_data = {
            'metadata': {
                'num_points': len(points),
                'num_triangles': len(triangles),
                'connectivity_strategy': result.get('strategy', 'unknown'),
                'added_edges': len(result.get('added_edges', [])),
                'added_triangles': len(result.get('added_triangles', [])),
                'is_fully_connected': True
            },
            'geometry': {
                'points': points.tolist(),
                'triangles': triangles,
                'vertex_features': vertex_features
            },
            'gft_config': {
                'field_config': {f"{k[0]},{k[1]},{k[2]}": {'real': v.real, 'imag': v.imag} 
                               for k, v in field_config.items()},
                'mass_squared': 0.02,
                'coupling_constant': 0.1,
                'temperature': 1.0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(gft_data, f, indent=2)
        
        print(f"\nGFT-ready data saved to: {output_path}")
        print(f"  • Points: {len(points)}")
        print(f"  • Triangles: {len(triangles)}")
        print(f"  • Field configurations: {len(field_config)}")
        print(f"  • Strategy: {result.get('strategy', 'unknown')}")


def visualize_connectivity_comparison(original_data: Dict, connected_data: Dict, 
                                     save_path: str = "connectivity_comparison.png"):
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


# ==================== 主程序 ====================

def optimize_for_gft(json_file: str = "first_rabbit_music_spacetime.json"):
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


if __name__ == "__main__":
    # 运行优化
    result = optimize_for_gft("first_rabbit_music_spacetime.json")
    
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