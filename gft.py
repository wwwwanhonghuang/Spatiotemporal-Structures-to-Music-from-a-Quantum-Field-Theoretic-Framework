"""
GFT-SPACETIME SIMULATOR (Engineering Version)
Purpose: Generate complex simplicial geometries for EEG-to-Music mapping
Optimized for: Speed, controllability, and interpretability
"""

import numpy as np
import random
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree

# ==================== 配置类 ====================
@dataclass
class GFTConfig:
    """GFT模拟器配置"""
    # 几何参数
    num_group_elements: int = 24      # 群元素数量
    target_vertices: int = 20         # 目标顶点数
    target_triangles: int = 200       # 目标三角形数
    
    # 物理参数
    mass_squared: float = 0.02        # 质量项
    lambda_coupling: float = 0.4      # 相互作用强度
    temperature: float = 1.0          # 温度（逆温度β）
    
    # 算法参数
    mcmc_steps: int = 3000            # MCMC步数
    batch_size: int = 6               # 批处理大小
    init_triangles: int = 40          # 初始三角形数
    
    # 几何约束
    max_edge_length: float = 1.2      # 最大边长
    min_triangle_quality: float = 0.3 # 最小三角形质量
    
    # 输出控制
    verbose: bool = True              # 详细输出
    save_intermediate: bool = False   # 保存中间结果

# ==================== 高效几何群 ====================
class EfficientGeometricGroup:
    """高效几何群：优化的球面点生成和邻居查找"""
    
    def __init__(self, size: int):
        self.size = size
        self.points = self._generate_fibonacci_sphere(size)
        self.kdtree = KDTree(self.points)
        self._build_neighbor_cache()
        
    def _generate_fibonacci_sphere(self, n: int) -> np.ndarray:
        """生成斐波那契螺旋点（均匀分布）"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角度
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _build_neighbor_cache(self):
        """预计算邻居关系"""
        self.neighbor_cache = {}
        for i, point in enumerate(self.points):
            # 查询最近的k个点
            distances, indices = self.kdtree.query(point, k=min(15, self.size))
            self.neighbor_cache[i] = {
                'indices': indices[1:],  # 排除自身
                'distances': distances[1:]
            }
    
    def get_neighbors(self, point_idx: int, max_dist: float = 1.2) -> List[int]:
        """快速获取邻近点"""
        cache = self.neighbor_cache[point_idx]
        neighbors = []
        for idx, dist in zip(cache['indices'], cache['distances']):
            if dist < max_dist:
                neighbors.append(idx)
        return neighbors
    
    def point_to_tuple(self, idx: int) -> tuple:
        """点索引转换为元组（作为字典键）"""
        return tuple(self.points[idx].round(6))  # 六位小数精度足够

# ==================== 工程GFT模型 ====================
class EngineeringGFTModel:
    """工程GFT模型：平衡准确性和效率"""
    
    def __init__(self, group: EfficientGeometricGroup, config: GFTConfig):
        self.group = group
        self.config = config
        self.point_positions = {i: group.points[i] for i in range(group.size)}
        
        # 性能优化
        self.quality_cache = {}
        self.action_cache = {}
        
    def calculate_triangle_quality(self, vertex_indices: Tuple[int, int, int]) -> float:
        """计算三角形质量（带缓存）"""
        if vertex_indices in self.quality_cache:
            return self.quality_cache[vertex_indices]
        
        p1, p2, p3 = [self.point_positions[i] for i in vertex_indices]
        
        # 计算边长
        edges = [
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p3 - p1)
        ]
        
        # 质量 = 1 - 归一化标准差
        mean_len = np.mean(edges)
        std_len = np.std(edges)
        if mean_len > 0:
            quality = 1.0 - min(1.0, std_len / mean_len)
        else:
            quality = 0.0
        
        # 惩罚过长或过短的边
        for length in edges:
            if length > self.config.max_edge_length or length < 0.1:
                quality *= 0.5
        
        self.quality_cache[vertex_indices] = quality
        return quality
    
    def triangle_action(self, triangle: Tuple[int, int, int], phi_value: complex) -> float:
        """单个三角形的能量贡献"""
        quality = self.calculate_triangle_quality(triangle)
        
        # 动能项：质量项 + 几何质量
        kinetic = self.config.mass_squared * abs(phi_value) ** 2
        kinetic -= 0.1 * quality * abs(phi_value) ** 2  # 质量好的三角形能量低
        
        return kinetic
    
    def interaction_action(self, triangle1: tuple, phi1: complex, 
                          triangle2: tuple, phi2: complex) -> float:
        """两个三角形的相互作用能量"""
        # 计算共享边数量
        shared_vertices = len(set(triangle1) & set(triangle2))
        
        if shared_vertices >= 2:  # 共享一条边
            # 检查是否可以形成四面体的一部分
            interaction = self.config.lambda_coupling * abs(phi1 * phi2).real
            
            # 几何一致性奖励
            quality1 = self.calculate_triangle_quality(triangle1)
            quality2 = self.calculate_triangle_quality(triangle2)
            geom_factor = (quality1 + quality2) / 2
            
            return interaction * geom_factor
        
        return 0.0
    
    def total_action_fast(self, field_config: Dict[tuple, complex]) -> float:
        """快速计算总作用量（优化版）"""
        total = 0.0
        
        # 动能项
        for triangle, phi_val in field_config.items():
            total += self.triangle_action(triangle, phi_val)
        
        # 相互作用项（采样而非穷举）
        triangles = list(field_config.keys())
        if len(triangles) >= 2:
            # 随机采样相互作用对
            samples = min(200, len(triangles) * 10)
            for _ in range(samples):
                i, j = random.sample(range(len(triangles)), 2)
                tri1, tri2 = triangles[i], triangles[j]
                phi1, phi2 = field_config[tri1], field_config[tri2]
                
                interaction = self.interaction_action(tri1, phi1, tri2, phi2)
                total += interaction / samples  # 归一化
        
        return total

# ==================== 智能MCMC采样器 ====================
class SmartMCMCSampler:
    """智能MCMC采样器：自适应提案分布"""
    
    def __init__(self, model: EngineeringGFTModel, config: GFTConfig):
        self.model = model
        self.config = config
        self.acceptance_history = []
        
    def run(self, initial_config: Dict[tuple, complex], 
            steps: int = None) -> Dict[tuple, complex]:
        """运行智能MCMC"""
        if steps is None:
            steps = self.config.mcmc_steps
        
        current_config = initial_config.copy()
        current_action = self.model.total_action_fast(current_config)
        
        print("\n" + "="*50)
        print("SMART MCMC SAMPLING")
        print("="*50)
        
        start_time = time.time()
        step_times = []
        
        for step in range(steps):
            step_start = time.time()
            
            # 自适应批处理大小
            batch_size = self._adaptive_batch_size(step, steps)
            
            # 提出新构型
            new_config = self._smart_proposal(current_config, batch_size)
            new_action = self.model.total_action_fast(new_config)
            
            # Metropolis-Hastings
            delta_action = new_action - current_action
            if delta_action < 0 or random.random() < np.exp(-delta_action / self.config.temperature):
                current_config = new_config
                current_action = new_action
                self.acceptance_history.append(1)
            else:
                self.acceptance_history.append(0)
            
            # 定期清理（移除低场值的三角形）
            if step % 100 == 0:
                current_config = self._cleanup_config(current_config, threshold=0.05)
                current_action = self.model.total_action_fast(current_config)
            
            step_times.append(time.time() - step_start)
            
            # 进度报告
            if step % 500 == 0:
                avg_step_time = np.mean(step_times[-100:]) if step_times else 0
                remaining = (steps - step) * avg_step_time
                
                print(f"Step {step:4d}/{steps} | "
                      f"Action: {current_action:7.2f} | "
                      f"Triangles: {len(current_config):3d} | "
                      f"Accept: {np.mean(self.acceptance_history[-100:]):.2f} | "
                      f"ETA: {remaining:.0f}s")
        
        total_time = time.time() - start_time
        print(f"\nSampling completed in {total_time:.1f} seconds")
        print(f"Average acceptance rate: {np.mean(self.acceptance_history):.3f}")
        
        return current_config
    
    def _adaptive_batch_size(self, step: int, total_steps: int) -> int:
        """自适应批处理大小"""
        # 开始阶段：大胆探索
        if step < total_steps // 3:
            return self.config.batch_size + 2
        # 中间阶段：平衡
        elif step < 2 * total_steps // 3:
            return self.config.batch_size
        # 最后阶段：精细调整
        else:
            return max(1, self.config.batch_size - 2)
    
    def _smart_proposal(self, config: Dict[tuple, complex], batch_size: int) -> Dict[tuple, complex]:
        """智能提案：根据当前状态选择操作"""
        new_config = config.copy()
        
        for _ in range(batch_size):
            # 根据当前三角形数量决定操作概率
            num_triangles = len(new_config)
            
            if num_triangles < self.config.target_triangles // 3:
                # 三角形太少，优先添加
                operation = 'add' if random.random() < 0.8 else 'modify'
            elif num_triangles > self.config.target_triangles * 1.5:
                # 三角形太多，优先删除
                operation = 'remove' if random.random() < 0.6 else 'modify'
            else:
                # 平衡状态
                r = random.random()
                if r < 0.4:
                    operation = 'add'
                elif r < 0.7:
                    operation = 'modify'
                else:
                    operation = 'remove'
            
            # 执行操作
            if operation == 'add':
                new_config = self._propose_add(new_config)
            elif operation == 'remove' and new_config:
                new_config = self._propose_remove(new_config)
            elif operation == 'modify' and new_config:
                new_config = self._propose_modify(new_config)
        
        return new_config
    
    def _propose_add(self, config: Dict[tuple, complex]) -> Dict[tuple, complex]:
        """智能添加三角形"""
        new_config = config.copy()
        
        # 选择"活跃"区域添加（已有三角形多的区域）
        existing_vertices = set()
        for triangle in new_config.keys():
            existing_vertices.update(triangle)
        
        if existing_vertices:
            # 选择已有顶点作为基础
            base_idx = random.choice(list(existing_vertices))
        else:
            # 随机选择
            base_idx = random.randint(0, self.model.group.size - 1)
        
        # 查找邻近点
        neighbors = self.model.group.get_neighbors(base_idx, self.config.max_edge_length)
        
        if len(neighbors) >= 2:
            # 选择两个邻近点
            selected = random.sample(neighbors, 2)
            triangle = tuple(sorted([base_idx] + selected))
            
            # 检查三角形质量
            quality = self.model.calculate_triangle_quality(triangle)
            if quality >= self.config.min_triangle_quality:
                # 设置初始场值（与质量相关）
                init_value = 0.3 + 0.4 * quality  # 质量越好，初始值越大
                new_config[triangle] = complex(init_value, 0)
        
        return new_config
    
    def _propose_remove(self, config: Dict[tuple, complex]) -> Dict[tuple, complex]:
        """智能删除：优先删除质量差的三角形"""
        new_config = config.copy()
        
        if new_config:
            # 计算所有三角形的质量
            qualities = []
            triangles = list(new_config.keys())
            
            for tri in triangles:
                quality = self.model.calculate_triangle_quality(tri)
                qualities.append(quality)
            
            # 根据质量加权选择（质量越差，越可能被删除）
            weights = [1.0 - q for q in qualities]  # 质量差的权重高
            total_weight = sum(weights)
            
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                idx = random.choices(range(len(triangles)), weights=weights)[0]
                del new_config[triangles[idx]]
        
        return new_config
    
    def _propose_modify(self, config: Dict[tuple, complex]) -> Dict[tuple, complex]:
        """智能修改场值"""
        new_config = config.copy()
        
        if new_config:
            triangle = random.choice(list(new_config.keys()))
            current_value = new_config[triangle]
            
            # 修改幅度与三角形质量相关
            quality = self.model.calculate_triangle_quality(triangle)
            max_change = 0.2 + 0.1 * (1.0 - quality)  # 质量越差，允许更大变化
            
            change = complex(random.uniform(-max_change, max_change), 0)
            new_value = current_value + change
            
            # 确保场值在合理范围
            if abs(new_value) > 0.01 and abs(new_value) < 2.0:
                new_config[triangle] = new_value
        
        return new_config
    
    def _cleanup_config(self, config: Dict[tuple, complex], threshold: float = 0.05) -> Dict[tuple, complex]:
        """清理低场值的三角形"""
        return {k: v for k, v in config.items() if abs(v) > threshold}

# ==================== 几何分析器 ====================
class GeometryAnalyzer:
    """几何结构分析器"""
    
    @staticmethod
    def analyze(config: Dict[tuple, complex], group: EfficientGeometricGroup) -> Dict:
        """分析几何结构"""
        triangles = list(config.keys())
        
        # 提取顶点
        vertices = set()
        for tri in triangles:
            vertices.update(tri)
        
        # 计算统计
        edge_lengths = []
        triangle_qualities = []
        field_values = []
        
        for tri, phi_val in config.items():
            # 边长
            p1, p2, p3 = [group.points[i] for i in tri]
            edges = [
                np.linalg.norm(p1 - p2),
                np.linalg.norm(p2 - p3),
                np.linalg.norm(p3 - p1)
            ]
            edge_lengths.extend(edges)
            
            # 三角形质量（简单计算）
            mean_len = np.mean(edges)
            std_len = np.std(edges)
            if mean_len > 0:
                quality = 1.0 - min(1.0, std_len / mean_len)
                triangle_qualities.append(quality)
            
            # 场值
            field_values.append(abs(phi_val))
        
        # 构建邻接图
        G = nx.Graph()
        for v in vertices:
            G.add_node(v)
        
        # 添加边
        edges_added = set()
        for tri in triangles:
            for i in range(len(tri)):
                for j in range(i+1, len(tri)):
                    edge = tuple(sorted((tri[i], tri[j])))
                    if edge not in edges_added:
                        G.add_edge(edge[0], edge[1])
                        edges_added.add(edge)
        
        # 计算拓扑性质
        components = list(nx.connected_components(G))
        avg_degree = sum(dict(G.degree()).values()) / max(len(G.nodes()), 1)
        
        return {
            'num_vertices': len(vertices),
            'num_triangles': len(triangles),
            'num_components': len(components),
            'avg_edge_length': np.mean(edge_lengths) if edge_lengths else 0,
            'std_edge_length': np.std(edge_lengths) if edge_lengths else 0,
            'avg_triangle_quality': np.mean(triangle_qualities) if triangle_qualities else 0,
            'avg_field_value': np.mean(field_values) if field_values else 0,
            'max_component_size': max(len(c) for c in components) if components else 0,
            'avg_vertex_degree': avg_degree,
            'is_connected': (len(components) == 1)
        }

# ==================== 可视化器 ====================
class GFTVisualizer:
    """GFT结果可视化"""
    
    @staticmethod
    def visualize_2d(config: Dict[tuple, complex], group: EfficientGeometricGroup, 
                    filename: str = 'gft_complex.png'):
        """2D可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 网络结构
        G = nx.Graph()
        vertices = set()
        for tri in config.keys():
            vertices.update(tri)
        
        for v in vertices:
            G.add_node(v)
        
        edges_added = set()
        for tri in config.keys():
            for i in range(len(tri)):
                for j in range(i+1, len(tri)):
                    edge = tuple(sorted((tri[i], tri[j])))
                    if edge not in edges_added:
                        G.add_edge(edge[0], edge[1])
                        edges_added.add(edge)
        
        # 使用点位置布局
        pos = {i: (group.points[i][0], group.points[i][1]) for i in vertices}
        
        nx.draw(G, pos, ax=axes[0, 0], node_size=50, width=0.8, 
                node_color='lightblue', edge_color='gray', alpha=0.8)
        axes[0, 0].set_title(f"Network Structure\n{len(vertices)} vertices, {len(config)} triangles")
        axes[0, 0].axis('equal')
        
        # 2. 场值分布
        field_values = [abs(v) for v in config.values()]
        axes[0, 1].hist(field_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(field_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(field_values):.3f}')
        axes[0, 1].set_xlabel('Field Amplitude |φ|')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Field Value Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 边长分布
        edge_lengths = []
        for tri in config.keys():
            p1, p2, p3 = [group.points[i] for i in tri]
            edges = [np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p3-p1)]
            edge_lengths.extend(edges)
        
        axes[1, 0].hist(edge_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(np.mean(edge_lengths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(edge_lengths):.3f}')
        axes[1, 0].set_xlabel('Edge Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Edge Length Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 顶点度分布
        degree_dist = {}
        for tri in config.keys():
            for v in tri:
                degree_dist[v] = degree_dist.get(v, 0) + 1
        
        degrees = list(degree_dist.values())
        axes[1, 1].hist(degrees, bins=15, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 1].axvline(np.mean(degrees), color='red', linestyle='--',
                          label=f'Mean: {np.mean(degrees):.2f}')
        axes[1, 1].set_xlabel('Vertex Degree')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Vertex Degree Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'GFT Complex Analysis | Vertices: {len(vertices)} | Triangles: {len(config)}', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {filename}")

# ==================== 主控制器 ====================
class GFTController:
    """GFT模拟主控制器"""
    
    def __init__(self, config: GFTConfig = None):
        self.config = config or GFTConfig()
        self.group = None
        self.model = None
        self.sampler = None
        self.results = {}
        
    def initialize(self):
        """初始化所有组件"""
        print("Initializing GFT simulator...")
        
        # 1. 创建几何群
        self.group = EfficientGeometricGroup(self.config.num_group_elements)
        print(f"  • Geometric group: {self.group.size} points")
        
        # 2. 创建GFT模型
        self.model = EngineeringGFTModel(self.group, self.config)
        print(f"  • GFT model: mass²={self.config.mass_squared}, λ={self.config.lambda_coupling}")
        
        # 3. 创建采样器
        self.sampler = SmartMCMCSampler(self.model, self.config)
        print(f"  • MCMC sampler: {self.config.mcmc_steps} steps, batch={self.config.batch_size}")
        
        print("Initialization complete.\n")
    
    def create_initial_config(self) -> Dict[tuple, complex]:
        """创建初始场构型"""
        print("Creating initial configuration...")
        
        initial_config = {}
        created = 0
        
        # 创建初始三角形网格
        for base_idx in range(min(15, self.group.size)):
            neighbors = self.group.get_neighbors(base_idx, self.config.max_edge_length)
            
            if len(neighbors) >= 2:
                # 创建2-3个三角形
                for _ in range(random.randint(1, 3)):
                    selected = random.sample(neighbors, 2)
                    triangle = tuple(sorted([base_idx] + selected))
                    
                    # 检查质量
                    quality = self.model.calculate_triangle_quality(triangle)
                    if quality >= self.config.min_triangle_quality:
                        initial_config[triangle] = complex(0.4 + 0.2 * quality, 0)
                        created += 1
        
        print(f"  • Created {created} initial triangles")
        return initial_config
    
    def run_simulation(self) -> Dict[tuple, complex]:
        """运行完整模拟"""
        print("\n" + "="*60)
        print("STARTING GFT SPACETIME SIMULATION")
        print("="*60)
        
        total_start = time.time()
        
        # 1. 初始化
        self.initialize()
        
        # 2. 创建初始构型
        initial_config = self.create_initial_config()
        
        # 3. 运行MCMC
        mcmc_start = time.time()
        final_config = self.sampler.run(initial_config)
        mcmc_time = time.time() - mcmc_start
        
        # 4. 分析结果
        analysis_start = time.time()
        analyzer = GeometryAnalyzer()
        analysis = analyzer.analyze(final_config, self.group)
        analysis_time = time.time() - analysis_start
        
        total_time = time.time() - total_start
        
        # 存储结果
        self.results = {
            'config': self.config.__dict__,
            'analysis': analysis,
            'final_config_size': len(final_config),
            'timing': {
                'total': total_time,
                'mcmc': mcmc_time,
                'analysis': analysis_time
            },
            'performance': {
                'mcmc_steps_per_second': self.config.mcmc_steps / mcmc_time,
                'acceptance_rate': np.mean(self.sampler.acceptance_history)
            }
        }
        
        # 输出结果
        self._print_results(analysis, total_time)
        
        return final_config
    
    def _print_results(self, analysis: Dict, total_time: float):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"\nGeometry Statistics:")
        print(f"  • Vertices: {analysis['num_vertices']}")
        print(f"  • Triangles: {analysis['num_triangles']}")
        print(f"  • Connected components: {analysis['num_components']}")
        print(f"  • Largest component: {analysis['max_component_size']} vertices")
        print(f"  • Average vertex degree: {analysis['avg_vertex_degree']:.2f}")
        print(f"  • Is connected: {'Yes' if analysis['is_connected'] else 'No'}")
        
        print(f"\nGeometric Quality:")
        print(f"  • Average edge length: {analysis['avg_edge_length']:.3f} ± {analysis['std_edge_length']:.3f}")
        print(f"  • Average triangle quality: {analysis['avg_triangle_quality']:.3f}")
        print(f"  • Average field value: {analysis['avg_field_value']:.3f}")
        
        print(f"\nPerformance:")
        print(f"  • MCMC steps/sec: {self.results['performance']['mcmc_steps_per_second']:.1f}")
        print(f"  • Acceptance rate: {self.results['performance']['acceptance_rate']:.3f}")
        
        print("\n" + "="*60)
    
    def save_results(self, final_config: Dict[tuple, complex], 
                    prefix: str = "gft_simulation"):
        """保存所有结果"""
        # 1. 保存配置和结果
        results_file = f"{prefix}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 2. 保存几何数据（用于后续处理）
        geometry_file = f"{prefix}_geometry.json"
        geometry_data = {
            'vertices': list({int(v) for tri in final_config.keys() for v in tri}),
            'triangles': [[int(v) for v in tri] for tri in final_config.keys()],
            'field_values': {f"{k[0]},{k[1]},{k[2]}": float(v.real) for k, v in final_config.items()}
        }
        with open(geometry_file, 'w') as f:
            json.dump(geometry_data, f, indent=2)
        
        # 3. 可视化
        visualizer = GFTVisualizer()
        visualizer.visualize_2d(final_config, self.group, f"{prefix}_visualization.png")
        
        print(f"\nResults saved:")
        print(f"  • Configuration & analysis: {results_file}")
        print(f"  • Geometry data: {geometry_file}")
        print(f"  • Visualization: {prefix}_visualization.png")

# ==================== 主函数 ====================
def main():
    """主函数：运行工程版GFT模拟"""
    
    # 配置模拟（可调整这些参数）
    config = GFTConfig(
        num_group_elements=24,
        target_vertices=25,
        target_triangles=250,
        mass_squared=0.02,
        lambda_coupling=0.35,
        mcmc_steps=2500,  # 减少步数以加快速度
        batch_size=5,
        verbose=True
    )
    
    # 创建控制器并运行
    controller = GFTController(config)
    
    try:
        final_config = controller.run_simulation()
        
        # 保存结果
        controller.save_results(final_config, "gft_engineering")
        
        print("\n" + "="*60)
        print("ENGINEERING GFT SIMULATION COMPLETE!")
        print("Ready for EEG-to-Music mapping pipeline.")
        print("="*60)
        
        return final_config, controller.results
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # 运行模拟
    final_config, results = main()
    
    # 可以在这里添加后续处理
    if final_config is not None:
        print("\nNext steps for EEG-to-Music mapping:")
        print("1. Load EEG data and convert to point cloud")
        print("2. Use this GFT engine to generate geometric representation")
        print("3. Map GFT complex to musical structure")
        print("4. Synthesize music from the mapped structure")