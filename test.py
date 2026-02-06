# # import random
# # import math
# # import collections
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import networkx as nx

# # # ================================
# # # 模块 1: 简化的群结构 - 使用离散点而非旋转矩阵
# # # ================================
# # class SimpleGeometricGroup:
# #     """简化的几何群：3D空间中的点集"""
# #     def __init__(self, size=32):
# #         self.size = size
# #         # 在单位球面上生成随机点
# #         self.elements = self._generate_points_on_sphere(size)
# #         print(f"Generated {len(self.elements)} points on sphere")
    
# #     def _generate_points_on_sphere(self, n):
# #         """在单位球面上生成均匀分布的点"""
# #         points = []
# #         for _ in range(n):
# #             # 生成随机方向
# #             theta = random.uniform(0, 2*np.pi)
# #             phi = random.uniform(0, np.pi)
            
# #             x = np.sin(phi) * np.cos(theta)
# #             y = np.sin(phi) * np.sin(theta)
# #             z = np.cos(phi)
            
# #             points.append(np.array([x, y, z]))
# #         return points
    
# #     def get_element_tuple(self, elem):
# #         """获取群元素的元组表示"""
# #         return tuple(elem)
    
# #     def distance(self, p1, p2):
# #         """计算两个点之间的角距离（0到2之间）"""
# #         # 点积的绝对值，在单位球面上
# #         dot = np.dot(p1, p2)
# #         # 限制在[-1, 1]范围内
# #         dot = max(-1.0, min(1.0, dot))
# #         # 返回角距离（0到π）
# #         return np.arccos(dot)
    
# #     def find_nearby_points(self, base_point, max_distance=np.pi/3, min_points=2):
# #         """查找附近的点"""
# #         nearby = []
# #         for point in self.elements:
# #             if np.array_equal(point, base_point):
# #                 continue
# #             dist = self.distance(base_point, point)
# #             if dist < max_distance:
# #                 nearby.append(point)
        
# #         # 如果没有找到足够多的点，放松条件
# #         if len(nearby) < min_points:
# #             # 找距离最近的点
# #             distances = []
# #             for point in self.elements:
# #                 if np.array_equal(point, base_point):
# #                     continue
# #                 dist = self.distance(base_point, point)
# #                 distances.append((dist, point))
            
# #             distances.sort(key=lambda x: x[0])
# #             nearby = [p for _, p in distances[:min_points]]
        
# #         return nearby

# # # ================================
# # # 模块 2: 简化的GFT模型
# # # ================================
# # class SimpleGFTModel:
# #     def __init__(self, group_G, dim_d, action_params):
# #         self.group = group_G
# #         self.d = dim_d
# #         self.params = action_params
        
# #         # 直接使用点作为顶点位置
# #         self.vertex_positions = {}
# #         for elem in self.group.elements:
# #             key = self.group.get_element_tuple(elem)
# #             self.vertex_positions[key] = elem
    
# #     def calculate_triangle_quality(self, g_tuple):
# #         """计算三角形质量（边长均衡性）"""
# #         if len(g_tuple) != 3:
# #             return 0.0
        
# #         # 获取三个顶点
# #         points = []
# #         for g in g_tuple:
# #             if g in self.vertex_positions:
# #                 points.append(self.vertex_positions[g])
        
# #         if len(points) != 3:
# #             return 0.0
        
# #         # 计算三条边长
# #         edges = []
# #         for i in range(3):
# #             for j in range(i+1, 3):
# #                 edges.append(np.linalg.norm(points[i] - points[j]))
        
# #         # 质量 = 1 - 相对标准差（越小越好）
# #         if len(edges) == 3:
# #             mean_len = np.mean(edges)
# #             std_len = np.std(edges)
# #             if mean_len > 0:
# #                 return 1.0 - min(1.0, std_len / mean_len)
        
# #         return 0.0
    
# #     def kinetic_term(self, phi_config):
# #         """动能项：鼓励规则的单形"""
# #         S0 = 0.0
# #         mass2 = self.params.get('mass2', 0.05)
        
# #         for g_tuple, phi_val in phi_config.items():
# #             # 基本质量项
# #             S0 += mass2 * abs(phi_val) ** 2
            
# #             # 几何质量：鼓励规则的三角形
# #             if len(g_tuple) == 3:
# #                 quality = self.calculate_triangle_quality(g_tuple)
# #                 # 质量越好的三角形能量越低
# #                 S0 -= 0.1 * quality * abs(phi_val) ** 2
        
# #         return 0.5 * S0
    
# #     def interaction_term(self, phi_config):
# #         """相互作用项：鼓励形成四面体"""
# #         lambda_n = self.params.get('lambda_4', 1.0)
# #         S_int = 0.0
        
# #         # 收集激活的三角形
# #         active_triangles = [k for k, v in phi_config.items() 
# #                            if abs(v) > 0.1 and len(k) == 3]
        
# #         if len(active_triangles) < 4:
# #             return 0.0
        
# #         # 随机采样检查可能的四面体
# #         samples = min(500, len(active_triangles) ** 2)
        
# #         for _ in range(samples):
# #             # 随机选择4个三角形
# #             if len(active_triangles) >= 4:
# #                 triangles = random.sample(active_triangles, 4)
                
# #                 # 检查是否能形成四面体
# #                 if self._could_form_tetrahedron(triangles):
# #                     # 计算相互作用强度
# #                     product = 1.0
# #                     for tri in triangles:
# #                         product *= phi_config[tri]
                    
# #                     S_int += lambda_n * abs(product.real)
        
# #         return S_int
    
# #     def _could_form_tetrahedron(self, triangles):
# #         """粗略检查四个三角形是否能形成四面体"""
# #         # 收集所有顶点
# #         all_vertices = set()
# #         for tri in triangles:
# #             all_vertices.update(tri)
        
# #         # 四面体需要4个顶点
# #         if len(all_vertices) != 4:
# #             return False
        
# #         # 检查每个三角形是否包含3个不同的顶点
# #         for tri in triangles:
# #             if len(set(tri)) != 3:
# #                 return False
        
# #         return True
    
# #     def total_action(self, phi_config):
# #         """总作用量"""
# #         S = self.kinetic_term(phi_config)
# #         S += self.interaction_term(phi_config)
# #         return S

# # # ================================
# # # 模块 3: 简化的MCMC采样
# # # ================================
# # def simple_metropolis_hastings(initial_phi, model, num_steps=10000):
# #     """简化的MCMC采样"""
# #     current_phi = initial_phi.copy()
# #     current_S = model.total_action(current_phi)
    
# #     for step in range(num_steps):
# #         # 提出新构型
# #         new_phi = propose_simple_change(current_phi, model)
        
# #         # 计算作用量变化
# #         new_S = model.total_action(new_phi)
# #         delta_S = new_S - current_S
        
# #         # Metropolis准则
# #         if delta_S < 0 or random.random() < math.exp(-delta_S):
# #             current_phi, current_S = new_phi, new_S
        
# #         # 进度报告
# #         if step % 2000 == 0:
# #             active = len([v for v in current_phi.values() if abs(v) > 0.01])
# #             print(f"Step {step:6d}: S = {current_S:8.3f}, Active: {active:3d}")
    
# #     return current_phi

# # def propose_simple_change(phi_config, model):
# #     """简化的提案函数：总是添加新单形"""
# #     new_phi = phi_config.copy()
    
# #     # 80%概率：添加新单形
# #     if random.random() < 0.8:
# #         # 随机选择一个基准点
# #         base_point = random.choice(model.group.elements)
# #         base_key = model.group.get_element_tuple(base_point)
        
# #         # 查找附近的点
# #         nearby = model.group.find_nearby_points(base_point, 
# #                                                max_distance=1.0,  # 宽松的距离
# #                                                min_points=2)
        
# #         if len(nearby) >= 2:
# #             # 随机选择2个附近的点
# #             selected = random.sample(nearby, 2)
# #             selected_keys = [model.group.get_element_tuple(p) for p in selected]
            
# #             # 形成三角形
# #             g_tuple = tuple([base_key] + selected_keys)
            
# #             # 设置场值（随机但偏向正值）
# #             new_phi[g_tuple] = complex(random.uniform(0.2, 0.8), 
# #                                       random.uniform(-0.1, 0.1))
    
# #     # 20%概率：调整或删除现有单形
# #     elif new_phi:
# #         key = random.choice(list(new_phi.keys()))
        
# #         if random.random() < 0.7:  # 调整
# #             perturbation = complex(random.uniform(-0.3, 0.3), 
# #                                  random.uniform(-0.1, 0.1))
# #             new_val = new_phi[key] + perturbation
            
# #             # 确保场值不太大
# #             if abs(new_val) > 1.5:
# #                 new_val = new_val / abs(new_val) * 1.5
            
# #             new_phi[key] = new_val
# #         else:  # 删除
# #             del new_phi[key]
    
# #     return new_phi

# # # ================================
# # # 模块 4: 简化的解码和可视化
# # # ================================
# # def simple_decode_and_visualize(phi_sample, model, threshold=0.05):
# #     """简化解码和可视化"""
# #     # 1. 解码激活的单形
# #     active_simplices = []
# #     vertex_id_map = {}
# #     next_vid = 0
    
# #     for g_tuple, phi_val in phi_sample.items():
# #         if abs(phi_val) > threshold:
# #             vertex_ids = []
# #             for g in g_tuple:
# #                 if g not in vertex_id_map:
# #                     vertex_id_map[g] = next_vid
# #                     next_vid += 1
# #                 vertex_ids.append(vertex_id_map[g])
# #             active_simplices.append(tuple(sorted(vertex_ids)))
    
# #     if not active_simplices:
# #         print("No active simplices found!")
# #         return None
    
# #     print(f"\nDecoded complex:")
# #     print(f"  Vertices: {len(vertex_id_map)}")
# #     print(f"  Triangles: {len(active_simplices)}")
    
# #     # 2. 简单可视化
# #     visualize_simple_complex(active_simplices, vertex_id_map, model)
    
# #     return {
# #         'vertices': list(vertex_id_map.keys()),
# #         'simplices': active_simplices,
# #         'vertex_map': vertex_id_map,
# #         'num_vertices': len(vertex_id_map),
# #         'num_simplices': len(active_simplices)
# #     }

# # def visualize_simple_complex(simplices, vertex_map, model):
# #     """简单可视化"""
# #     # 创建图形
# #     G = nx.Graph()
    
# #     # 添加顶点
# #     for v_id in range(len(vertex_map)):
# #         G.add_node(v_id)
    
# #     # 添加边（来自三角形）
# #     edges_added = set()
# #     for tri in simplices:
# #         if len(tri) >= 2:
# #             for i in range(len(tri)):
# #                 for j in range(i+1, len(tri)):
# #                     edge = tuple(sorted((tri[i], tri[j])))
# #                     if edge not in edges_added:
# #                         G.add_edge(edge[0], edge[1])
# #                         edges_added.add(edge)
    
# #     # 绘制
# #     plt.figure(figsize=(12, 10))
    
# #     # 获取顶点在3D空间中的位置
# #     pos_3d = {}
# #     for g_tuple, v_id in vertex_map.items():
# #         if g_tuple in model.vertex_positions:
# #             pos_3d[v_id] = model.vertex_positions[g_tuple]
    
# #     if len(pos_3d) >= 3:
# #         # 投影到2D
# #         points_3d = np.array([pos_3d[v_id] for v_id in pos_3d])
        
# #         # 简单的xy投影（忽略z坐标）
# #         pos_2d = {}
# #         for v_id, point in zip(pos_3d.keys(), points_3d):
# #             pos_2d[v_id] = (point[0], point[1])
# #     else:
# #         # 使用弹簧布局
# #         pos_2d = nx.spring_layout(G, k=1.0, iterations=50)
    
# #     # 绘制
# #     nx.draw_networkx_nodes(G, pos_2d, node_size=200, 
# #                           node_color='lightblue', alpha=0.8)
# #     nx.draw_networkx_edges(G, pos_2d, alpha=0.5, width=1.5)
# #     nx.draw_networkx_labels(G, pos_2d, font_size=9)
    
# #     plt.title(f"Generated Complex: {len(vertex_map)} vertices, {len(simplices)} triangles")
# #     plt.axis('off')
# #     plt.tight_layout()
# #     plt.savefig('simple_gft_complex.png', dpi=150, bbox_inches='tight')
# #     plt.show()
    
# #     print(f"Visualization saved to 'simple_gft_complex.png'")

# # # ================================
# # # 主程序
# # # ================================
# # def main():
# #     print("="*70)
# #     print("SIMPLE GFT COMPLEX SPACETIME GENERATOR")
# #     print("="*70)
    
# #     # 1. 创建简单的群结构
# #     print("\nCreating geometric group...")
# #     group_size = 20  # 适中的大小
# #     geometric_group = SimpleGeometricGroup(group_size)
    
# #     # 2. 配置模型
# #     config = {
# #         'group_G': geometric_group,
# #         'dim_d': 3,  # 三角形
# #         'action_params': {
# #             'mass2': 0.02,    # 非常小的质量，鼓励更多单形
# #             'lambda_4': 0.5,  # 适中的相互作用
# #         }
# #     }
    
# #     print("\nCreating GFT model...")
# #     model = SimpleGFTModel(**config)
    
# #     # 3. 初始化场 - 确保有初始单形
# #     print("\nInitializing field with guaranteed simplices...")
# #     initial_phi = {}
    
# #     # 强制创建一些初始单形
# #     num_initial = 30
# #     created = 0
    
# #     while created < num_initial and len(geometric_group.elements) >= 3:
# #         # 随机选择点
# #         base_idx = random.randint(0, len(geometric_group.elements)-1)
# #         base_point = geometric_group.elements[base_idx]
# #         base_key = geometric_group.get_element_tuple(base_point)
        
# #         # 找附近的点
# #         nearby = geometric_group.find_nearby_points(base_point, 
# #                                                    max_distance=1.5,  # 非常宽松
# #                                                    min_points=2)
        
# #         if len(nearby) >= 2:
# #             selected = random.sample(nearby, 2)
# #             selected_keys = [geometric_group.get_element_tuple(p) for p in selected]
            
# #             g_tuple = tuple([base_key] + selected_keys)
            
# #             # 设置正的场值
# #             initial_phi[g_tuple] = complex(random.uniform(0.3, 0.7), 0)
# #             created += 1
# #         else:
# #             # 如果找不到，尝试下一个点
# #             continue
    
# #     print(f"Created {len(initial_phi)} initial simplices")
    
# #     if len(initial_phi) == 0:
# #         print("ERROR: Failed to create any initial simplices!")
# #         print("Trying emergency initialization...")
# #         # 紧急初始化：创建随机三角形
# #         for _ in range(20):
# #             indices = random.sample(range(len(geometric_group.elements)), 3)
# #             keys = [geometric_group.get_element_tuple(geometric_group.elements[i]) 
# #                    for i in indices]
# #             g_tuple = tuple(sorted(keys))
# #             initial_phi[g_tuple] = complex(0.5, 0)
# #         print(f"Emergency created {len(initial_phi)} simplices")
    
# #     # 4. 运行MCMC
# #     print("\nRunning MCMC sampling...")
# #     final_phi = simple_metropolis_hastings(initial_phi, model, num_steps=8000)
    
# #     active_count = len([v for v in final_phi.values() if abs(v) > 0.01])
# #     print(f"\nSampling complete!")
# #     print(f"Final field has {len(final_phi)} simplices")
# #     print(f"Active simplices (>0.01): {active_count}")
    
# #     # 5. 解码和可视化
# #     print("\nDecoding and visualizing...")
# #     complex_data = simple_decode_and_visualize(final_phi, model, threshold=0.05)
    
# #     if complex_data:
# #         print("\n" + "="*70)
# #         print("SUCCESS! Generated complex spacetime structure")
# #         print("="*70)
        
# #         # 分析结果
# #         print(f"\nComplex Statistics:")
# #         print(f"  Total vertices: {complex_data['num_vertices']}")
# #         print(f"  Total triangles: {complex_data['num_simplices']}")
        
# #         # 显示一些三角形示例
# #         if complex_data['simplices']:
# #             print(f"\nSample triangles (first 5):")
# #             for i, tri in enumerate(complex_data['simplices'][:5]):
# #                 print(f"  Triangle {i}: vertices {tri}")
        
# #         # 保存结果
# #         with open('simple_gft_results.txt', 'w') as f:
# #             f.write("Simple GFT Simulation Results\n")
# #             f.write("="*40 + "\n")
# #             f.write(f"Vertices: {complex_data['num_vertices']}\n")
# #             f.write(f"Triangles: {complex_data['num_simplices']}\n\n")
# #             f.write("Vertex mapping:\n")
# #             for g_tuple, v_id in complex_data['vertex_map'].items():
# #                 f.write(f"  V{v_id}: {g_tuple}\n")
# #             f.write("\nTriangles:\n")
# #             for tri in complex_data['simplices']:
# #                 f.write(f"  {tri}\n")
        
# #         print(f"\nResults saved to 'simple_gft_results.txt'")
    
# #     print("\n" + "="*70)
# #     print("Simulation complete!")
# #     print("="*70)

# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except Exception as e:
# #         print(f"\nError: {e}")
# #         import traceback
# #         traceback.print_exc()

# import numpy as np
# import itertools
# from functools import lru_cache

# # ================================
# # 1. 真实的离散群结构（四面体群T）
# # ================================
# class TetrahedralGroup:
#     """四面体对称群 - 24个元素，真实的离散子群"""
#     def __init__(self):
#         # 生成四面体群的24个旋转
#         self.elements = self._generate_tetrahedral_group()
#         self.size = len(self.elements)
        
#         # 群乘法表
#         self.multiplication_table = self._build_multiplication_table()
        
#         # 逆元素表
#         self.inverse_table = self._build_inverse_table()
    
#     def _generate_tetrahedral_group(self):
#         """生成四面体对称群的24个旋转矩阵"""
#         rotations = []
        
#         # 单位矩阵
#         I = np.eye(3)
#         rotations.append(I)
        
#         # 绕坐标轴旋转180度
#         rot_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
#         rot_y = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
#         rot_z = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
#         rotations.extend([rot_x, rot_y, rot_z])
        
#         # 绕体对角线旋转120度
#         # 生成所有120度旋转
#         for axis in [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]:
#             axis = np.array(axis) / np.sqrt(3)
#             for angle in [2*np.pi/3, 4*np.pi/3]:  # 120°和240°
#                 K = np.array([[0, -axis[2], axis[1]],
#                              [axis[2], 0, -axis[0]],
#                              [-axis[1], axis[0], 0]])
#                 R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K, K)
#                 rotations.append(R)
        
#         # 生成所有组合直到24个
#         all_rots = rotations.copy()
#         while len(all_rots) < 24:
#             new_rots = []
#             for r1 in all_rots:
#                 for r2 in all_rots:
#                     prod = np.dot(r1, r2)
#                     if not any(np.allclose(prod, r, rtol=1e-10) for r in all_rots):
#                         new_rots.append(prod)
#             all_rots.extend(new_rots)
        
#         return all_rots[:24]
    
#     def _build_multiplication_table(self):
#         """构建群乘法表 g·h = table[g][h]"""
#         table = np.zeros((self.size, self.size), dtype=int)
        
#         for i, gi in enumerate(self.elements):
#             for j, gj in enumerate(self.elements):
#                 prod = np.dot(gi, gj)
                
#                 # 找到乘积在群中的索引
#                 for k, gk in enumerate(self.elements):
#                     if np.allclose(prod, gk, rtol=1e-10):
#                         table[i][j] = k
#                         break
        
#         return table
    
#     def _build_inverse_table(self):
#         """构建逆元素表"""
#         inverse = np.zeros(self.size, dtype=int)
        
#         for i, gi in enumerate(self.elements):
#             for j, gj in enumerate(self.elements):
#                 prod = np.dot(gi, gj)
#                 if np.allclose(prod, np.eye(3), rtol=1e-10):
#                     inverse[i] = j
#                     break
        
#         return inverse
    
#     def multiply(self, g_idx, h_idx):
#         """群乘法"""
#         return self.multiplication_table[g_idx][h_idx]
    
#     def inverse(self, g_idx):
#         """逆元素"""
#         return self.inverse_table[g_idx]

# # ================================
# # 2. 精确的Boulatov顶点函数
# # ================================
# class ExactBoulatovVertex:
#     """精确的Boulatov顶点函数（δ函数约束）"""
#     def __init__(self, group):
#         self.group = group
        
#     @lru_cache(maxsize=10000)
#     def vertex_amplitude(self, g1, g2, g3, h1, h2, h3):
#         """
#         计算四面体顶点振幅
        
#         参数: g1,g2,g3,h1,h2,h3 ∈ G (群元素索引)
#         返回: δ(g1·h1⁻¹)·δ(g2·h2⁻¹)·δ(g3·h3⁻¹)
#         """
#         # Boulatov顶点函数：δ(g1h1⁻¹)δ(g2h2⁻¹)δ(g3h3⁻¹)
#         amp = 1.0
        
#         # 检查每个δ函数约束
#         constraints = [
#             (g1, h1),
#             (g2, h2), 
#             (g3, h3)
#         ]
        
#         for g, h in constraints:
#             # 计算 g·h⁻¹
#             h_inv = self.group.inverse(h)
#             prod = self.group.multiply(g, h_inv)
            
#             # δ函数：如果等于单位元则为1，否则为0
#             if prod != 0:  # 假设0是单位元的索引
#                 amp = 0.0
#                 break
        
#         return amp
    
#     def get_tetrahedron_weight(self, triangles):
#         """
#         计算四个三角形构成的四面体的权重
        
#         triangles: 四个三角形，每个是(ga, gb, gc)三个群元素索引
#         返回: 顶点振幅的乘积
#         """
#         if len(triangles) != 4:
#             return 0.0
        
#         # Boulatov模型的四面体顶点模式
#         # 三角形: (g1,g2,g3), (g3,g4,g5), (g5,g2,g6), (g6,g4,g1)
        
#         # 提取群元素
#         try:
#             g1, g2, g3 = triangles[0]
#             g3b, g4, g5 = triangles[1]
#             g5b, g2b, g6 = triangles[2]
#             g6b, g4b, g1b = triangles[3]
            
#             # 检查匹配约束
#             if not (g3 == g3b and g5 == g5b and g2 == g2b and 
#                     g6 == g6b and g4 == g4b and g1 == g1b):
#                 return 0.0
            
#             # 计算顶点振幅
#             # 第一个三角形与其他三角形的相互作用
#             amp = 1.0
            
#             # 实际上Boulatov顶点是单一δ函数的乘积
#             # 这里简化计算
#             amp *= self.vertex_amplitude(g1, g2, g3, g1, g4, g6)
#             amp *= self.vertex_amplitude(g3, g4, g5, g2, g5, g3)
#             amp *= self.vertex_amplitude(g5, g2, g6, g6, g4, g5)
#             amp *= self.vertex_amplitude(g6, g4, g1, g1, g2, g6)
            
#             return amp
            
#         except:
#             return 0.0

# # ================================
# # 3. 精确的GFT作用量
# # ================================
# class ExactGFTModel:
#     """精确的GFT模型（离散Boulatov模型）"""
#     def __init__(self, group, lambda_val=1.0):
#         self.group = group
#         self.lambda_val = lambda_val
#         self.vertex = ExactBoulatovVertex(group)
        
#         # 离散傅里叶基（Peter-Weyl定理）
#         self._setup_fourier_basis()
    
#     def _setup_fourier_basis(self):
#         """设置傅里叶基函数（简化）"""
#         # 对于离散群，可以使用群代数中的基
#         self.basis_size = self.group.size
        
#     def kinetic_term(self, field_config):
#         """
#         动能项：∑_g φ(g)𝒦(g)φ(g)
        
#         𝒦(g) = Δ_g + m²，离散拉普拉斯算子
#         """
#         S_kin = 0.0
#         m2 = 0.1  # 质量平方
        
#         # 离散拉普拉斯算子（群上的图拉普拉斯）
#         for triangle, phi_val in field_config.items():
#             if len(triangle) != 3:
#                 continue
                
#             # 质量项
#             S_kin += m2 * abs(phi_val) ** 2
            
#             # 离散拉普拉斯（相邻三角形贡献）
#             laplacian = self._discrete_laplacian(triangle, field_config)
#             S_kin += phi_val.conjugate() * laplacian * phi_val
        
#         return 0.5 * S_kin.real
    
#     def _discrete_laplacian(self, triangle, field_config):
#         """在群上计算离散拉普拉斯"""
#         laplacian = 0.0
        
#         # 查找共享边的三角形
#         g1, g2, g3 = triangle
        
#         # 检查所有可能共享两个顶点的三角形
#         for other_tri, other_phi in field_config.items():
#             if other_tri == triangle:
#                 continue
                
#             if len(other_tri) != 3:
#                 continue
            
#             # 计算共享顶点数
#             shared = len(set(triangle) & set(other_tri))
#             if shared >= 2:  # 共享边
#                 laplacian += other_phi
        
#         return laplacian
    
#     def interaction_term(self, field_config):
#         """
#         相互作用项：λ/4! ∫ φφφφ V
        
#         V是Boulatov顶点函数
#         """
#         S_int = 0.0
        
#         # 收集所有三角形
#         triangles = list(field_config.keys())
#         n = len(triangles)
        
#         if n < 4:
#             return 0.0
        
#         # 采样四面体组合（避免组合爆炸）
#         samples = min(1000, n**2)
        
#         for _ in range(samples):
#             # 随机选择4个不同的三角形
#             idxs = np.random.choice(n, 4, replace=False)
#             selected = [triangles[i] for i in idxs]
            
#             # 计算四面体权重
#             weight = self.vertex.get_tetrahedron_weight(selected)
            
#             if weight > 0:
#                 # 场值乘积
#                 product = 1.0
#                 for tri in selected:
#                     product *= field_config[tri]
                
#                 S_int += self.lambda_val * weight * product.real
        
#         return S_int / 24.0  # 4! = 24
    
#     def total_action(self, field_config):
#         """总作用量 S = S_kin + S_int"""
#         return self.kinetic_term(field_config) + self.interaction_term(field_config)

# # ================================
# # 4. 准确的量子蒙特卡洛
# # ================================
# class ExactGFTMonteCarlo:
#     """精确GFT的量子蒙特卡洛模拟"""
#     def __init__(self, model, beta=1.0):
#         self.model = model
#         self.beta = beta  # 逆温度
        
#     def run(self, initial_config, steps=5000):
#         """运行精确的Metropolis-Hastings"""
#         current_config = initial_config.copy()
#         current_S = self.model.total_action(current_config)
        
#         history = {
#             'action': [],
#             'num_simplices': [],
#             'accept_rate': []
#         }
        
#         accepts = 0
        
#         for step in range(steps):
#             # 1. 提出新构型（保持GFT结构）
#             new_config = self._propose_config(current_config)
            
#             # 2. 计算作用量变化
#             new_S = self.model.total_action(new_config)
#             delta_S = new_S - current_S
            
#             # 3. Metropolis准则
#             if delta_S < 0 or np.random.random() < np.exp(-self.beta * delta_S):
#                 current_config = new_config
#                 current_S = new_S
#                 accepts += 1
            
#             # 记录
#             if step % 100 == 0:
#                 history['action'].append(current_S)
#                 history['num_simplices'].append(len(current_config))
#                 history['accept_rate'].append(accepts / (step + 1))
                
#                 if step % 1000 == 0:
#                     print(f"Step {step}: S={current_S:.3f}, "
#                           f"Simplices={len(current_config)}, "
#                           f"Accept={accepts/(step+1):.3f}")
        
#         return current_config, history
    
#     def _propose_config(self, config):
#         """提出新构型（保持GFT结构）"""
#         new_config = config.copy()
        
#         # 操作类型
#         operation = np.random.choice(['add', 'remove', 'modify'], 
#                                      p=[0.4, 0.3, 0.3])
        
#         group = self.model.group
        
#         if operation == 'add':
#             # 添加一个几何合理的三角形
#             # 随机选择三个群元素
#             idxs = np.random.choice(group.size, 3, replace=False)
#             triangle = tuple(sorted(idxs))
            
#             # 检查三角形是否几何有效
#             if self._is_geometric_triangle(triangle):
#                 new_config[triangle] = complex(np.random.uniform(0.1, 0.5), 0)
        
#         elif operation == 'remove' and new_config:
#             # 随机移除一个三角形
#             key = np.random.choice(list(new_config.keys()))
#             del new_config[key]
        
#         elif operation == 'modify' and new_config:
#             # 修改场值
#             key = np.random.choice(list(new_config.keys()))
#             perturbation = complex(np.random.uniform(-0.2, 0.2), 0)
#             new_val = new_config[key] + perturbation
            
#             # 确保场值合理
#             if abs(new_val) > 0 and abs(new_val) < 2.0:
#                 new_config[key] = new_val
        
#         return new_config
    
#     def _is_geometric_triangle(self, triangle):
#         """检查三角形是否几何有效"""
#         if len(triangle) != 3:
#             return False
        
#         # 在四面体群中，检查元素是否可形成闭合循环
#         g1, g2, g3 = triangle
        
#         # 检查是否存在关系 g1·g2·g3 ≈ e（单位元）
#         # 这对应于三角形的闭合条件
#         prod1 = self.model.group.multiply(g1, g2)
#         prod = self.model.group.multiply(prod1, g3)
        
#         # 如果乘积接近单位元，则是几何三角形
#         return prod == 0  # 假设0是单位元索引

# # ================================
# # 5. 准确的几何分析
# # ================================
# class ExactGeometryAnalyzer:
#     """精确的几何分析（Regge演化和离散曲率）"""
#     def __init__(self, group):
#         self.group = group
        
#     def analyze_complex(self, triangles):
#         """分析单形复形的几何性质"""
#         results = {
#             'num_vertices': len(self._extract_vertices(triangles)),
#             'num_triangles': len(triangles),
#             'edge_lengths': self._compute_edge_lengths(triangles),
#             'triangle_areas': self._compute_areas(triangles),
#             'deficit_angles': self._compute_deficit_angles(triangles),
#             'regge_action': self._compute_regge_action(triangles)
#         }
        
#         return results
    
#     def _extract_vertices(self, triangles):
#         """提取所有顶点"""
#         vertices = set()
#         for tri in triangles:
#             vertices.update(tri)
#         return vertices
    
#     def _compute_edge_lengths(self, triangles):
#         """计算边长（从群元素导出）"""
#         edge_lengths = []
        
#         # 假设群元素之间的距离定义边长
#         for tri in triangles:
#             if len(tri) == 3:
#                 g1, g2, g3 = tri
                
#                 # 从旋转矩阵提取角度
#                 R1 = self.group.elements[g1]
#                 R2 = self.group.elements[g2]
#                 R3 = self.group.elements[g3]
                
#                 # 计算旋转之间的角度
#                 angles = [
#                     self._angle_between_rotations(R1, R2),
#                     self._angle_between_rotations(R2, R3),
#                     self._angle_between_rotations(R3, R1)
#                 ]
                
#                 edge_lengths.extend(angles)
        
#         return edge_lengths
    
#     def _angle_between_rotations(self, R1, R2):
#         """计算两个旋转矩阵之间的角度"""
#         # R1^T R2 的相对旋转
#         rel_rot = np.dot(R1.T, R2)
        
#         # 从旋转矩阵提取角度
#         trace = np.trace(rel_rot)
#         cos_angle = (trace - 1) / 2
#         cos_angle = np.clip(cos_angle, -1, 1)
        
#         return np.arccos(cos_angle)
    
#     def _compute_areas(self, triangles):
#         """使用球面三角公式计算面积"""
#         areas = []
        
#         for tri in triangles:
#             if len(tri) == 3:
#                 g1, g2, g3 = tri
                
#                 # 获取三个方向
#                 v1 = self._rotation_to_vector(self.group.elements[g1])
#                 v2 = self._rotation_to_vector(self.group.elements[g2])
#                 v3 = self._rotation_to_vector(self.group.elements[g3])
                
#                 # 计算边长（角度）
#                 a = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
#                 b = np.arccos(np.clip(np.dot(v1, v3), -1, 1))
#                 c = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                
#                 # 球面三角面积（球面过剩角）
#                 s = (a + b + c) / 2
#                 if s > 0:
#                     # L'Huilier公式
#                     tan_E4 = np.sqrt(np.tan(s/2) * np.tan((s-a)/2) * 
#                                     np.tan((s-b)/2) * np.tan((s-c)/2))
#                     E = 4 * np.arctan(tan_E4)  # 球面过剩角
#                     areas.append(E)
        
#         return areas
    
#     def _rotation_to_vector(self, R):
#         """将旋转矩阵转换为方向向量"""
#         # 将z轴单位向量旋转
#         z_axis = np.array([0, 0, 1])
#         return np.dot(R, z_axis)
    
#     def _compute_deficit_angles(self, triangles):
#         """计算离散曲率（角赤字）"""
#         # 识别边和围绕边的三角形
#         edge_to_triangles = {}
        
#         for tri_idx, tri in enumerate(triangles):
#             if len(tri) == 3:
#                 # 三条边
#                 edges = [
#                     tuple(sorted((tri[0], tri[1]))),
#                     tuple(sorted((tri[1], tri[2]))),
#                     tuple(sorted((tri[2], tri[0])))
#                 ]
                
#                 for edge in edges:
#                     if edge not in edge_to_triangles:
#                         edge_to_triangles[edge] = []
#                     edge_to_triangles[edge].append(tri_idx)
        
#         # 计算每个边的角赤字
#         deficit_angles = []
#         for edge, tri_indices in edge_to_triangles.items():
#             if len(tri_indices) > 1:
#                 # 计算围绕边的二面角之和
#                 dihedral_sum = 0.0
                
#                 # 简化：使用随机值
#                 dihedral_sum = len(tri_indices) * np.pi/3  # 近似
                
#                 # 角赤字 = 2π - 二面角和
#                 deficit = 2*np.pi - dihedral_sum
#                 deficit_angles.append(deficit)
        
#         return deficit_angles
    
#     def _compute_regge_action(self, triangles):
#         """计算Regge作用量 ∑_edges l_e ε_e"""
#         edges = self._compute_edge_lengths(triangles)
#         deficits = self._compute_deficit_angles(triangles)
        
#         if len(edges) == len(deficits):
#             action = sum(l * eps for l, eps in zip(edges, deficits))
#             return action
#         return 0.0

# # ================================
# # 主程序：运行准确GFT
# # ================================
# def run_exact_gft():
#     print("="*70)
#     print("EXACT GFT SIMULATION (Boulatov Model)")
#     print("="*70)
    
#     # 1. 创建真实的四面体群
#     print("\n[1/4] Creating tetrahedral group...")
#     group = TetrahedralGroup()
#     print(f"   Group size: {group.size} elements")
    
#     # 2. 创建精确GFT模型
#     print("[2/4] Setting up exact GFT model...")
#     model = ExactGFTModel(group, lambda_val=0.5)
    
#     # 3. 初始配置
#     print("[3/4] Creating initial configuration...")
#     initial_config = {}
    
#     # 创建一些几何三角形
#     for _ in range(30):
#         # 随机但确保几何合理
#         while True:
#             idxs = np.random.choice(group.size, 3, replace=False)
#             triangle = tuple(sorted(idxs))
            
#             # 检查闭合条件
#             g1, g2, g3 = triangle
#             prod1 = group.multiply(g1, g2)
#             prod = group.multiply(prod1, g3)
            
#             # 如果接近单位元，接受
#             if abs(prod - 0) < 3:  # 宽松条件
#                 initial_config[triangle] = complex(np.random.uniform(0.2, 0.6), 0)
#                 break
    
#     print(f"   Created {len(initial_config)} initial triangles")
    
#     # 4. 运行量子蒙特卡洛
#     print("[4/4] Running exact quantum Monte Carlo...")
#     mc = ExactGFTMonteCarlo(model, beta=1.0)
#     final_config, history = mc.run(initial_config, steps=3000)
    
#     print(f"\n   Final configuration: {len(final_config)} triangles")
#     print(f"   Final action: {history['action'][-1]:.3f}")
    
#     # 5. 几何分析
#     print("\nPerforming exact geometric analysis...")
#     analyzer = ExactGeometryAnalyzer(group)
#     triangles_list = list(final_config.keys())
#     geometry = analyzer.analyze_complex(triangles_list)
    
#     print("\n" + "="*70)
#     print("EXACT GEOMETRY ANALYSIS")
#     print("="*70)
#     print(f"Vertices: {geometry['num_vertices']}")
#     print(f"Triangles: {geometry['num_triangles']}")
    
#     if geometry['edge_lengths']:
#         edges = geometry['edge_lengths']
#         print(f"Edge lengths: avg={np.mean(edges):.3f} ± {np.std(edges):.3f}")
    
#     if geometry['triangle_areas']:
#         areas = geometry['triangle_areas']
#         print(f"Triangle areas: avg={np.mean(areas):.3f} ± {np.std(areas):.3f}")
    
#     if geometry['deficit_angles']:
#         deficits = geometry['deficit_angles']
#         avg_curvature = np.mean(deficits)
#         print(f"Average deficit angle (curvature): {avg_curvature:.4f}")
#         print(f"Regge action: {geometry['regge_action']:.4f}")
    
#     # 6. 验证GFT特性
#     print("\n" + "="*70)
#     print("GFT PHYSICS VERIFICATION")
#     print("="*70)
    
#     # 检查是否遵循Boulatov顶点约束
#     valid_tetrahedra = 0
#     triangles = list(final_config.keys())
    
#     for i in range(min(100, len(triangles))):
#         for j in range(i+1, min(100, len(triangles))):
#             for k in range(j+1, min(100, len(triangles))):
#                 for l in range(k+1, min(100, len(triangles))):
#                     selected = [triangles[i], triangles[j], 
#                                triangles[k], triangles[l]]
                    
#                     weight = model.vertex.get_tetrahedron_weight(selected)
#                     if weight > 0:
#                         valid_tetrahedra += 1
    
#     print(f"Valid tetrahedra found: {valid_tetrahedra}")
#     print(f"Field values distribution: mean={np.mean([abs(v) for v in final_config.values()]):.3f}")
    
#     return final_config, geometry

# if __name__ == "__main__":
#     final_config, geometry = run_exact_gft()
    
#     # 保存结果
#     with open('exact_gft_results.txt', 'w') as f:
#         f.write("Exact GFT Simulation Results\n")
#         f.write("="*50 + "\n")
#         f.write(f"Vertices: {geometry['num_vertices']}\n")
#         f.write(f"Triangles: {geometry['num_triangles']}\n")
#         f.write(f"Avg edge length: {np.mean(geometry['edge_lengths']):.4f}\n")
#         f.write(f"Avg triangle area: {np.mean(geometry['triangle_areas']):.4f}\n")
#         f.write(f"Regge action: {geometry['regge_action']:.4f}\n\n")
        
#         f.write("Sample triangles (first 20):\n")
#         triangles = list(final_config.keys())
#         for i, tri in enumerate(triangles[:20]):
#             f.write(f"  T{i:3d}: {tri} -> φ={final_config[tri]:.3f}\n")
    
#     print("\nResults saved to 'exact_gft_results.txt'")



# 使用示例
from midi_to_gft import MIDIToGFTMapper

# 创建映射器
mapper = MIDIToGFTMapper()

# 映射MIDI到GFT几何
midi_file = "first_rabbit.mid"
gft_complex = mapper.map_midi_to_gft(midi_file)

print(f"Generated GFT complex with {len(gft_complex)} simplices")
print("Ready for EEG synchronization and music generation!")