import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import json
import os

def visualize_complete_triangulation(json_path: str):
    """完整可视化三角化结果"""
    
    print(f"Loading data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    points = np.array(data['points'])
    triangles = [[int(p) for p in triangle] for triangle in data['triangles']]
    features = data['features']
    
    print(f"\nData loaded:")
    print(f"  • Points: {len(points)}")
    print(f"  • Triangles: {len(triangles)}")
    
    # 分析三角形分布
    triangle_centers = []
    triangle_areas = []
    
    for tri in triangles:
        if len(tri) == 3:
            v1, v2, v3 = points[int(tri[0])], points[int(tri[1])], points[int(tri[2])]
            center = (v1 + v2 + v3) / 3
            triangle_centers.append(center)
            
            # 计算面积
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            triangle_areas.append(area)
    
    triangle_centers = np.array(triangle_centers)
    
    print(f"\nTriangle analysis:")
    print(f"  • Min area: {min(triangle_areas):.6f}")
    print(f"  • Max area: {max(triangle_areas):.6f}")
    print(f"  • Avg area: {np.mean(triangle_areas):.6f}")
    
    # 创建完整可视化
    fig = plt.figure(figsize=(25, 20))
    
    # 1. 3D完整视图（显示所有三角形）
    ax1 = fig.add_subplot(331, projection='3d')
    ax1.set_title("Complete 3D Triangulation (All Triangles)", 
                 fontsize=12, fontweight='bold', pad=20)
    
    # 绘制所有三角形
    for i, tri in enumerate(triangles[:2000]):  # 显示前2000个以避免过载
        if len(tri) == 3:
            triangle_points = points[list(tri) + [tri[0]]]  # 闭合
            # 根据面积着色
            area = triangle_areas[i]
            alpha = min(0.3, 0.1 + area * 10)  # 面积越大越明显
            
            ax1.plot(triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2],
                    color='blue', alpha=alpha, linewidth=0.5)
    
    # 绘制点
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='red', s=10, alpha=0.5, depthshade=True)
    
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Pitch', fontsize=10)
    ax1.set_zlabel('Velocity', fontsize=10)
    ax1.view_init(elev=25, azim=45)
    
    # 2. 三角形中心分布
    ax2 = fig.add_subplot(332)
    ax2.set_title("Triangle Centers Distribution", fontsize=12, fontweight='bold')
    
    if len(triangle_centers) > 0:
        # 2D直方图
        h = ax2.hist2d(triangle_centers[:, 0], triangle_centers[:, 1], 
                      bins=30, cmap='hot', alpha=0.8)
        plt.colorbar(h[3], ax=ax2, label='Density')
    
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Pitch', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 三角形面积分布
    ax3 = fig.add_subplot(333)
    ax3.set_title("Triangle Area Distribution", fontsize=12, fontweight='bold')
    
    ax3.hist(triangle_areas, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(triangle_areas), color='red', linestyle='--',
               label=f'Mean: {np.mean(triangle_areas):.6f}')
    ax3.set_xlabel('Triangle Area', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 连通性分析
    ax4 = fig.add_subplot(334)
    ax4.set_title("Point Connectivity", fontsize=12, fontweight='bold')
    
    # 计算每个点的三角形数量
    point_triangle_count = np.zeros(len(points))
    for tri in triangles:
        for vertex in tri:
            if vertex < len(point_triangle_count):
                point_triangle_count[vertex] += 1
    
    # 绘制点的大小基于连接数
    sizes = 10 + 50 * (point_triangle_count / max(1, point_triangle_count.max()))
    
    scatter = ax4.scatter(points[:, 0], points[:, 1], s=sizes, 
                         c=point_triangle_count, cmap='viridis',
                         alpha=0.7, edgecolors='k', linewidth=0.5)
    
    plt.colorbar(scatter, ax=ax4, label='Number of triangles')
    ax4.set_xlabel('Time', fontsize=10)
    ax4.set_ylabel('Pitch', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 分区域可视化
    ax5 = fig.add_subplot(335)
    ax5.set_title("Regional Connectivity", fontsize=12, fontweight='bold')
    
    # 将空间分成4个区域
    time_mid = (points[:, 0].min() + points[:, 0].max()) / 2
    pitch_mid = (points[:, 1].min() + points[:, 1].max()) / 2
    
    regions = {
        'Bottom-Left': (points[:, 0] < time_mid) & (points[:, 1] < pitch_mid),
        'Bottom-Right': (points[:, 0] >= time_mid) & (points[:, 1] < pitch_mid),
        'Top-Left': (points[:, 0] < time_mid) & (points[:, 1] >= pitch_mid),
        'Top-Right': (points[:, 0] >= time_mid) & (points[:, 1] >= pitch_mid)
    }
    
    colors = ['red', 'blue', 'green', 'purple']
    
    for (region_name, mask), color in zip(regions.items(), colors):
        region_points = points[mask]
        if len(region_points) > 0:
            ax5.scatter(region_points[:, 0], region_points[:, 1], 
                       color=color, s=20, alpha=0.7, label=region_name)
    
    ax5.set_xlabel('Time', fontsize=10)
    ax5.set_ylabel('Pitch', fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 三角形密度热图
    ax6 = fig.add_subplot(336)
    ax6.set_title("Triangle Density Heatmap", fontsize=12, fontweight='bold')
    
    # 创建密度网格
    grid_size = 20
    x_edges = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size)
    y_edges = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size)
    
    density = np.zeros((grid_size-1, grid_size-1))
    
    for center in triangle_centers:
        x_idx = np.searchsorted(x_edges, center[0]) - 1
        y_idx = np.searchsorted(y_edges, center[1]) - 1
        
        if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
            density[x_idx, y_idx] += 1
    
    im = ax6.imshow(density.T, origin='lower', aspect='auto',
                   extent=[points[:, 0].min(), points[:, 0].max(), 
                          points[:, 1].min(), points[:, 1].max()],
                   cmap='hot', interpolation='bilinear')
    
    plt.colorbar(im, ax=ax6, label='Triangle count')
    ax6.set_xlabel('Time', fontsize=10)
    ax6.set_ylabel('Pitch', fontsize=10)
    
    # 7. 按区域统计
    ax7 = fig.add_subplot(337)
    ax7.set_title("Triangles per Region", fontsize=12, fontweight='bold')
    
    region_triangle_counts = {}
    for region_name, mask in regions.items():
        # 计算该区域的三角形数量
        region_count = 0
        for tri in triangles:
            if len(tri) == 3:
                # 检查三角形的顶点是否都在该区域
                vertices_in_region = all(mask[vertex] for vertex in tri 
                                       if vertex < len(mask))
                if vertices_in_region:
                    region_count += 1
        region_triangle_counts[region_name] = region_count
    
    bars = ax7.bar(region_triangle_counts.keys(), region_triangle_counts.values(),
                  color=colors, edgecolor='black', alpha=0.7)
    
    ax7.set_xlabel('Region', fontsize=10)
    ax7.set_ylabel('Triangle Count', fontsize=10)
    ax7.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, region_triangle_counts.values()):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    # 8. 三角形大小 vs 位置
    ax8 = fig.add_subplot(338)
    ax8.set_title("Triangle Size vs Position", fontsize=12, fontweight='bold')
    
    if len(triangle_centers) > 0:
        scatter = ax8.scatter(triangle_centers[:, 0], triangle_centers[:, 1],
                            s=50*np.array(triangle_areas)/max(triangle_areas),
                            c=triangle_areas, cmap='coolwarm',
                            alpha=0.6, edgecolors='k', linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax8, label='Triangle Area')
        ax8.set_xlabel('Time (center)', fontsize=10)
        ax8.set_ylabel('Pitch (center)', fontsize=10)
        ax8.grid(True, alpha=0.3)
    
    # 9. 统计信息
    ax9 = fig.add_subplot(339)
    ax9.axis('off')
    
    # 计算详细统计
    avg_vertices_per_triangle = 3.0  # 所有三角形都有3个顶点
    isolated_points = np.sum(point_triangle_count == 0)
    highly_connected = np.sum(point_triangle_count > np.mean(point_triangle_count) * 2)
    
    info_text = f"""
    Complete Triangulation Analysis
    {'='*45}
    
    Basic Statistics:
    • Points: {len(points)}
    • Triangles: {len(triangles)}
    • Triangle/Point Ratio: {len(triangles)/max(1, len(points)):.2f}
    
    Geometry Quality:
    • Min Triangle Area: {min(triangle_areas):.6f}
    • Max Triangle Area: {max(triangle_areas):.6f}
    • Avg Triangle Area: {np.mean(triangle_areas):.6f}
    
    Connectivity:
    • Isolated Points: {isolated_points} ({isolated_points/len(points)*100:.1f}%)
    • Highly Connected Points: {highly_connected}
    • Avg Triangles per Point: {np.mean(point_triangle_count):.1f}
    
    Regional Distribution:
    """
    
    for region_name, count in region_triangle_counts.items():
        percentage = count / max(1, len(triangles)) * 100
        info_text += f"  • {region_name}: {count} triangles ({percentage:.1f}%)\n"
    
    ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(f"Complete Music Spacetime Triangulation Analysis", 
                fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    # 保存
    output_name = os.path.splitext(json_path)[0] + "_complete_analysis.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nComplete analysis saved to: {output_name}")
    
    # 返回分析结果
    return {
        'points': points,
        'triangles': triangles,
        'triangle_centers': triangle_centers,
        'triangle_areas': triangle_areas,
        'point_triangle_count': point_triangle_count,
        'region_triangle_counts': region_triangle_counts,
        'isolated_points': isolated_points,
        'avg_area': np.mean(triangle_areas)
    }


def create_enhanced_network_visualization(json_path: str):
    """增强的网络拓扑可视化（显示所有连接）"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    points = np.array(data['points'])
    triangles = data['triangles']
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 完整的网络图
    ax1 = fig.add_subplot(221)
    ax1.set_title("Complete Network Topology", fontsize=12, fontweight='bold')
    
    # 收集所有边
    edges = set()
    for tri in triangles:
        if len(tri) == 3:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add(tuple(sorted([tri[i], tri[j]])))
    
    print(f"Total unique edges: {len(edges)}")
    
    # 绘制所有边
    for edge in list(edges)[:1000]:  # 限制数量避免过载
        p1, p2 = points[edge[0]], points[edge[1]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'gray', alpha=0.1, linewidth=0.3)
    
    # 绘制点
    ax1.scatter(points[:, 0], points[:, 1], s=10, c='red', alpha=0.5)
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Pitch', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 按连接强度着色
    ax2 = fig.add_subplot(222)
    ax2.set_title("Connection Strength", fontsize=12, fontweight='bold')
    
    # 计算每个点的连接数
    connection_count = np.zeros(len(points))
    for edge in edges:
        connection_count[edge[0]] += 1
        connection_count[edge[1]] += 1
    
    scatter = ax2.scatter(points[:, 0], points[:, 1], 
                         s=10 + 30 * (connection_count / max(1, connection_count.max())),
                         c=connection_count, cmap='hot',
                         alpha=0.8, edgecolors='k', linewidth=0.5)
    
    plt.colorbar(scatter, ax=ax2, label='Connection degree')
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Pitch', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 连通组件
    ax3 = fig.add_subplot(223)
    ax3.set_title("Connected Components", fontsize=12, fontweight='bold')
    
    # 使用Union-Find找连通组件
    parent = list(range(len(points)))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # 合并所有边
    for edge in edges:
        union(edge[0], edge[1])
    
    # 找出所有组件
    components = {}
    for i in range(len(points)):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)
    
    # 为每个组件分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(components))))
    
    for idx, (root, comp_points) in enumerate(components.items()):
        if idx < 20:  # 最多显示20个组件
            comp_coords = points[comp_points]
            color = colors[idx % len(colors)]
            ax3.scatter(comp_coords[:, 0], comp_coords[:, 1],
                       color=color, s=20, alpha=0.7,
                       label=f'Component {idx+1} ({len(comp_points)} points)')
    
    ax3.set_xlabel('Time', fontsize=10)
    ax3.set_ylabel('Pitch', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 统计信息
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # 计算组件统计
    component_sizes = [len(comp) for comp in components.values()]
    largest_component = max(component_sizes) if component_sizes else 0
    
    info_text = f"""
    Network Analysis
    {'='*30}
    
    Basic Stats:
    • Points: {len(points)}
    • Triangles: {len(triangles)}
    • Unique Edges: {len(edges)}
    • Avg Degree: {np.mean(connection_count):.2f}
    
    Connectivity:
    • Connected Components: {len(components)}
    • Largest Component: {largest_component} points
    • Isolated Points: {np.sum(connection_count == 0)}
    
    Component Sizes:
    """
    
    # 按大小排序
    sorted_sizes = sorted(component_sizes, reverse=True)[:10]
    for i, size in enumerate(sorted_sizes[:5]):
        info_text += f"  • #{i+1}: {size} points\n"
    
    if len(sorted_sizes) > 5:
        info_text += f"  • ... and {len(sorted_sizes)-5} more\n"
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle("Enhanced Network Topology Visualization", 
                fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    output_name = os.path.splitext(json_path)[0] + "_enhanced_network.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Enhanced network visualization saved to: {output_name}")
    
    return {
        'edges': list(edges),
        'components': components,
        'connection_count': connection_count,
        'component_sizes': component_sizes
    }


# 使用示例
if __name__ == "__main__":
    # 假设您的JSON文件名为 'first_rabbit_music_spacetime.json'
    json_file = "first_rabbit_music_spacetime.json"
    
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