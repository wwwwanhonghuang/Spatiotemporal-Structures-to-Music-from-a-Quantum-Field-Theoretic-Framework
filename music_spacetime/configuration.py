from dataclasses import dataclass


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
