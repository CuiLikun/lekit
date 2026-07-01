import math
import time
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ===================== 全局配置项 =====================
# 地图网格尺寸
MAP_WIDTH = 40  # 地图宽度（字符列数）
MAP_HEIGHT = 20  # 地图高度（字符行数）
# 机器人实际世界坐标范围
WORLD_X_MIN = -10.0
WORLD_X_MAX = 10.0
WORLD_Y_MIN = -10.0
WORLD_Y_MAX = 10.0
# 显示标记
ROBOT_MARKER = "🤖"  # 机器人标记，终端不支持emoji可换成 "R"
# 历史轨迹配置
TRAJ_MAX_LEN = 30  # 轨迹最大点数（越大拖尾越长）
TRAJ_BASE_CHAR = "•"  # 轨迹点字符，可选 · • ● *
SHOW_TRAJECTORY = True  # 总开关：是否显示轨迹
# 轨迹颜色渐变：最旧点 → 最新点（暗灰 → 亮青）
TRAJ_COLOR_START = (60, 60, 80)
TRAJ_COLOR_END = (0, 210, 255)
# =====================================================


# ---------------------- 模拟机器人类（测试用，实际替换为真实robot） ----------------------
class MockChassisInfo:
    def __init__(self, x, y, z):
        self.position = type("Position", (), {"x": x, "y": y, "z": z})()


class MockRobot:
    def __init__(self):
        self.start_time = time.time()

    def get_chassis_info(self):
        t = time.time() - self.start_time
        x = math.sin(t * 0.8) * 7 + math.sin(t * 0.3) * 2
        y = math.cos(t * 0.8) * 5 + math.cos(t * 0.4) * 2
        z = 0.2 + math.sin(t) * 0.1
        return MockChassisInfo(x, y, z)


robot = MockRobot()


# ---------------------- 坐标映射核心 ----------------------
def world_to_grid(x: float, y: float) -> tuple[int, int]:
    """世界坐标 → 网格(行,列)，Y轴做上下翻转适配终端坐标系"""
    col = int((x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) * MAP_WIDTH)
    row = int((WORLD_Y_MAX - y) / (WORLD_Y_MAX - WORLD_Y_MIN) * MAP_HEIGHT)
    col = max(0, min(MAP_WIDTH - 1, col))
    row = max(0, min(MAP_HEIGHT - 1, row))
    return row, col


# ---------------------- 带轨迹的地图面板 ----------------------
def build_map_panel(x: float, y: float, z: float, trajectory: deque) -> Panel:
    # 计算机器人当前网格位置
    robot_row, robot_col = world_to_grid(x, y)

    # 1. 预计算轨迹点映射表：key=(行,列)，value=(字符, 样式)
    traj_map = {}
    if SHOW_TRAJECTORY and len(trajectory) > 0:
        total_points = len(trajectory)
        for idx, (tx, ty) in enumerate(trajectory):
            r, c = world_to_grid(tx, ty)
            # 按索引计算颜色渐变比例（0=最旧，1=最新）
            ratio = idx / max(total_points - 1, 1)
            r_col = int(TRAJ_COLOR_START[0] + (TRAJ_COLOR_END[0] - TRAJ_COLOR_START[0]) * ratio)
            g_col = int(TRAJ_COLOR_START[1] + (TRAJ_COLOR_END[1] - TRAJ_COLOR_START[1]) * ratio)
            b_col = int(TRAJ_COLOR_START[2] + (TRAJ_COLOR_END[2] - TRAJ_COLOR_START[2]) * ratio)
            style = f"rgb({r_col},{g_col},{b_col})"
            traj_map[(r, c)] = (TRAJ_BASE_CHAR, style)

    # 2. 逐格渲染地图：空白 → 轨迹 → 机器人（优先级从低到高）
    map_text = Text()
    for row_idx in range(MAP_HEIGHT):
        for col_idx in range(MAP_WIDTH):
            char = "."
            style = "dim white"
            # 叠加轨迹点
            if (row_idx, col_idx) in traj_map:
                char, style = traj_map[(row_idx, col_idx)]
            # 叠加机器人当前位置（优先级最高，覆盖轨迹）
            if row_idx == robot_row and col_idx == robot_col:
                char = ROBOT_MARKER
                style = "bold bright_cyan"
            map_text.append(char, style=style)
        map_text.append("\n")

    # 3. 底部追加坐标信息
    map_text.append(f"\n📍 X: {x:+.2f} | Y: {y:+.2f} | Z: {z:+.2f} | 轨迹点数: {len(trajectory)}", style="bold cyan")

    return Panel(map_text, title="Robot Real-time Map", border_style="blue", padding=(1, 2), title_align="left")


# ---------------------- 整体布局 ----------------------
def build_full_layout(chassis_info, trajectory: deque) -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="status_panel", ratio=1),
        Layout(name="map_panel", ratio=2),
    )

    # 左侧状态面板
    status_text = Text()
    status_text.append("🤖 Chassis Status\n\n", style="bold yellow")
    status_text.append(f"Position X: {chassis_info.position.x:.2f}\n")
    status_text.append(f"Position Y: {chassis_info.position.y:.2f}\n")
    status_text.append(f"Position Z: {chassis_info.position.z:.2f}\n")
    status_text.append(f"\nTrajectory Points: {len(trajectory)}/{TRAJ_MAX_LEN}", style="dim")

    layout["status_panel"].update(Panel(status_text, title="Status", border_style="green", padding=1))

    # 右侧地图面板（含轨迹）
    x, y, z = chassis_info.position.x, chassis_info.position.y, chassis_info.position.z
    layout["map_panel"].update(build_map_panel(x, y, z, trajectory))

    return layout


# ---------------------- 主循环 ----------------------
def main():
    console = Console()
    # 初始化轨迹队列（固定长度，自动淘汰最旧的点）
    trajectory = deque(maxlen=TRAJ_MAX_LEN)

    with Live(console=console, refresh_per_second=10, screen=True, vertical_overflow="ellipsis") as live:
        while True:
            # 1. 获取最新底盘信息
            chassis_info = robot.get_chassis_info()
            x, y = chassis_info.position.x, chassis_info.position.y

            # 2. 更新轨迹队列
            trajectory.append((x, y))

            # 3. 渲染并刷新
            full_layout = build_full_layout(chassis_info, trajectory)
            live.update(full_layout)

            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止实时地图渲染")
