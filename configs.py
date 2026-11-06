"""
configs.py - Định nghĩa 3 Cấu hình Map tích hợp 4 Setting bắt buộc
Đề tài 29: Timed Gates / Traffic-Light Grid

4 Setting bắt buộc được tích hợp vào 3 Map:
- Setting 1: Cổng chu kỳ (chu kỳ 2: mở ở phase chẵn, đóng ở phase lẻ).
- Setting 2: Nhiều cổng có chu kỳ khác nhau (chu kỳ 2 và chu kỳ 3 -> P = LCM(2,3) = 6).
- Setting 3: Có đường vòng dài hơn không qua cổng (để so sánh Wait vs Đi vòng).
- Setting 4: Cấp 1 pass (pass_available = 1) để phá 1 cổng đang đóng.
"""

import math
from typing import Dict, List, Tuple


class Gate:
    """
    Đại diện cho một cổng có chu kỳ mở/đóng theo thời gian.
    
    Tham số:
    - pos: (row, col) tọa độ cổng trên lưới.
    - period: Chu kỳ cổng (vd: 2 hoặc 3).
    - open_phases: Tuple các phase mà cổng MỞ (mặc định (0,)).
      Ví dụ period=2, open_phases=(0,):
        - phase 0: MỞ (chẵn)
        - phase 1: ĐÓNG (lẻ)
      Ví dụ period=3, open_phases=(0,):
        - phase 0: MỞ (chia hết cho 3)
        - phase 1, 2: ĐÓNG
    """
    def __init__(self, pos: Tuple[int, int], period: int, open_phases: Tuple[int, ...] = (0,), name: str = "Gate"):
        self.pos = pos
        self.period = period
        self.open_phases = tuple(open_phases)
        self.name = name

    def is_open(self, phase: int) -> bool:
        """Kiểm tra cổng có đang MỞ tại phase hiện tại hay không."""
        return (phase % self.period) in self.open_phases

    def __repr__(self): # Thông tin trạng thái hiện tại của cổng trên bản đồ
        return f"{self.name}(pos={self.pos}, period={self.period}, open_phases={self.open_phases})"


class MapConfig:
    """
    Cấu hình toàn bộ một bản đồ GridWorld.
    
    Tính toán kích thước không gian trạng thái:
    |S| = Height * Width * P * n_passes
    trong đó:
    - P = LCM(các chu kỳ cổng)
    - n_passes = 2 (nếu allow_pass=True, pass in {0, 1}) hoặc 1 (nếu allow_pass=False, pass in {0})
    """
    def __init__(
        self,
        name: str, 
        grid_lines: List[str],
        start_pos: Tuple[int, int],
        goal_pos: Tuple[int, int], 
        gates: List[Gate], 
        allow_pass: bool = False,
        initial_pass: int = 0, # 
        description: str = "",
    ):
        self.name = name
        self.grid = [
            list(line.strip().replace(" ", "")) if isinstance(line, str) else list(line) 
            for line in grid_lines
        ]
        self.height = len(self.grid) 
        self.width = len(self.grid[0]) 
        self.start_pos = start_pos
        self.goal_pos = goal_pos 
        self.gates: Dict[Tuple[int, int], Gate] = {g.pos: g for g in gates} 
        # trạng thái vé pass  
        self.allow_pass = allow_pass
        self.initial_pass = initial_pass if allow_pass else 0
        self.n_passes = 2 if allow_pass else 1 
        self.description = description

        # Chu kỳ chung P = LCM(các chu kỳ cổng)
        periods = [g.period for g in gates]
        self.period = math.lcm(*periods) if periods else 1

        # Kích thước không gian trạng thái |S|
        self.n_states = self.height * self.width * self.period * self.n_passes #tổng số trường hợp có trong 1 map

    def print_layout(self):
        """In sơ đồ trực quan của bản đồ ra console."""
        print(f"=== {self.name} ===")
        print(f"Kích thước: {self.height}x{self.width} | P={self.period} | Pass={self.n_passes} | |S|={self.n_states} states")
        print(f"Mô tả: {self.description}")
        for r in range(self.height):
            row_str = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self.start_pos:
                    row_str.append(" S ")
                elif pos == self.goal_pos:
                    row_str.append(" G ")
                elif pos in self.gates:
                    row_str.append(f"G{self.gates[pos].period}")
                else:
                    cell = self.grid[r][c]
                    row_str.append(f" {cell} ")
            print("".join(row_str))
        print("=" * 30)



MAP_1_CONFIG = MapConfig(
    name="Map 1 (Dễ - 4x4)",
    grid_lines=[
        ". . . .",
        ". # # .",
        ". . . .",
        "# # # #",
    ],
    start_pos=(2, 0),
    goal_pos=(2, 3),
    gates=[
        Gate(pos=(2, 1), period=2, open_phases=(0,), name="G1"),
    ],
    allow_pass=False,
    initial_pass=0,
    description="Setting 1+3: 1 cổng chu kỳ 2 trên đường thẳng, có đường vòng dài hơn, không pass. |S|=32",
)



MAP_2_CONFIG = MapConfig(
    name="Map 2 (Trung bình - 5x5)",
    grid_lines=[
        ". . . . .",
        ". # # # .",
        ". . . . .",
        "# # # # #",
        "# # # # #",
    ],
    start_pos=(2, 0),
    goal_pos=(2, 4),
    gates=[
        Gate(pos=(2, 1), period=2, open_phases=(0,), name="G1"),
        Gate(pos=(2, 3), period=3, open_phases=(0,), name="G2"),
    ],
    allow_pass=False,
    initial_pass=0,
    description="Setting 2+3: 2 cổng chu kỳ 2 & 3 (P=6), có ô chờ giữa 2 cổng và đường vòng. |S|=150",
)



MAP_3_CONFIG = MapConfig(
    name="Map 3 (Khó - 6x6)",
    grid_lines=[
        ". . . . . .",
        ". # # # # .",
        ". . . . . .",
        "# # # # # #",
        "# # # # # #",
        "# # # # # #",
    ],
    start_pos=(2, 0),
    goal_pos=(2, 5),
    gates=[
        Gate(pos=(2, 1), period=2, open_phases=(0,), name="G1"),
        Gate(pos=(2, 3), period=3, open_phases=(0,), name="G2"),
        Gate(pos=(2, 4), period=2, open_phases=(0,), name="G3"),
    ],
    allow_pass=True,
    initial_pass=1,
    description="Setting 2+3+4: 3 cổng chu kỳ 2 & 3 (P=6), đường vòng, cấp 1 vé pass phá cổng. |S|=432",
)

CONFIGS = {
    "map1": MAP_1_CONFIG,
    "map2": MAP_2_CONFIG,
    "map3": MAP_3_CONFIG,
}
