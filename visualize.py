"""
visualize.py - Module trực quan hóa đồ họa chuyên nghiệp cho TimedGateEnv
Sử dụng Matplotlib vẽ giao diện lưới màu sắc, trực quan, phục vụ báo cáo & thuyết trình.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from configs import CONFIGS
from env import TimedGateEnv, make_env, ACTION_NAMES, UP, DOWN, LEFT, RIGHT, WAIT


# Bảng màu chuẩn hiện đại (Flat UI Colors)
COLOR_EMPTY = "#F8F9FA"       # Trắng ngà cho ô đi được
COLOR_WALL = "#2C3E50"        # Xanh đen cho tường đá
COLOR_START = "#E8F8F5"       # Xanh ngọc nhạt cho điểm xuất phát
COLOR_GOAL = "#FEF9E7"        # Vàng nhạt cho Đích
COLOR_GATE_OPEN = "#D4EFDF"   # Xanh lá cây nhạt khi cổng MỞ
COLOR_GATE_CLOSED = "#FADBD8" # Đỏ nhạt khi cổng ĐÓNG
COLOR_AGENT = "#2980B9"       # Xanh dương đậm cho Agent
COLOR_GRID_LINE = "#BDC3C7"   # Viền ô


def draw_env(env: TimedGateEnv, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    """
    Vẽ trạng thái hiện tại của lưới GridWorld lên matplotlib Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    h, w = env.height, env.width
    r_agent, c_agent, phase, pass_avail = env.decode_state(env.state)

    ax.clear()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Đảo trục Y: row 0 ở trên cùng, row h ở dưới cùng
    ax.set_aspect("equal")

    # 1. Vẽ từng ô vuông trên lưới
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            cell_type = env.grid[r][c]

            # Xác định màu nền và nhãn của ô
            if cell_type == "#":
                bg_color = COLOR_WALL
                label = ""
                text_color = "white"
            elif pos == env.goal_pos:
                bg_color = COLOR_GOAL
                label = "★ GOAL"
                text_color = "#B7950B"
            elif pos in env.gates:
                gate = env.gates[pos]
                is_open = gate.is_open(phase)
                bg_color = COLOR_GATE_OPEN if is_open else COLOR_GATE_CLOSED
                status_text = "MỞ" if is_open else "ĐÓNG"
                label = f"{gate.name}\n({status_text})"
                text_color = "#1E8449" if is_open else "#C0392B"
            elif pos == env.start_pos:
                bg_color = COLOR_START
                label = "START"
                text_color = "#117864"
            else:
                bg_color = COLOR_EMPTY
                label = ""
                text_color = "#7F8C8D"

            # Vẽ hình chữ nhật ô
            rect = patches.Rectangle(
                (c, r), 1, 1,
                linewidth=1.5,
                edgecolor=COLOR_GRID_LINE,
                facecolor=bg_color
            )
            ax.add_patch(rect)

            # Vẽ chữ nhãn nếu không có agent che
            if label and pos != (r_agent, c_agent):
                ax.text(
                    c + 0.5, r + 0.5, label,
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                    ha="center", va="center"
                )

    # 2. Vẽ Agent lên trên cùng
    agent_circle = patches.Circle(
        (c_agent + 0.5, r_agent + 0.5),
        radius=0.35,
        facecolor=COLOR_AGENT,
        edgecolor="#1B4F72",
        linewidth=2,
        zorder=5
    )
    ax.add_patch(agent_circle)
    ax.text(
        c_agent + 0.5, r_agent + 0.5, "A",
        color="white",
        fontsize=12,
        fontweight="bold",
        ha="center", va="center",
        zorder=6
    )

    # 3. Định dạng trục và tiêu đề
    ax.set_xticks(np.arange(w) + 0.5)
    ax.set_yticks(np.arange(h) + 0.5)
    ax.set_xticklabels([f"C{c}" for c in range(w)], fontsize=8)
    ax.set_yticklabels([f"R{r}" for r in range(h)], fontsize=8)
    ax.tick_params(length=0)

    # Tiêu đề hiển thị thông số runtime
    if title is None:
        gate_info = ", ".join([
            f"{g.name}: {'MỞ' if g.is_open(phase) else 'ĐÓNG'}"
            for g in env.gates.values()
        ])
        title = f"{env.config.name}\nPhase: {phase}/{env.period} | Vé Pass: {pass_avail} | {gate_info}"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)


def plot_3_maps():
    """Vẽ cả 3 bản đồ cạnh nhau trong 1 hình duy nhất để so sánh trực quan khi thuyết trình."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    map_keys = ["map1", "map2", "map3"]
    
    for idx, key in enumerate(map_keys):
        env = make_env(key)
        env.reset()
        t = f"{env.config.name}\n|S| = {env.n_states} states | P = {env.period} | Pass = {env.initial_pass}"
        draw_env(env, ax=axes[idx], title=t)

    plt.suptitle("3 CẤU HÌNH BẢN ĐỒ TÍCH HỢP 4 SETTING BẮT BUỘC (ĐỀ TÀI 29)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    return fig


def plot_progression(steps_history: List[dict], main_title: str = "Diễn biến từng bước"):
    """
    Vẽ dải hình ảnh diễn biến từng bước liên tiếp theo hàng ngang.
    Mỗi step hiển thị: hình lưới + hành động + reward + ghi chú.
    """
    n = len(steps_history)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for idx, step_info in enumerate(steps_history):
        env = step_info["env"]
        title = (
            f"Bước {idx}: {step_info.get('action_name', 'Bắt đầu')}\n"
            f"Reward: {step_info.get('reward', 0.0)} | Phase: {step_info.get('phase', 0)}\n"
            f"({step_info.get('note', '')})"
        )
        draw_env(env, ax=axes[idx], title=title)

    plt.suptitle(main_title, fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    return fig


def plot_policy_grid(env: TimedGateEnv, V: np.ndarray, policy: np.ndarray, title: str = "Chính sách tối ưu π* theo từng Phase"):
    """
    Vẽ lưới hiển thị mũi tên chính sách (↑, ↓, ←, →, W) và giá trị V*(s) theo từng phase.
    Giúp thấy rõ: Khi cổng ĐÓNG -> ô trước cổng hiện 'W' (chờ) hoặc rẽ lên đường vòng.
                  Khi cổng MỞ   -> ô trước cổng hiện mũi tên đi xuyên qua.
    """
    ARROW_SYMBOLS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→", WAIT: "W"}
    h, w, p = env.height, env.width, env.period
    pass_avail = 0  # Xem xét lát cắt với pass = 0

    fig, axes = plt.subplots(1, p, figsize=(4.8 * p, 4.5))
    if p == 1:
        axes = [axes]

    for phase_idx in range(p):
        ax = axes[phase_idx]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")

        # Vẽ từng ô
        for r in range(h):
            for c in range(w):
                pos = (r, c)
                cell_type = env.grid[r][c]
                s_id = env.encode_state(r, c, phase_idx, pass_avail)

                if cell_type == "#":
                    bg = COLOR_WALL
                    txt = ""
                elif pos == env.goal_pos:
                    bg = COLOR_GOAL
                    txt = "★ GOAL\nV=0.0"
                elif pos in env.gates:
                    gate = env.gates[pos]
                    is_open = gate.is_open(phase_idx)
                    bg = COLOR_GATE_OPEN if is_open else COLOR_GATE_CLOSED
                    status = "MỞ" if is_open else "ĐÓNG"
                    act = policy[s_id]
                    v_val = V[s_id]
                    txt = f"{gate.name}:{status}\n{ARROW_SYMBOLS[act]}\nV={v_val:.1f}"
                else:
                    bg = COLOR_EMPTY
                    act = policy[s_id]
                    v_val = V[s_id]
                    act_sym = ARROW_SYMBOLS[act]
                    txt = f"{act_sym}\nV={v_val:.1f}"

                rect = patches.Rectangle((c, r), 1, 1, linewidth=1.2, edgecolor=COLOR_GRID_LINE, facecolor=bg)
                ax.add_patch(rect)

                if txt:
                    color = "#2C3E50" if bg != COLOR_WALL else "white"
                    ax.text(c + 0.5, r + 0.5, txt, fontsize=8, fontweight="bold", ha="center", va="center", color=color)

        ax.set_xticks(np.arange(w) + 0.5)
        ax.set_yticks(np.arange(h) + 0.5)
        ax.set_xticklabels([f"C{c}" for c in range(w)], fontsize=8)
        ax.set_yticklabels([f"R{r}" for r in range(h)], fontsize=8)
        
        gate_statuses = ", ".join([f"{g.name}: {'MỞ' if g.is_open(phase_idx) else 'ĐÓNG'}" for g in env.gates.values()])
        ax.set_title(f"Phase {phase_idx}/{p} ({gate_statuses})", fontsize=10, fontweight="bold", pad=8)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    return fig


def plot_learning_curves(curves_dict: dict, ground_truth_v: Optional[float] = None, window: int = 25, title: str = "So sánh đường học (Learning Curves)"):
    """
    Vẽ biểu đồ đường học Return theo từng Episode (có làm mượt moving average)
    kèm đường chuẩn Ground Truth từ Value Iteration.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, returns in curves_dict.items():
        if len(returns) == 0:
            continue
        # Moving average
        if len(returns) >= window:
            smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
            x = np.arange(window, len(returns) + 1)
            ax.plot(x, smoothed, label=f"{name} (MA-{window})", linewidth=2)
        else:
            ax.plot(returns, label=name, alpha=0.7)

    if ground_truth_v is not None:
        ax.axhline(y=ground_truth_v, color="red", linestyle="--", linewidth=2, label=f"V*(start) Ground Truth ({ground_truth_v:.2f})")

    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Discounted Return ($G_0$)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    return fig


def plot_multi_seed_comparison(
    multi_seed_results: dict,
    ground_truth_v: Optional[float] = None,
    window: int = 20,
    title: str = "So sánh thuật toán trên 5 Random Seeds (Độ lệch chuẩn ±σ)"
):
    """
    Vẽ biểu đồ học tập trên nhiều Random Seeds:
    - Đường nét liền đậm: Giá trị trung bình (Mean Return qua các seeds).
    - Dải bóng mờ (Shaded Band): Khoảng độ lệch chuẩn [Mean - Std, Mean + Std].
    - Đường nét đứt đỏ: Giá trị tối ưu Ground Truth từ Value Iteration.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"SARSA": "#2980B9", "Q-Learning": "#E67E22", "3-step SARSA": "#27AE60", "MC Epsilon-Greedy": "#8E44AD"}

    for name, seed_runs in multi_seed_results.items():
        # seed_runs shape: (n_seeds, n_episodes)
        arr = np.array(seed_runs, dtype=float)
        n_seeds, n_eps = arr.shape

        # Làm mượt moving average cho từng seed
        smoothed_runs = []
        for s in range(n_seeds):
            if n_eps >= window:
                sm = np.convolve(arr[s], np.ones(window) / window, mode="valid")
            else:
                sm = arr[s]
            smoothed_runs.append(sm)
        smoothed_runs = np.array(smoothed_runs)

        mean_curve = np.mean(smoothed_runs, axis=0)
        std_curve = np.std(smoothed_runs, axis=0)
        x = np.arange(len(mean_curve)) + (window if n_eps >= window else 1)

        color = colors.get(name, None)
        ax.plot(x, mean_curve, label=f"{name} (Trung bình {n_seeds} seeds)", linewidth=2.2, color=color)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=color)

    if ground_truth_v is not None:
        ax.axhline(y=ground_truth_v, color="red", linestyle="--", linewidth=2, label=f"V*(start) Ground Truth ({ground_truth_v:.2f})")

    ax.set_xlabel("Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Discounted Return trung bình ($G_0$)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    return fig
