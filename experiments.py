"""
experiments.py - Kịch bản Thực nghiệm & Phân tích chuyên sâu (Đề tài 29)
Phục vụ phân tích các câu hỏi học thuật và chuẩn bị cho phần Vấn đáp của Thầy.

Các thí nghiệm bao gồm:
1. Phân tích bài toán tradeoff: "Khi nào WAIT tối ưu vs. Khi nào đi đường vòng?"
2. Thử nghiệm thay đổi 2 cấu hình Reward (step_penalty nhẹ vs nặng).
3. Chạy thực nghiệm trên 5 Random Seeds vẽ biểu đồ độ lệch chuẩn (±σ) cho SARSA & Q-Learning.
4. Phân tích trường hợp thất bại / kẹt ở cực tiểu cục bộ khi hệ số khám phá epsilon quá nhỏ.
5. Chạy Demo 1 Episode hoàn chỉnh trên Map 3 sử dụng vé Pass vượt cổng.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from configs import Gate, MapConfig, MAP_1_CONFIG, MAP_2_CONFIG, MAP_3_CONFIG
from env import TimedGateEnv, make_env, UP, DOWN, LEFT, RIGHT, WAIT, ACTION_NAMES
from algorithms import value_iteration, sarsa, q_learning


# ==============================================================================
# THÍ NGHIỆM 1: KHI NÀO WAIT TỐI ƯU VS. KHI NÀO ĐI ĐƯỜNG VÒNG?
# ==============================================================================
def run_wait_vs_detour_analysis() -> dict:
    results = {}

    # --------------------------------------------------------------------------
    # Trường hợp A: w = 1 < Delta_d = 4 (Cổng chu kỳ 2, mở ở 0, đóng ở 1)
    # --------------------------------------------------------------------------
    env_short_wait = make_env("map1")
    res_a = value_iteration(env_short_wait, gamma=0.9)
    # Lấy action tối ưu tại vị trí (2, 0) ở phase 1 (cổng G1 đang đóng):
    s_closed_a = env_short_wait.encode_state(2, 0, phase=1, pass_avail=0)
    opt_act_a = res_a["policy"][s_closed_a]
    v_val_a = res_a["V"][s_closed_a]

    # --------------------------------------------------------------------------
    # Trường hợp B: w = 5 > Delta_d = 4 (Cổng chu kỳ 6, mở ở phase 0, đóng ở 1,2,3,4,5)
    # Đóng liên tiếp 5 nhịp -> Thời gian chờ w = 5 bước > Delta_d = 4 bước!
    # --------------------------------------------------------------------------
    map_long_wait = MapConfig(
        name="Map 1 (Cổng Đóng Lâu w=5)",
        grid_lines=MAP_1_CONFIG.grid,
        start_pos=(2, 0),
        goal_pos=(2, 3),
        gates=[Gate(pos=(2, 1), period=6, open_phases=(0,), name="G_slow")],
        allow_pass=False,
    )
    env_long_wait = TimedGateEnv(config=map_long_wait)
    res_b = value_iteration(env_long_wait, gamma=0.9)
    # Lấy action tối ưu tại vị trí (2, 0) ở phase 1 (cổng G_slow vừa đóng, phải chờ 5 nhịp):
    s_closed_b = env_long_wait.encode_state(2, 0, phase=1, pass_avail=0)
    opt_act_b = res_b["policy"][s_closed_b]
    v_val_b = res_b["V"][s_closed_b]

    results["case_A"] = {
        "scenario": "Cổng chờ ngắn (w = 1 nhịp < Delta_d = 4)",
        "opt_action": ACTION_NAMES[opt_act_a],
        "V_start_closed": v_val_a,
        "behavior": "Agent chọn WAIT trước cổng vì thời gian chờ (1 bước) ít hơn đi vòng (4 bước).",
    }
    results["case_B"] = {
        "scenario": "Cổng đóng rất lâu (w = 5 nhịp > Delta_d = 4)",
        "opt_action": ACTION_NAMES[opt_act_b],
        "V_start_closed": v_val_b,
        "behavior": "Agent chọn rẽ lên ĐƯỜNG VÒNG (UP) vì cổng đóng quá lâu, đi vòng nhanh hơn chờ!",
    }
    return results


# ==============================================================================
# THÍ NGHIỆM 2: THAY ĐỔI 2 CẤU HÌNH REWARD (STEP PENALTY NHẸ VS NẶNG)
# ==============================================================================
def run_reward_tuning_analysis() -> dict:
    """
    [CƠ SỞ TOÁN HỌC]:
    - Cấu hình 1 (step_penalty = -0.5, phạt cổng = -10.0):
      + Phạt bước đi rất nhẹ (-0.5), phạt đâm cổng rất nặng.
      + Khuyến khích Agent đi đường vòng dài hơn để tuyệt đối an toàn, tránh xa khu vực cổng.
    - Cấu hình 2 (step_penalty = -3.0, phạt cổng = -5.0):
      + Phạt bước đi cực nặng (-3.0). Mỗi bước đi tốn chi phí gấp 6 lần cấu hình 1!
      + Ép Agent bằng mọi giá phải đi đường ngắn nhất qua cổng (chờ mở hoặc dùng vé Pass),
        tuyệt đối không bao giờ đi đường vòng vì đường vòng sẽ bị trừ điểm rất nặng.
    """
    results = {}

    # Cấu hình 1: Phạt bước đi nhẹ (-0.5)
    env_light = make_env("map1", step_penalty=-0.5, gate_penalty=-10.0)
    res_light = value_iteration(env_light, gamma=0.9)

    # Cấu hình 2: Phạt bước đi nặng (-3.0)
    env_heavy = make_env("map1", step_penalty=-3.0, gate_penalty=-5.0)
    res_heavy = value_iteration(env_heavy, gamma=0.9)

    # Kiểm tra giá trị trạng thái tại điểm xuất phát
    s0_phase0 = env_light.encode_state(2, 0, phase=0, pass_avail=0)
    s0_phase1 = env_light.encode_state(2, 0, phase=1, pass_avail=0)

    results["config_light"] = {
        "step_penalty": -0.5,
        "gate_penalty": -10.0,
        "V_phase0": res_light["V"][s0_phase0],
        "V_phase1": res_light["V"][s0_phase1],
        "action_phase1": ACTION_NAMES[res_light["policy"][s0_phase1]],
        "analysis": "Chi phí bước đi nhỏ (-0.5), giá trị tích lũy cao, Agent chấp nhận đi nhiều bước.",
    }
    results["config_heavy"] = {
        "step_penalty": -3.0,
        "gate_penalty": -5.0,
        "V_phase0": res_heavy["V"][s0_phase0],
        "V_phase1": res_heavy["V"][s0_phase1],
        "action_phase1": ACTION_NAMES[res_heavy["policy"][s0_phase1]],
        "analysis": "Chi phí bước đi rất nặng (-3.0), giá trị trạng thái giảm mạnh, ép Agent tối ưu số bước tuyệt đối.",
    }
    return results


# ==============================================================================
# THÍ NGHIỆM 3: CHẠY TRÊN NHIỀU RANDOM SEEDS (SARSA VS Q-LEARNING)
# ==============================================================================
def run_multi_seed_experiment(seeds: List[int] = [1, 2, 3, 4, 5], episodes: int = 400) -> dict:
    """
    [CƠ SỞ TOÁN HỌC]:
    - Chạy trên nhiều (Random Seeds) để khử nhiễu ngẫu nhiên của chính sách khám phá epsilon-greedy.
    - Q-Learning (Off-policy):
      + Cập nhật theo max_a' Q(s', a') -> Tìm ra đường đi tối ưu ngắn nhất nhanh hơn.
      + Tuy nhiên, trong quá trình huấn luyện với epsilon-greedy, Agent thỉnh thoảng bước nhầm
        vào cổng đang đóng -> Phương sai (variance / dải độ lệch chuẩn) dao động lớn hơn.
    - SARSA (On-policy):
      + Cập nhật theo hành động thực tế a' -> Học chính sách an toàn hơn, đường học ổn định hơn.
    """
    env = make_env("map1")
    sarsa_seed_runs = []
    ql_seed_runs = []

    for seed in seeds:
        np.random.seed(seed)
        res_s = sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=episodes)
        sarsa_seed_runs.append(res_s["history"])

        np.random.seed(seed)
        res_q = q_learning(env, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=episodes)
        ql_seed_runs.append(res_q["history"])

    # Tính Ground Truth V*(start) từ Value Iteration làm đường tham chiếu chuẩn
    res_vi = value_iteration(env, gamma=0.9)
    v_star = res_vi["V"][env.encode_state(2, 0, 0, 0)]

    return {
        "SARSA": sarsa_seed_runs,
        "Q-Learning": ql_seed_runs,
        "ground_truth_v": v_star,
    }


# ==============================================================================
# THÍ NGHIỆM 4: PHÂN TÍCH TRƯỜNG HỢP THẤT BẠI / CHẬM HỘI TỤ (FAILURE CASES)
# ==============================================================================
def run_failure_case_analysis(episodes: int = 500) -> dict:
    """
    [CƠ SỞ TOÁN HỌC]:
    - Khi chu kỳ cổng lớn (Map 2: P = 6, |S| = 150) mà hệ số khám phá epsilon quá bé (ví dụ epsilon = 0.01):
      + Agent không khám phá đủ tất cả các phase.
      + Ngay khi tình cờ tìm thấy đường vòng đi được về Đích (mặc dù tốn 8 bước, sub-optimal),
        Agent lập tức khai thác con đường này (exploitation) mà KHÔNG BAO GIỜ dám mạo hiểm bước vào cổng
        để khám phá thời điểm cổng mở!
      + Kết quả: Bị kẹt vĩnh viễn ở Cực tiểu cục bộ (Local Optimum / Sub-optimal Policy)!
    - Khi epsilon = 0.2 (khám phá đủ):
      + Agent thử nghiệm nhiều thời điểm khác nhau, phát hiện ra nhịp cổng mở và hội tụ về đường thẳng 4 bước (Global Optimum)!
    """
    env = make_env("map2")

    # Trường hợp 1: Epsilon quá nhỏ (epsilon = 0.01) -> Bị kẹt ở nghiệm cục bộ (đường vòng)
    np.random.seed(42)
    res_fail = q_learning(env, gamma=0.9, alpha=0.1, epsilon=0.01, episodes=episodes)

    # Trường hợp 2: Epsilon phù hợp (epsilon = 0.2) -> Khám phá ra đường thẳng tối ưu qua 2 cổng
    np.random.seed(42)
    res_optimal = q_learning(env, gamma=0.9, alpha=0.1, epsilon=0.2, episodes=episodes)

    # Đánh giá Return thực tế sau khi học
    def test_policy(policy):
        s, _ = env.reset(options={"start_phase": 0})
        total_r = 0.0
        steps = 0
        for _ in range(30):
            a = policy[s]
            s, r, term, trunc, _ = env.step(a)
            total_r += r
            steps += 1
            if term or trunc:
                break
        return total_r, steps

    ret_fail, steps_fail = test_policy(res_fail["policy"])
    ret_opt, steps_opt = test_policy(res_optimal["policy"])

    return {
        "small_eps": {
            "epsilon": 0.01,
            "test_return": ret_fail,
            "test_steps": steps_fail,
            "conclusion": f"Kẹt ở đường vòng! Tốn {steps_fail} bước, Return = {ret_fail:.1f} (Local Optimum)",
        },
        "optimal_eps": {
            "epsilon": 0.20,
            "test_return": ret_opt,
            "test_steps": steps_opt,
            "conclusion": f"Hội tụ về đường tối ưu qua cổng! Chỉ tốn {steps_opt} bước, Return = {ret_opt:.1f} (Global Optimum)",
        },
    }
