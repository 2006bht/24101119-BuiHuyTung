"""
algorithms.py - Cài đặt trọn bộ 7 Thuật toán Học Tăng Cường Rời Rạc (Tabular RL)
Đề tài 29: Timed Gates / Traffic-Light Grid

Cấu trúc gồm 3 Nhóm thuật toán bám sát giáo trình GS. Shiyu Zhao:
---------------------------------------------------------------------------------
Nhóm 1: Model-Based / Quy hoạch động (Chương 4)
  1. Value Iteration (VI)
  2. Policy Iteration (PI)
  3. Truncated Policy Iteration (Truncated PI)

Nhóm 2: Monte Carlo / Model-Free Offline (Chương 5)
  4. MC Epsilon-Greedy

Nhóm 3: Temporal-Difference / Model-Free Online (Chương 7)
  5. SARSA (1-step SARSA)
  6. Q-Learning
  7. n-step SARSA
---------------------------------------------------------------------------------
"""

from typing import Dict, Tuple, Optional
import numpy as np
from env import TimedGateEnv, UP, DOWN, LEFT, RIGHT, WAIT


# ==============================================================================
# HÀM BỔ TRỢ (UTILITIES)
# ==============================================================================
def eps_greedy(Q: np.ndarray, s: int, epsilon: float, n_actions: int = 5) -> int:
    """
    Chọn hành động theo chính sách epsilon-greedy:
    - Với xác suất epsilon: Khám phá ngẫu nhiên (exploration)
    - Với xác suất 1 - epsilon: Khai thác hành động tốt nhất (exploitation)
    Phá vỡ thế hòa (tie-breaking) ngẫu nhiên nếu có nhiều hành động cùng Q lớn nhất.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_vals = Q[s]
    max_val = np.max(q_vals)
    best_actions = np.where(q_vals == max_val)[0]
    return int(np.random.choice(best_actions))


def extract_policy_from_v(env: TimedGateEnv, V: np.ndarray, gamma: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính bảng Q(s, a) và trích xuất greedy policy pi(s) = argmax_a Q(s, a) từ mảng V(s).
    """
    n_s, n_a = env.n_states, env.n_actions
    Q = np.zeros((n_s, n_a))
    policy = np.zeros(n_s, dtype=int)

    for s in range(n_s):
        for a in range(n_a):
            next_s, r, term = env.get_transition(s, a)
            Q[s, a] = r + gamma * V[next_s] * (not term)
        max_q = np.max(Q[s])
        # Ưu tiên chọn WAIT nếu WAIT đạt giá trị cực đại (thay vì đâm vào tường để đứng yên)
        if np.isclose(Q[s, WAIT], max_q):
            policy[s] = WAIT
        else:
            policy[s] = int(np.argmax(Q[s]))

    return Q, policy


# ==============================================================================
# NHÓM 1: MODEL-BASED / QUY HOẠCH ĐỘNG (DYNAMIC PROGRAMMING - CHƯƠNG 4)
# ==============================================================================
def value_iteration(env: TimedGateEnv, gamma: float = 0.9, theta: float = 1e-6) -> dict:
    """
    [THUẬT TOÁN]: Value Iteration (VI) - Chương 4
    [BẢN CHẤT]: Giải trực tiếp phương trình tối ưu Bellman Optimality bằng cách lặp giá trị V*(s).
    [CÔNG THỨC]: V_{k+1}(s) = max_a [ r(s,a) + gamma * V_k(s') ]
    """
    n_s, n_a = env.n_states, env.n_actions
    V = np.zeros(n_s)
    delta_history = []
    iterations = 0

    while True:
        delta = 0.0
        iterations += 1
        for s in range(n_s):
            v_old = V[s]
            # Quét qua tất cả action để tìm giá trị cực đại
            q_values = np.zeros(n_a)
            for a in range(n_a):
                next_s, r, term = env.get_transition(s, a)
                q_values[a] = r + gamma * V[next_s] * (not term)
            V[s] = np.max(q_values)
            delta = max(delta, abs(v_old - V[s]))

        delta_history.append(delta)
        if delta < theta:
            break

    Q, policy = extract_policy_from_v(env, V, gamma)
    return {"V": V, "Q": Q, "policy": policy, "iterations": iterations, "delta_history": delta_history}


def policy_iteration(env: TimedGateEnv, gamma: float = 0.9, theta: float = 1e-6) -> dict:
    """
    [THUẬT TOÁN]: Policy Iteration (PI) - Chương 4
    [BẢN CHẤT]: Xen kẽ 2 bước: Đánh giá chính sách (PE) đến hội tụ tuyệt đối -> Cải tiến chính sách (PI).
    [CÔNG THỨC]:
      - Policy Eval: V(s) = r(s, pi(s)) + gamma * V(s')
      - Policy Imprv: pi'(s) = argmax_a [ r(s, a) + gamma * V(s') ]
    """
    n_s, n_a = env.n_states, env.n_actions
    V = np.zeros(n_s)
    policy = np.zeros(n_s, dtype=int)  # Khởi tạo policy ban đầu
    outer_iters = 0
    total_eval_steps = 0

    while True:
        outer_iters += 1
        # Bước 1: Policy Evaluation (Đánh giá chính sách pi hiện tại đến khi hội tụ)
        while True:
            total_eval_steps += 1
            delta = 0.0
            for s in range(n_s):
                v_old = V[s]
                a = policy[s]
                next_s, r, term = env.get_transition(s, a)
                V[s] = r + gamma * V[next_s] * (not term)
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break

        # Bước 2: Policy Improvement (Cải tiến chính sách tham lam theo V)
        policy_stable = True
        for s in range(n_s):
            old_action = policy[s]
            q_values = np.zeros(n_a)
            for a in range(n_a):
                next_s, r, term = env.get_transition(s, a)
                q_values[a] = r + gamma * V[next_s] * (not term)
            max_q = np.max(q_values)
            best_action = WAIT if np.isclose(q_values[WAIT], max_q) else int(np.argmax(q_values))
            if best_action != old_action:
                policy_stable = False
            policy[s] = best_action

        if policy_stable:
            break

    Q, _ = extract_policy_from_v(env, V, gamma)
    return {"V": V, "Q": Q, "policy": policy, "iterations": outer_iters, "eval_steps": total_eval_steps}


def truncated_policy_iteration(env: TimedGateEnv, gamma: float = 0.9, max_k: int = 5, theta: float = 1e-6) -> dict:
    """
    [THUẬT TOÁN]: Truncated Policy Iteration - Chương 4
    [BẢN CHẤT]: Giống PI nhưng chỉ đánh giá chính sách (PE) đúng max_k bước rồi cải tiến ngay.
    [ĐIỂM ĐẶC BIỆT]:
      - Khi max_k = 1: Chính là Value Iteration!
      - Khi max_k = vô cùng: Chính là Policy Iteration chuẩn!
    """
    n_s, n_a = env.n_states, env.n_actions
    V = np.zeros(n_s)
    policy = np.zeros(n_s, dtype=int)
    outer_iters = 0

    while True:
        outer_iters += 1
        v_old_outer = V.copy()

        # Bước 1: Truncated Policy Evaluation (chỉ lặp tối đa max_k lần)
        for _ in range(max_k):
            for s in range(n_s):
                a = policy[s]
                next_s, r, term = env.get_transition(s, a)
                V[s] = r + gamma * V[next_s] * (not term)

        # Bước 2: Policy Improvement
        for s in range(n_s):
            q_values = np.zeros(n_a)
            for a in range(n_a):
                next_s, r, term = env.get_transition(s, a)
                q_values[a] = r + gamma * V[next_s] * (not term)
            max_q = np.max(q_values)
            policy[s] = WAIT if np.isclose(q_values[WAIT], max_q) else int(np.argmax(q_values))

        # Kiểm tra hội tụ của hàm giá trị
        if np.max(np.abs(V - v_old_outer)) < theta:
            break

    Q, _ = extract_policy_from_v(env, V, gamma)
    return {"V": V, "Q": Q, "policy": policy, "iterations": outer_iters}


# ==============================================================================
# NHÓM 2: MONTE CARLO / MODEL-FREE OFFLINE (CHƯƠNG 5)
# ==============================================================================
def mc_epsilon_greedy(env: TimedGateEnv, gamma: float = 0.9, epsilon: float = 0.1, alpha: float = 0.1, episodes: int = 1000) -> dict:
    """
    [THUẬT TOÁN]: MC Epsilon-Greedy (On-Policy MC Control) - Chương 5
    [BẢN CHẤT]: Xuất phát từ start_pos chuẩn, tự duy trì thăm dò bằng chính sách mềm epsilon-greedy.
    """
    n_s, n_a = env.n_states, env.n_actions
    Q = np.zeros((n_s, n_a))
    episode_returns = []

    for ep in range(episodes):
        s, _ = env.reset()
        traj_s, traj_a, traj_r = [], [], []
        done = False

        while not done:
            a = eps_greedy(Q, s, epsilon, n_a)
            next_s, r, term, trunc, _ = env.step(a)
            done = term or trunc

            traj_s.append(s)
            traj_a.append(a)
            traj_r.append(r)
            s = next_s

        disc_r = sum(r * (gamma ** idx) for idx, r in enumerate(traj_r))
        episode_returns.append(disc_r)

        # Duyệt ngược tính Return G
        G = 0.0
        visited = set()
        for t in reversed(range(len(traj_r))):
            st, at, rt = traj_s[t], traj_a[t], traj_r[t]
            G = rt + gamma * G
            if (st, at) not in visited:
                visited.add((st, at))
                Q[st, at] += alpha * (G - Q[st, at])

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return {"Q": Q, "V": V, "policy": policy, "history": episode_returns}


# ==============================================================================
# NHÓM 3: TEMPORAL-DIFFERENCE / MODEL-FREE ONLINE (CHƯƠNG 7)
# ==============================================================================
def sarsa(env: TimedGateEnv, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0.1, episodes: int = 500) -> dict:
    """
    [THUẬT TOÁN]: SARSA (1-step On-policy TD Control) - Chương 7
    [BẢN CHẤT]: Học theo chính sách thực tế (vừa đi vừa học). On-policy.
    [CÔNG THỨC]: Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
    """
    n_s, n_a = env.n_states, env.n_actions
    Q = np.zeros((n_s, n_a))
    episode_returns = []

    for ep in range(episodes):
        s, _ = env.reset()
        a = eps_greedy(Q, s, epsilon, n_a)
        disc_r = 0.0
        gamma_t = 1.0
        done = False

        while not done:
            next_s, r, term, trunc, _ = env.step(a)
            disc_r += gamma_t * r
            gamma_t *= gamma
            done = term or trunc
            next_a = eps_greedy(Q, next_s, epsilon, n_a)

            # TD Target của SARSA: Dùng Q(next_s, next_a) thực tế
            td_target = r + gamma * Q[next_s, next_a] * (not term)
            Q[s, a] += alpha * (td_target - Q[s, a])

            s, a = next_s, next_a

        episode_returns.append(disc_r)

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return {"Q": Q, "V": V, "policy": policy, "history": episode_returns}


def q_learning(env: TimedGateEnv, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0.1, episodes: int = 500) -> dict:
    """
    [THUẬT TOÁN]: Q-Learning (Off-policy TD Control) - Chương 7
    [BẢN CHẤT]: Học trực tiếp hàm giá trị tối ưu Q* mà không phụ thuộc vào hành động tiếp theo. Off-policy.
    [CÔNG THỨC]: Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    """
    n_s, n_a = env.n_states, env.n_actions
    Q = np.zeros((n_s, n_a))
    episode_returns = []

    for ep in range(episodes):
        s, _ = env.reset()
        disc_r = 0.0
        gamma_t = 1.0
        done = False

        while not done:
            a = eps_greedy(Q, s, epsilon, n_a)
            next_s, r, term, trunc, _ = env.step(a)
            disc_r += gamma_t * r
            gamma_t *= gamma
            done = term or trunc

            # TD Target của Q-Learning: Dùng max_a' Q(next_s, a') (Off-policy!)
            td_target = r + gamma * np.max(Q[next_s]) * (not term)
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = next_s

        episode_returns.append(disc_r)

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return {"Q": Q, "V": V, "policy": policy, "history": episode_returns}


def n_step_sarsa(env: TimedGateEnv, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0.1, n: int = 3, episodes: int = 500) -> dict:
    """
    [THUẬT TOÁN]: n-step SARSA - Chương 7
    [BẢN CHẤT]: Cầu nối giữa 1-step SARSA (n=1) và Monte Carlo (n=vô cùng).
    [CÔNG THỨC]: G_{t:t+n} = r_{t+1} + gamma * r_{t+2} + ... + gamma^n * Q(s_{t+n}, a_{t+n})
    """
    n_s, n_a = env.n_states, env.n_actions
    Q = np.zeros((n_s, n_a))
    episode_returns = []

    for ep in range(episodes):
        s, _ = env.reset()
        a = eps_greedy(Q, s, epsilon, n_a)

        states = [s]
        actions = [a]
        rewards = [0.0]  # R0 bù vị trí để index khớp với t

        T = float("inf")
        t = 0
        disc_r = 0.0

        while True:
            if t < T:
                next_s, r, term, trunc, _ = env.step(actions[t])
                disc_r += (gamma ** t) * r
                states.append(next_s)
                rewards.append(r)

                if term or trunc:
                    T = t + 1
                else:
                    next_a = eps_greedy(Q, next_s, epsilon, n_a)
                    actions.append(next_a)

            tau = t - n + 1  # Thời điểm cập nhật
            if tau >= 0:
                # Tính Return n-bước G
                G = 0.0
                upper_bound = tau + n if T == float("inf") else min(tau + n, int(T))
                for i in range(tau + 1, upper_bound + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * Q[states[tau + n], actions[tau + n]]

                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

        episode_returns.append(disc_r)

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return {"Q": Q, "V": V, "policy": policy, "history": episode_returns}


# Bảng tra cứu 7 thuật toán sẵn sàng gọi theo tên
ALL_ALGORITHMS = {
    "Value Iteration": value_iteration,
    "Policy Iteration": policy_iteration,
    "Truncated PI": truncated_policy_iteration,
    "MC Epsilon-Greedy": mc_epsilon_greedy,
    "SARSA": sarsa,
    "Q-Learning": q_learning,
    "3-step SARSA": n_step_sarsa,
}
