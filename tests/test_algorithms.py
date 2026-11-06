import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
test_algorithms.py - Kiểm thử trọn vẹn 7 thuật toán Tabular RL
"""

import io
import sys
import time
# Thiết lập stdout utf-8 để in tiếng Việt
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass
from env import make_env, ACTION_NAMES
from algorithms import ALL_ALGORITHMS


def test_all_algorithms():
    env = make_env("map1")
    print("==================================================")
    print(f"KIỂM THỬ 7 THUẬT TOÁN TRÊN MAP 1 (|S|={env.n_states})")
    print("==================================================\n")

    for name, algo_func in ALL_ALGORITHMS.items():
        t0 = time.time()
        # Chạy thuật toán
        if "MC" in name:
            res = algo_func(env, gamma=0.9, episodes=500)
        elif "SARSA" in name or "Q-Learning" in name:
            res = algo_func(env, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=500)
        else:
            res = algo_func(env, gamma=0.9)

        elapsed = time.time() - t0
        V = res["V"]
        policy = res["policy"]
        start_val = V[16]  # Vị trí start (2, 0) tại phase 0

        # Kiểm tra chạy 1 episode theo greedy policy vừa học
        s, _ = env.reset(options={"start_phase": 0})
        total_r = 0.0
        steps = 0
        done = False
        traj = []
        while not done and steps < 20:
            a = policy[s]
            traj.append(ACTION_NAMES[a])
            next_s, r, term, trunc, _ = env.step(a)
            total_r += r
            steps += 1
            done = term or trunc
            s = next_s

        print(f"[{name:20s}] Time: {elapsed*1000:6.1f}ms | V(start): {start_val:6.2f} | Test Return: {total_r:5.1f} | Traj: {' -> '.join(traj)}")


if __name__ == "__main__":
    test_all_algorithms()
