import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
test_env.py - Kiểm thử toàn diện Môi trường TimedGateEnv (Phần 1)

Kiểm tra:
1. Kích thước không gian trạng thái |S| của 3 Map (32, 150, 432).
2. Tính toàn vẹn 2 chiều của hàm encode_state và decode_state.
3. Cơ chế đóng/mở cổng và hành động WAIT.
4. Cơ chế phạt cổng (-5) khi đâm vào cổng đóng không có vé.
5. Cơ chế tiêu thụ vé Pass (1 -> 0) trên Map 3 để xuyên qua cổng đóng.
6. Tuân thủ chuẩn Gymnasium (gymnasium.utils.env_checker.check_env).
"""

import io
import sys
# Thiết lập stdout utf-8 để hỗ trợ in tiếng Việt trên console Windows (nếu có buffer)
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from configs import CONFIGS, MAP_1_CONFIG, MAP_2_CONFIG, MAP_3_CONFIG
from env import TimedGateEnv, make_env, UP, DOWN, LEFT, RIGHT, WAIT


def test_map_sizes():
    print("--- [TEST 1]: Kiểm tra kích thước không gian trạng thái |S| ---")
    expected = {
        "map1": 32,   # 4 x 4 x 2 x 1 = 32
        "map2": 150,  # 5 x 5 x 6 x 1 = 150
        "map3": 432,  # 6 x 6 x 6 x 2 = 432
    }
    for map_name, exp_size in expected.items():
        env = make_env(map_name)
        assert env.n_states == exp_size, f"Lỗi {map_name}: n_states={env.n_states} != {exp_size}"
        assert env.observation_space.n == exp_size, f"Lỗi {map_name}: obs_space={env.observation_space.n} != {exp_size}"
        assert env.action_space.n == 5, f"Lỗi {map_name}: n_actions={env.action_space.n} != 5"
        print(f"  [OK] {map_name.upper()}: |S| = {env.n_states} states, |A| = {env.n_actions} actions")
    print("=> Test 1 PASSED!\n")


def test_encode_decode_roundtrip():
    print("--- [TEST 2]: Kiểm tra tính toàn vẹn 2 chiều encode_state <-> decode_state ---")
    for map_name in ["map1", "map2", "map3"]:
        env = make_env(map_name)
        for s in range(env.n_states):
            r, c, phase, pass_avail = env.decode_state(s)
            s_reconstructed = env.encode_state(r, c, phase, pass_avail)
            assert s == s_reconstructed, f"Lỗi khôi phục {map_name} tại s={s}: ({r},{c},{phase},{pass_avail}) -> {s_reconstructed}"
        print(f"  [OK] {map_name.upper()}: Đã kiểm tra 100% {env.n_states}/{env.n_states} states khớp hoàn hảo 2 chiều.")
    print("=> Test 2 PASSED!\n")


def test_gate_and_wait_mechanics():
    print("--- [TEST 3]: Kiểm tra cơ chế đóng/mở cổng và hành động WAIT (Map 1) ---")
    env = make_env("map1", step_penalty=-1.0, gate_penalty=-5.0, goal_reward=10.0)

    # 1. Start tại (2, 0) với phase = 0 (cổng G1 tại (2,1) MỞ vì 0 % 2 == 0)
    s, _ = env.reset(options={"start_phase": 0})
    r, c, phase, pass_avail = env.decode_state(s)
    assert (r, c) == (2, 0) and phase == 0, f"Trạng thái xuất phát sai: ({r}, {c}), phase={phase}"

    # Agent đi RIGHT vào ô (2, 1) có cổng G1 đang MỞ:
    next_s, rwd, term, trunc, _ = env.step(RIGHT)
    nr, nc, n_phase, _ = env.decode_state(next_s)
    assert (nr, nc) == (2, 1), f"Đi vào cổng mở thất bại: tới ({nr}, {nc})"
    assert rwd == -1.0, f"Reward bước đi thường phải là -1.0, nhận được: {rwd}"
    assert n_phase == 1, f"Phase tiếp theo phải là 1, nhận được: {n_phase}"
    print("  [OK] Đi qua cổng MỞ tại phase 0 thành công!")

    # 2. Reset về (2, 0) nhưng với phase = 1 (cổng G1 ĐÓNG vì 1 % 2 == 1)
    s, _ = env.reset(options={"start_phase": 1})
    r, c, phase, _ = env.decode_state(s)
    assert phase == 1

    # Thử đi RIGHT vào cổng G1 đang ĐÓNG (không có pass):
    next_s, rwd, term, trunc, _ = env.step(RIGHT)
    nr, nc, n_phase, _ = env.decode_state(next_s)
    # Bị giữ lại tại (2, 0) và phạt -5.0:
    assert (nr, nc) == (2, 0), f"Phải bị giữ lại tại (2, 0), nhưng lại ở ({nr}, {nc})"
    assert rwd == -5.0, f"Phạt đâm vào cổng đóng phải là -5.0, nhận được: {rwd}"
    assert n_phase == 0, f"Phase tiếp theo phải chuyển về 0, nhận được: {n_phase}"
    print("  [OK] Đâm vào cổng ĐÓNG (không có vé) bị chặn lại và phạt -5.0 chính xác!")

    # 3. Thử hành động WAIT tại phase 1:
    s, _ = env.reset(options={"start_phase": 1})
    next_s, rwd, term, trunc, _ = env.step(WAIT)
    nr, nc, n_phase, _ = env.decode_state(next_s)
    assert (nr, nc) == (2, 0), f"WAIT phải giữ nguyên vị trí, nhưng lại ở ({nr}, {nc})"
    assert rwd == -1.0, f"WAIT chỉ bị phạt bước đi -1.0, nhận được: {rwd}"
    assert n_phase == 0, f"Sau WAIT, phase chuyển từ 1 sang 0 (cổng mở ra)"
    print("  [OK] Hành động WAIT tại phase 1 giữ vị trí, tốn -1.0, phase chuyển thành 0 (Cổng MỞ)!")

    # Ngay sau khi WAIT, đi RIGHT qua cổng MỞ:
    next_s, rwd, term, trunc, _ = env.step(RIGHT)
    nr, nc, n_phase, _ = env.decode_state(next_s)
    assert (nr, nc) == (2, 1), f"Đi qua cổng sau khi WAIT thất bại"
    print("  [OK] Sau khi WAIT nhịp đèn, đi qua cổng thành công!")
    print("=> Test 3 PASSED!\n")


def test_pass_mechanics():
    print("--- [TEST 4]: Kiểm tra cơ chế vé Pass (Map 3 - Setting 4) ---")
    env = make_env("map3")
    # Reset bắt đầu tại (2, 0) với phase = 1 (cổng G1 tại (2,1) ĐÓNG), có 1 vé Pass
    s, _ = env.reset(options={"start_phase": 1, "initial_pass": 1})
    r, c, phase, pass_avail = env.decode_state(s)
    assert pass_avail == 1, "Ban đầu phải có 1 vé Pass"
    assert phase == 1, "Phase ban đầu là 1 (Cổng G1 đang ĐÓNG)"

    # Hành động: Đi RIGHT xuyên qua cổng G1 đang ĐÓNG bằng vé Pass:
    next_s, rwd, term, trunc, _ = env.step(RIGHT)
    nr, nc, n_phase, next_pass = env.decode_state(next_s)
    assert (nr, nc) == (2, 1), f"Có pass phải đi xuyên qua được cổng đóng tới (2, 1), nhưng ở ({nr}, {nc})"
    assert next_pass == 0, f"Vé Pass phải bị tiêu thụ từ 1 về 0, hiện tại là: {next_pass}"
    assert rwd == -1.0, f"Dùng pass chỉ tốn bước đi thông thường -1.0, nhận được: {rwd}"
    print("  [OK] Dùng vé Pass xuyên qua cổng đóng thành công, pass giảm từ 1 xuống 0, phạt bước đi -1.0!")

    # Bây giờ vé pass đã hết (pass=0). Tiến tới (2, 2)
    next_s, _, _, _, _ = env.step(RIGHT)  # sang (2, 2)
    # Tại (2, 2), cổng G2 tại (2, 3) có chu kỳ 3. Giả sử ta ép trạng thái cổng G2 đang ĐÓNG:
    # Thử bước vào G2 đang ĐÓNG khi pass=0:
    s_at_22_closed = env.encode_state(2, 2, phase=1, pass_avail=0)
    env.state = s_at_22_closed
    next_s, rwd, _, _, _ = env.step(RIGHT)
    nr, nc, n_phase, next_pass = env.decode_state(next_s)
    assert (nr, nc) == (2, 2), f"Hết pass phải bị chặn lại tại (2, 2), nhưng ở ({nr}, {nc})"
    assert rwd == -5.0, f"Hết pass đâm vào cổng đóng phải chịu phạt -5.0, nhận được: {rwd}"
    print("  [OK] Khi hết vé Pass, đâm vào cổng đóng tiếp theo bị chặn và phạt -5.0 chuẩn xác!")
    print("=> Test 4 PASSED!\n")


def test_goal_termination():
    print("--- [TEST 5]: Kiểm tra điều kiện kết thúc khi đến Đích (Goal) ---")
    env = make_env("map1", goal_reward=10.0)
    # Đặt agent đứng ngay sát Goal: (2, 2), goal tại (2, 3)
    s_near_goal = env.encode_state(2, 2, phase=0, pass_avail=0)
    env.state = s_near_goal

    next_s, rwd, term, trunc, _ = env.step(RIGHT)
    nr, nc, _, _ = env.decode_state(next_s)
    assert (nr, nc) == (2, 3), f"Tới goal tại (2, 3), thực tế: ({nr}, {nc})"
    assert rwd == 10.0, f"Reward khi tới goal phải là +10.0, nhận được: {rwd}"
    assert term is True, "terminated phải là True khi tới goal!"
    print("  [OK] Về đích thành công: reward = +10.0, terminated = True!")
    print("=> Test 5 PASSED!\n")


def test_gym_compliance():
    print("--- [TEST 6]: Kiểm tra chuẩn Gymnasium với gymnasium.utils.env_checker.check_env ---")
    for map_name in ["map1", "map2", "map3"]:
        env = make_env(map_name)
        check_env(env.unwrapped)
        print(f"  [OK] {map_name.upper()} vượt qua 100% tiêu chí kiểm định của Gymnasium check_env!")
    print("=> Test 6 PASSED!\n")


def test_render_output():
    print("--- [TEST 7]: Thử nghiệm hiển thị ASCII Render ---")
    env = make_env("map1")
    env.reset()
    ascii_art = env.render()
    print(ascii_art)
    print("=> Test 7 PASSED!\n")


if __name__ == "__main__":
    print("==================================================")
    print("TIẾN HÀNH TEST TOÀN DIỆN PHẦN 1: MÔI TRƯỜNG & MAP")
    print("==================================================\n")
    test_map_sizes()
    test_encode_decode_roundtrip()
    test_gate_and_wait_mechanics()
    test_pass_mechanics()
    test_goal_termination()
    test_gym_compliance()
    test_render_output()
    print("==================================================")
    print("TẤT CẢ 7 BÀI TEST ĐỀU ĐẠT 100%! SẴN SÀNG CHO PHẦN 2 & 3")
    print("==================================================")
