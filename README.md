# Timed Gates GridWorld: Tabular Reinforcement Learning Benchmark

## 📌 Giới thiệu đề tài

Trong thực tế, các bài toán điều hướng tự hành (như xe AGV trong kho thông minh hay hệ thống giao thông đô thị) thường xuyên gặp chướng ngại vật biến đổi theo thời gian (**Timed Gates**). 

Môi trường **Timed Gates GridWorld** được thiết kế chuẩn [Gymnasium](https://gymnasium.farama.org/) với các đặc tính:
1. **Bảo toàn tính chất Markov:** Không gian trạng thái 4 thành phần $s = (\text{row}, \text{col}, \text{phase}, \text{pass})$ được ánh xạ song ánh (Linear Indexing) thành `state_id` nguyên duy nhất.
2. **Cơ chế đèn tín hiệu / Cổng chu kỳ:** Cổng mở hoặc đóng dựa theo nhịp $t \pmod P$. 
3. **Đa cổng lệch pha & Đồng bộ LCM:** Nhiều cổng với các chu kỳ khác nhau được đồng bộ toàn cục qua bội chung nhỏ nhất $\text{LCM}(P_1, P_2, \dots)$.
4. **Vé Pass phá cổng:** giúp agent vượt qua cổng mà không bị phạt mất điểm.

Ngoài ra project còn sử dụng **7 thuật toán Tabular RL** để so sánh đối chiếu:
* *Dynamic Programming:** Value Iteration, Policy Iteration, Truncated Policy Iteration ($k=5$).
* *Monte Carlo Methods:** On-Policy First-Visit MC $\varepsilon$-Greedy.
* *Temporal-Difference Learning:** 1-step SARSA (On-Policy), Q-Learning (Off-Policy), và 3-step SARSA (Multi-step TD).

---

## 📂 Cấu trúc thư mục (Repository Structure)

```text
Final_RL/
├── .gitignore               # Danh sách loại trừ file rác, cache cho Git
├── README.md                # Tài liệu hướng dẫn & tổng kết kết quả
├── requirements.txt         # Danh sách thư viện Python cần cài đặt
├── main.ipynb               # Jupyter Notebook thực thi chính và trực quan hóa
├── env.py                   # Mô hình môi trường TimedGateEnv chuẩn Gymnasium
├── algorithms.py            # Cài đặt trọn bộ 7 thuật toán Tabular RL
├── configs.py               # Cấu hình tham số 3 Map (Map 1, Map 2, Map 3)
├── experiments.py           # Mã nguồn 5 kịch bản thực nghiệm chuyên sâu
├── visualize.py             # Hàm trực quan hóa bản đồ, chính sách và learning curve
└── tests/                   # Bộ kiểm thử tự động (Unit Test & Integration Test)
    ├── __init__.py
    ├── test_env.py          # Kiểm tra 7 tiêu chí môi trường và chuẩn Gymnasium
    └── test_algorithms.py   # Kiểm tra tính hội tụ nhanh của 7 thuật toán trên Map 1
```

---

## 🚀 Hướng dẫn cài đặt & Chạy chương trình

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.9 trở lên trong môi trường ảo:

```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
venv\Scripts\activate   # Trên Windows

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Chạy bộ kiểm thử tự động (Tests)
Trước khi chạy thực nghiệm lớn, bạn có thể kiểm tra môi trường và thuật toán:

```bash
# Kiểm tra môi trường (Markov state indexing, cơ chế WAIT, vé Pass, Gymnasium check)
python tests/test_env.py

# Kiểm tra nhanh 7 thuật toán hội tụ trên Map 1
python tests/test_algorithms.py
```

### 3. Chạy toàn bộ thực nghiệm (Main Notebook)
Khởi chạy Jupyter Notebook để xem toàn bộ quá trình huấn luyện và biểu đồ:

```bash
jupyter notebook main.ipynb
```

---

## 📊 Tóm tắt kết quả đối chuẩn (Benchmark)

### Bảng xếp hạng trên Map 2 ($5 \times 5$, $|\mathcal{S}| = 150$, 2 cổng lệch pha $P=6$):

| Thuật toán | Họ lý thuyết | Thời gian CPU | $V(\text{Start})$ | Test Return | Đánh giá lộ trình |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Value Iteration** | Dynamic Programming | 205.3 ms | 3.12 | **+6.0** | Tìm nghiệm tối ưu toàn cục $V^*$, 5 bước chuẩn. |
| **Policy Iteration** | Dynamic Programming | 102.4 ms | 3.12 | **+6.0** | Hội tụ chính sách cực nhanh, đạt lộ trình tối ưu. |
| **Truncated PI ($k=5$)** | Dynamic Programming | 181.8 ms | 3.12 | **+6.0** | Cân bằng hoàn hảo, cho ra cùng chính sách tối ưu. |
| **MC $\varepsilon$-Greedy** | Monte Carlo Methods | 116.5 ms | 1.11 | **+6.0** | Học offline, test tham lam ($\varepsilon=0$) đạt 5 bước tối ưu. |
| **SARSA** | Temporal-Difference | 147.2 ms | 1.73 | **+6.0** | On-policy an toàn, tránh đâm cổng trong quá trình học. |
| **Q-Learning** | Temporal-Difference | 136.0 ms | 3.12 | **+6.0** | Off-policy bám sát $V^*$, tối ưu hóa số bước lý thuyết. |
| **3-step SARSA** | Multi-step TD | **92.9 ms** | 1.57 | **+6.0** | Tốc độ hội tụ nhanh nhất nhóm TD nhờ lan truyền 3 bước. |

---
