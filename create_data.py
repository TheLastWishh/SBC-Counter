import numpy as np
import pandas as pd

# ===== 1. Tạo dải thời gian 3 tháng gần nhất, cách nhau 5 phút =====
end_time = pd.Timestamp.now()          # thời điểm hiện tại
start_time = end_time - pd.DateOffset(months=3)

# Tạo các mốc thời gian cách nhau 5 phút
time_index = pd.date_range(start=start_time, end=end_time, freq="5min")
n_time = len(time_index)
print("Số timestamp:", n_time)

# Chuyển sang epoch time (giây, dạng int)
epoch_times = (time_index.view("int64") // 10**9).astype("int64")

# ===== 2. Thông tin node & counter =====
node_ids = np.arange(1, 6 + 1)        # 1..6
counter_ids = np.arange(1, 20 + 1)    # 1..20

n_nodes = len(node_ids)
n_counters = len(counter_ids)
n_rows = n_time * n_nodes * n_counters

print("Tổng số dòng sẽ tạo:", n_rows)

# ===== 3. Tạo các cột thời gian, node, counter =====
trigger_time_col = np.repeat(epoch_times, n_nodes * n_counters)
node_id_col = np.tile(np.repeat(node_ids, n_counters), n_time)
counter_id_col = np.tile(counter_ids, n_time * n_nodes)

# ===== 4. Sinh counter_value theo trend =====
# 4.1. Tính các yếu tố theo giờ / ngày / xu hướng dài hạn

hours = time_index.hour          # 0..23
dow = time_index.dayofweek       # 0=Mon, 6=Sun

# Factor theo giờ trong ngày:
#  - 0-5h  : rất thấp
#  - 6-10h : tăng dần (sáng)
#  - 11-16h: trung bình
#  - 17-21h: cao điểm
#  - 22-23h: hơi giảm
hour_factor = np.select(
    [
        hours < 6,
        hours < 11,
        hours < 17,
        hours < 22
    ],
    [
        0.5,   # 0-5
        0.8,   # 6-10
        1.0,   # 11-16
        1.3    # 17-21 (cao điểm)
    ],
    default=0.6  # 22-23
)

# Factor theo ngày trong tuần:
#  - Thứ 7, CN (5,6) giảm nhẹ 15% (ví dụ hệ doanh nghiệp)
dow_factor = np.where(dow >= 5, 0.85, 1.0)

# Trend tăng dần trong 3 tháng: từ 1.0 lên 1.1 (tăng 10%)
long_term_factor = 1.0 + 0.1 * np.linspace(0, 1, n_time)

# Base level trung bình (khoảng 18000)
base_level = 18000

# baseline theo từng timestamp
baseline_per_ts = base_level * hour_factor * dow_factor * long_term_factor

# 4.2. Nhân lên cho tất cả node × counter và thêm nhiễu
# Lặp baseline theo số node × counter
baseline_repeated = np.repeat(baseline_per_ts, n_nodes * n_counters)

# Nhiễu ngẫu nhiên (noise), để dữ liệu không phẳng
noise = np.random.normal(loc=0, scale=1500, size=n_rows)

values = baseline_repeated + noise

# Giới hạn trong khoảng 10000 -> 30000
values = np.clip(values, 10000, 30000)

counter_value_col = values.astype("int32")

# ===== 5. counter_option mặc định = 0 =====
counter_option_col = np.zeros(n_rows, dtype="int32")

# ===== 6. id tự tăng =====
id_col = np.arange(1, n_rows + 1, dtype="int64")

# ===== 7. Tạo DataFrame =====
df = pd.DataFrame({
    "id": id_col,
    "node_id": node_id_col,
    "counter_id": counter_id_col,
    "counter_value": counter_value_col,
    "counter_option": counter_option_col,
    "trigger_time": trigger_time_col,
})

print(df.head())
print(df.tail())

# ===== 8. Ghi ra CSV =====
df.to_csv("sbc_counter_result.csv", index=False)
print("Đã ghi file sbc_counter_result.csv")
