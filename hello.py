import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape
import matplotlib.pyplot as plt

# ==== CONFIG ====
csv_path = "sbc_counter_result.csv"

n_past = 24                   # số bước thời gian dùng làm input
target_counters = [6, 9]   # các counter_id cần DỰ ĐOÁN (n_counter)

# ==== LOAD & PIVOT ====
df = pd.read_csv(csv_path)

# bỏ cột không dùng
df = df.drop(columns=['id', 'counter_option'])

# chuẩn hóa thời gian
df['trigger_time'] = pd.to_datetime(df['trigger_time'], unit='s')
df = df.sort_values('trigger_time')

# === pivot TẤT CẢ counter làm feature ===
# index = time, columns = (node_id, counter_id), value = counter_value
df_pivot = df.pivot_table(
    index='trigger_time',
    columns=['node_id', 'counter_id'],
    values='counter_value'
)

df_pivot = df_pivot.sort_index(axis=1)   # sắp xếp cột theo (node_id, counter_id)

print("df_pivot shape:", df_pivot.shape)     # (n_time, n_nodes * n_all_counters)
print("Một vài cột:", df_pivot.columns[:10])
df_pivot.head()

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_pivot.values)

n_time, n_features_all = data_scaled.shape
node_ids = df_pivot.columns.get_level_values(0).unique()
all_counters = df_pivot.columns.get_level_values(1).unique()

n_nodes = len(node_ids)
n_all_counters = len(all_counters)

print("n_time:", n_time)
print("n_features_all:", n_features_all)
print("n_nodes:", n_nodes, "n_all_counters:", n_all_counters)
print("node_ids: ", node_ids)

# Lấy MultiIndex columns
cols = df_pivot.columns  # MultiIndex (node_id, counter_id)

# Tìm index các cột có counter_id thuộc target_counters
target_col_indices = [
    i for i, (nid, cid) in enumerate(zip(cols.get_level_values(0), cols.get_level_values(1)))
    if cid in target_counters
]
print(target_col_indices)

print("Số cột target:", len(target_col_indices))
# n_target_features = n_nodes * n_target_counters
n_target_counters = len(target_counters)
assert len(target_col_indices) == n_nodes * n_target_counters

def create_xy_all_features_target_subset(
    data_scaled,
    n_past,
    target_col_indices,
    n_nodes,
    n_target_counters
):
    X, Y = [], []
    n_time, n_features_all = data_scaled.shape

    for i in range(n_past, n_time):
        # input: n_past bước trước
        X.append(data_scaled[i - n_past:i, :])  # (n_past, n_features_all)

        # output: time step hiện tại, chỉ lấy các cột target
        y_vec = data_scaled[i, target_col_indices]  # (n_nodes * n_target_counters,)
        Y.append(y_vec)

    X = np.array(X)
    Y = np.array(Y)

    # reshape Y về (N, n_nodes, n_target_counters)
    Y = Y.reshape(-1, n_nodes, n_target_counters)

    return X, Y

X, Y = create_xy_all_features_target_subset(
    data_scaled,
    n_past,
    target_col_indices,
    n_nodes,
    n_target_counters
)

print("X:", X.shape)  # (N, n_past, n_features_all)
print("Y:", Y.shape)  # (N, n_nodes, n_target_counters)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("Y_train:", Y_train.shape, "Y_test:", Y_test.shape)

n_past = X_train.shape[1]
n_features_all = X_train.shape[2]

inputs = Input(shape=(n_past, n_features_all))

x = LSTM(128, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.2)(x)

# Dense ra vector cho tất cả node * số counter cần dự đoán
x = Dense(n_nodes * n_target_counters)(x)

# Reshape thành (n_nodes, n_target_counters)
outputs = Reshape((n_nodes, n_target_counters))(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 1. Dự đoán (đang ở scale)
Y_pred_scaled = model.predict(X_test)  # (N_test, n_nodes, n_target_counters)

# Flatten (N_test, n_nodes * n_target_counters)
N_test = Y_test.shape[0]
Y_test_scaled_flat = Y_test.reshape(N_test, n_nodes * n_target_counters)
Y_pred_scaled_flat = Y_pred_scaled.reshape(N_test, n_nodes * n_target_counters)

# 2. Tạo mảng tạm full feature
temp_true = np.zeros((N_test, n_features_all))
temp_pred = np.zeros((N_test, n_features_all))

# 3. Gán các cột target vào đúng vị trí
temp_true[:, target_col_indices] = Y_test_scaled_flat
temp_pred[:, target_col_indices] = Y_pred_scaled_flat

# 4. inverse_transform cho toàn bộ rồi lấy lại phần cần
temp_true_inv = scaler.inverse_transform(temp_true)
temp_pred_inv = scaler.inverse_transform(temp_pred)

Y_test_inv_flat = temp_true_inv[:, target_col_indices]
Y_pred_inv_flat = temp_pred_inv[:, target_col_indices]

# 5. reshape lại (N_test, n_nodes, n_target_counters)
Y_test_inv = Y_test_inv_flat.reshape(N_test, n_nodes, n_target_counters)
Y_pred_inv = Y_pred_inv_flat.reshape(N_test, n_nodes, n_target_counters)

# Hàm đánh giá chi tiết từng node × counter
def evaluate_per_node_counter(Y_test, Y_pred_scaled, target_counters, node_ids):
    """
    Tính MAE, RMSE cho từng (node_id, counter_id).
    """
    N_test, n_nodes, n_counters = Y_test.shape

    results = []

    for i in range(n_nodes):
        for j in range(n_counters):
            actual = Y_test[:, i, j]
            pred   = Y_pred_scaled[:, i, j]

            mae  = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))

            results.append({
                "node_id": node_ids[i],
                "counter_id": target_counters[j],
                "MAE": mae,
                "RMSE": rmse
            })

    return results

    
results_node_counter = evaluate_per_node_counter(
    Y_test, Y_pred_scaled, target_counters, node_ids
)

df_eval = pd.DataFrame(results_node_counter)
df_eval

# Hàm đánh giá theo từng counter
def evaluate_per_counter(Y_test, Y_pred_scaled, target_counters):
    """
    MAE/RMSE cho từng counter_id (toàn bộ node).
    """
    N_test, n_nodes, n_counters = Y_test.shape

    results = []

    for j in range(n_counters):
        actual = Y_test[:, :, j].reshape(-1)   # flatten toàn bộ node
        pred   = Y_pred_scaled[:, :, j].reshape(-1)

        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

        results.append({
            "counter_id": target_counters[j],
            "MAE": mae,
            "RMSE": rmse
        })

    return results

df_counter = pd.DataFrame(evaluate_per_counter(Y_test, Y_pred_scaled, target_counters))
df_counter

# Hàm đánh giá toàn mô hình
def evaluate_overall(Y_test, Y_pred_scaled):
    """
    MAE / RMSE toàn mô hình (mọi node + mọi counter).
    """
    actual = Y_test.reshape(-1)
    pred   = Y_pred_scaled.reshape(-1)

    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))

    return {"MAE": mae, "RMSE": rmse}

overall = evaluate_overall(Y_test, Y_pred_scaled)
overall

def plot_single_node_counter(
    time_index_test,
    Y_test_inv, 
    Y_pred_inv,
    node_idx,            # index node (0..n_nodes-1)
    counter_idx,         # index counter trong target_counters (0..n_target_counters-1)
    target_counters,
    node_ids,
    title_prefix="Prediction vs Actual"
):
    """
    Vẽ biểu đồ đường cho 1 node và 1 counter.
    """
    actual = Y_test_inv[:, node_idx, counter_idx]
    pred   = Y_pred_inv[:, node_idx, counter_idx]

    counter_id = target_counters[counter_idx]
    node_id    = node_ids[node_idx]

    plt.figure(figsize=(14,5))
    plt.plot(time_index_test, actual, label="Actual", linewidth=2)
    plt.plot(time_index_test, pred, label="Predicted", linestyle="--")
    
    plt.title(f"{title_prefix} | Node {node_id} | Counter {counter_id}")
    plt.xlabel("Time")
    plt.ylabel("Counter Value")
    plt.grid(True)
    plt.legend()
    plt.show()

time_index = df_pivot.index[n_past:]       # thời gian tương ứng với X,Y
time_index_train = time_index[:len(X_train)]
time_index_test  = time_index[len(X_train):]

plot_single_node_counter(
    time_index_test,
    Y_test_inv,
    Y_pred_inv,
    node_idx=0,
    counter_idx=0,
    target_counters=target_counters,
    node_ids=node_ids
)

def plot_single_node_counter_timerange(
    time_index_test,
    Y_test_inv,
    Y_pred_inv,
    node_idx,
    counter_idx,
    target_counters,
    node_ids,
    start_time,
    end_time,
    title_prefix="Prediction vs Actual"
):
    """
    Vẽ biểu đồ đường cho 1 node và 1 counter trong khoảng thời gian chỉ định.
    
    start_time, end_time: dạng string '2025-01-01 00:00:00' hoặc datetime
    """

    # Convert thời gian nếu cần
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)

    # --- Tìm index trong đoạn thời gian X -> Y ---
    mask = (time_index_test >= start_time) & (time_index_test <= end_time)

    if mask.sum() == 0:
        print("❌ Không có dữ liệu trong khoảng thời gian yêu cầu!")
        return

    t = time_index_test[mask]
    actual = Y_test_inv[mask, node_idx, counter_idx]
    pred   = Y_pred_inv[mask, node_idx, counter_idx]

    counter_id = target_counters[counter_idx]
    node_id = node_ids[node_idx]

    # --- Vẽ ---
    plt.figure(figsize=(14, 5))
    plt.plot(t, actual, label="Actual", linewidth=2)
    plt.plot(t, pred, label="Prediction", linestyle="--")

    plt.title(f"{title_prefix} | Node {node_id} | Counter {counter_id}\n{start_time} → {end_time}")
    plt.xlabel("Time")
    plt.ylabel("Counter Value")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_single_node_counter_timerange(
    time_index_test,
    Y_test_inv,
    Y_pred_inv,
    node_idx=0,
    counter_idx=2,
    target_counters=target_counters,
    node_ids=node_ids,
    start_time="2025-11-20 00:00:00",
    end_time="2025-11-21 00:00:00"
)

def plot_all_nodes_timerange(
    time_index_test,
    Y_test_inv,
    Y_pred_inv,
    counter_idx,
    target_counters,
    node_ids,
    start_time,
    end_time
):
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)

    mask = (time_index_test >= start_time) & (time_index_test <= end_time)

    if mask.sum() == 0:
        print("❌ Không có dữ liệu trong khoảng thời gian yêu cầu!")
        return

    t = time_index_test[mask]
    n_nodes = Y_test_inv.shape[1]

    plt.figure(figsize=(14, 6))

    for i in range(n_nodes):
        plt.plot(t, Y_test_inv[mask, i, counter_idx], label=f"Actual N{i}")
        plt.plot(t, Y_pred_inv[mask, i, counter_idx], linestyle="--", label=f"Pred N{i}")

    counter_id = target_counters[counter_idx]

    plt.title(f"Actual vs Pred for counter {counter_id}\n{start_time} → {end_time}")
    plt.xlabel("Time")
    plt.ylabel("Counter Value")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.show()

plot_all_nodes_timerange(
    time_index_test,
    Y_test_inv,
    Y_pred_inv,
    counter_idx=1,
    target_counters=target_counters,
    node_ids=node_ids,
    start_time="2025-11-20 00:00:00",
    end_time="2025-11-23 00:00:00"
)

def evaluate_weekday_vs_weekend(Y_test, Y_pred, time_index_test):
    """
    Đánh giá MAE, RMSE giữa ngày trong tuần (Mon-Fri) và cuối tuần (Sat-Sun).
    
    Y_test_inv, Y_pred_inv: (N_test, n_nodes, n_target_counters)
    time_index_test: DatetimeIndex hoặc Series có length = N_test
    """
    # Đảm bảo time_index_test là DatetimeIndex
    time_index_test = pd.to_datetime(time_index_test)
    
    # Tạo mask
    weekday_mask = time_index_test.weekday < 5    # 0..4: Mon-Fri
    weekend_mask = time_index_test.weekday >= 5   # 5..6: Sat-Sun
    
    results = {}
    
    def _calc_metrics(mask, name):
        if mask.sum() == 0:
            return {f"{name}_MAE": np.nan, f"{name}_RMSE": np.nan}
        
        actual = Y_test[mask].reshape(-1)
        pred   = Y_pred[mask].reshape(-1)
        
        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        return {f"{name}_MAE": mae, f"{name}_RMSE": rmse}
    
    results.update(_calc_metrics(weekday_mask, "weekday"))
    results.update(_calc_metrics(weekend_mask, "weekend"))
    
    return results

weekday_weekend_metrics = evaluate_weekday_vs_weekend(
    Y_test, Y_pred_scaled, time_index_test
)
print(weekday_weekend_metrics)

# Hàm đánh giá theo từng giờ trong ngày
def evaluate_by_hour_of_day(Y_test, Y_pred, time_index_test):
    """
    Đánh giá MAE, RMSE cho từng giờ trong ngày (0-23).
    
    Returns: list[dict] để dễ chuyển thành DataFrame.
    """
    time_index_test = pd.to_datetime(time_index_test)
    hours = time_index_test.hour.values  # array (N_test,)
    
    results = []
    
    for h in range(24):
        mask = (hours == h)
        if mask.sum() == 0:
            # Không có mẫu nào ở giờ này
            results.append({
                "hour": h,
                "MAE": np.nan,
                "RMSE": np.nan,
                "n_samples": 0
            })
            continue
        
        actual = Y_test[mask].reshape(-1)
        pred   = Y_pred[mask].reshape(-1)
        
        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        
        results.append({
            "hour": h,
            "MAE": mae,
            "RMSE": rmse,
            "n_samples": mask.sum()
        })
    
    return results

hour_metrics = evaluate_by_hour_of_day(Y_test, Y_pred_scaled, time_index_test)

import pandas as pd
df_hour_metrics = pd.DataFrame(hour_metrics)
print(df_hour_metrics)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(df_hour_metrics["hour"], df_hour_metrics["MAE"], marker="o", label="MAE")
plt.plot(df_hour_metrics["hour"], df_hour_metrics["RMSE"], marker="s", label="RMSE")
plt.xlabel("Hour of day")
plt.ylabel("Error")
plt.title("MAE / RMSE theo từng giờ trong ngày")
plt.grid(True)
plt.legend()
plt.xticks(range(0, 24))
plt.show()