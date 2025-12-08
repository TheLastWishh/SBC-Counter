import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Đọc dữ liệu
df = pd.read_csv('sbc_counter_result.csv')

# Chuyển đổi trigger_time sang datetime
df['datetime'] = pd.to_datetime(df['trigger_time'], unit='s')
df = df.sort_values(['node_id', 'counter_id', 'trigger_time'])

print("Thông tin dữ liệu:")
print(f"Số lượng bản ghi: {len(df)}")
print(f"Số node: {df['node_id'].nunique()}")
print(f"Số counter: {df['counter_id'].nunique()}")
print(f"Khoảng thời gian: {df['datetime'].min()} đến {df['datetime'].max()}")
print("\n" + "="*70 + "\n")

# Tạo features từ thời gian
def create_time_features(data):
    """Tạo các đặc trưng từ timestamp"""
    data['hour'] = data['datetime'].dt.hour / 24.0  # Normalize
    data['day'] = data['datetime'].dt.day / 31.0
    data['dayofweek'] = data['datetime'].dt.dayofweek / 7.0
    data['month'] = data['datetime'].dt.month / 12.0
    return data

df = create_time_features(df)

# Chuẩn bị dữ liệu cho LSTM
def prepare_sequences(data, node_id, counter_id, seq_length=24, test_split=0.8):
    """
    Chuẩn bị sequences cho LSTM
    seq_length: số bước thời gian sử dụng để dự đoán bước tiếp theo
    """
    # Lọc dữ liệu theo node_id và counter_id
    subset = data[(data['node_id'] == node_id) & 
                  (data['counter_id'] == counter_id)].copy()
    subset = subset.sort_values('trigger_time')
    
    if len(subset) < seq_length + 1:
        return None, None, None, None, None
    
    # Features: counter_value và time features
    features = subset[['counter_value', 'hour', 'day', 'dayofweek', 'month']].values
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Tạo sequences
    X, y = [], []
    for i in range(len(features_scaled) - seq_length):
        X.append(features_scaled[i:i+seq_length])
        y.append(features_scaled[i+seq_length, 0])  # Chỉ dự đoán counter_value
    
    X = np.array(X)
    y = np.array(y)
    
    # Chia train/test
    split_idx = int(len(X) * test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

# Xây dựng mô hình LSTM
def build_lstm_model(seq_length, n_features):
    """Xây dựng mô hình LSTM"""
    model = keras.Sequential([
        layers.LSTM(128, activation='relu', return_sequences=True, 
                   input_shape=(seq_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Đánh giá mô hình
def evaluate_predictions(y_true, y_pred, scaler):
    """Đánh giá hiệu suất mô hình"""
    # Inverse transform để có giá trị thực
    y_true_inv = scaler.inverse_transform(
        np.concatenate([y_true.reshape(-1, 1), 
                       np.zeros((len(y_true), 4))], axis=1)
    )[:, 0]
    
    y_pred_inv = scaler.inverse_transform(
        np.concatenate([y_pred.reshape(-1, 1), 
                       np.zeros((len(y_pred), 4))], axis=1)
    )[:, 0]
    
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2 = r2_score(y_true_inv, y_pred_inv)
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-8))) * 100
    
    return mae, rmse, r2, mape, y_true_inv, y_pred_inv

# Hàm train và evaluate cho một cặp node_id, counter_id
def train_and_evaluate(data, node_id, counter_id, seq_length=24, epochs=50, batch_size=32):
    """Train và đánh giá mô hình cho một cặp node_id, counter_id"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Node {node_id}, Counter {counter_id}")
    print(f"{'='*70}")
    
    # Chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test, scaler = prepare_sequences(
        data, node_id, counter_id, seq_length
    )
    
    if X_train is None:
        print(f"Không đủ dữ liệu cho Node {node_id}, Counter {counter_id}")
        return None
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Sequence length: {seq_length}, Features: {X_train.shape[2]}")
    
    # Xây dựng mô hình
    model = build_lstm_model(seq_length, X_train.shape[2])
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Training
    print("\nBắt đầu training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Dự đoán
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    
    # Đánh giá
    print("\n" + "-"*70)
    print("KẾT QUẢ TRAIN SET:")
    mae_train, rmse_train, r2_train, mape_train, _, _ = evaluate_predictions(
        y_train, y_pred_train, scaler
    )
    print(f"  MAE:  {mae_train:.2f}")
    print(f"  RMSE: {rmse_train:.2f}")
    print(f"  R²:   {r2_train:.4f}")
    print(f"  MAPE: {mape_train:.2f}%")
    
    print("\n" + "-"*70)
    print("KẾT QUẢ TEST SET:")
    mae_test, rmse_test, r2_test, mape_test, y_true_inv, y_pred_inv = evaluate_predictions(
        y_test, y_pred_test, scaler
    )
    print(f"  MAE:  {mae_test:.2f}")
    print(f"  RMSE: {rmse_test:.2f}")
    print(f"  R²:   {r2_test:.4f}")
    print(f"  MAPE: {mape_test:.2f}%")
    
    # Vẽ biểu đồ training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'Model Loss - Node {node_id}, Counter {counter_id}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title(f'Model MAE - Node {node_id}, Counter {counter_id}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Vẽ biểu đồ dự đoán
    n_plot = min(200, len(y_true_inv))
    plt.figure(figsize=(14, 5))
    plt.plot(y_true_inv[:n_plot], label='Giá trị thực', marker='o', markersize=3, linewidth=1.5)
    plt.plot(y_pred_inv[:n_plot], label='Giá trị dự đoán', marker='x', markersize=3, linewidth=1.5)
    plt.title(f'Dự đoán Counter Value - Node {node_id}, Counter {counter_id}')
    plt.xlabel('Thời điểm')
    plt.ylabel('Counter Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'metrics': {
            'mae_train': mae_train,
            'rmse_train': rmse_train,
            'r2_train': r2_train,
            'mape_train': mape_train,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'mape_test': mape_test
        }
    }

# Train mô hình cho một số node_id và counter_id mẫu
print("\n" + "="*70)
print("BẮT ĐẦU TRAINING MÔ HÌNH LSTM")
print("="*70)

# Lấy 2 node và 2 counter đầu tiên để demo
sample_nodes = df['node_id'].unique()[:2]
sample_counters = df['counter_id'].unique()[:2]

# Dictionary lưu trữ các mô hình đã train
trained_models = {}

seq_length = 24  # Sử dụng 24 bước thời gian trước đó để dự đoán
epochs = 50
batch_size = 32

for node in sample_nodes:
    for counter in sample_counters:
        key = f"node_{node}_counter_{counter}"
        result = train_and_evaluate(
            df, node, counter, 
            seq_length=seq_length, 
            epochs=epochs, 
            batch_size=batch_size
        )
        if result is not None:
            trained_models[key] = result

# Hàm dự đoán cho dữ liệu mới
def predict_future(model, scaler, last_sequence, n_steps=10):
    """
    Dự đoán n_steps bước tiếp theo
    
    Parameters:
    - model: Mô hình LSTM đã train
    - scaler: MinMaxScaler đã fit
    - last_sequence: Sequence cuối cùng (đã scaled) shape (seq_length, n_features)
    - n_steps: Số bước cần dự đoán
    
    Returns:
    - predictions: Mảng giá trị dự đoán (đã inverse transform)
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_steps):
        # Dự đoán bước tiếp theo
        pred = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]), 
                            verbose=0)[0, 0]
        predictions.append(pred)
        
        # Cập nhật sequence (giữ nguyên time features, chỉ update counter_value)
        new_row = current_seq[-1].copy()
        new_row[0] = pred  # Update counter_value
        
        # Shift sequence
        current_seq = np.vstack([current_seq[1:], new_row])
    
    # Inverse transform
    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(
        np.concatenate([predictions.reshape(-1, 1), 
                       np.zeros((len(predictions), 4))], axis=1)
    )[:, 0]
    
    return predictions_inv

print("\n" + "="*70)
print("HOÀN THÀNH TRAINING")
print("="*70)
print(f"\nĐã train {len(trained_models)} mô hình")
print("\nCác mô hình đã train:", list(trained_models.keys()))

print("\n" + "="*70)
print("HƯỚNG DẪN SỬ DỤNG")
print("="*70)
print("""
1. Train mô hình cho các node và counter khác:
   result = train_and_evaluate(df, node_id, counter_id, seq_length=24, epochs=50)

2. Dự đoán tương lai:
   # Lấy sequence cuối cùng từ dữ liệu
   last_seq = X_test[-1]  # hoặc từ dữ liệu mới
   predictions = predict_future(model, scaler, last_seq, n_steps=10)

3. Lưu mô hình:
   model.save('lstm_model_node1_counter1.h5')
   
4. Load mô hình:
   loaded_model = keras.models.load_model('lstm_model_node1_counter1.h5')
""")