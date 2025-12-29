import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib

# 設定中文字型支援
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# ==========================================
# 1. 參數設定 (Configuration)
# ==========================================
DATA_PATH = "../data/task1_dataset_kotae.csv"
MODEL_SAVE_DIR = "../models"
MODEL_SAVE_PATH = f"{MODEL_SAVE_DIR}/seq2seq_multivariate.pth"
SCALER_SAVE_PATH = f"{MODEL_SAVE_DIR}/scaler_multivariate.pkl"
LOG_SAVE_PATH = f"{MODEL_SAVE_DIR}/eval_log_multivariate.txt"

# 模型超參數
INPUT_SEQ_LEN = 144   # 輸入過去 72 小時
OUTPUT_SEQ_LEN = 48   # 預測未來 24 小時
BATCH_SIZE = 512
HIDDEN_SIZE = 256
NUM_LAYERS = 4
EPOCHS = 200
PATIENCE = 20  # Early Stopping 耐心值
LEARNING_RATE = 0.001

# 特徵維度設定
INPUT_SIZE = 2        # [人數, is_weekend]
OUTPUT_SIZE = 1       # [人數]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 資料處理與標籤生成 (Data Processing)
# ==========================================
def load_and_preprocess_data(path):
    print("Loading raw data...")
    raw_df = pd.read_csv(path)
    
    # --- 修正點：先將原始資料聚合算出人數 ---
    print("Aggregating data to calculate 'number of people'...")
    # 計算每個 (d, t, x, y) 組合出現了幾次 (即人數)
    df = raw_df.groupby(['d', 't', 'x', 'y']).size().reset_index(name='number of people')
    
    print(f"Aggregated data shape: {df.shape}")
    
    # --- A. 自動生成 Weekend 標籤 (K-Means) ---
    print("Generating 'is_weekend' labels using K-Means...")
    
    # 選擇總人數最多的網格作為基準
    # 注意：這裡要用 groupby sum 來找總人數最多的點
    top_grid_idx = df.groupby(['x', 'y'])['number of people'].sum().idxmax()
    print(f"Base grid for clustering: {top_grid_idx}")
    
    base_df = df[(df['x'] == top_grid_idx[0]) & (df['y'] == top_grid_idx[1])].copy()
    
    # 注意：為了與 model1.ipynb 對齊比較，此處移除時間補零處理
    # 直接使用原始資料進行後續處理

    # 轉成矩陣: Index=天數, Columns=時間點
    pivot_matrix = base_df.pivot(index='d', columns='t', values='number of people').fillna(0)
    
    # K-Means 分群
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(pivot_matrix)
    labels = kmeans.labels_
    
    # 判斷哪一群是週末
    c0_idx = np.where(labels == 0)[0]
    c1_idx = np.where(labels == 1)[0]
    
    # 比較 t=16 (早上8點) 的平均人流
    if len(c0_idx) > 0 and len(c1_idx) > 0:
        avg_flow_0 = pivot_matrix.iloc[c0_idx, 16].mean()
        avg_flow_1 = pivot_matrix.iloc[c1_idx, 16].mean()
        weekend_label_cluster = 0 if avg_flow_0 < avg_flow_1 else 1
    else:
        # 極端情況處理
        weekend_label_cluster = 1 if len(c0_idx) < len(c1_idx) else 0

    # 建立映射表
    day_is_weekend = [1 if l == weekend_label_cluster else 0 for l in labels]
    label_map = pd.DataFrame({'d': pivot_matrix.index, 'is_weekend': day_is_weekend})
    
    print(f"Weekday count: {len(label_map[label_map['is_weekend']==0])}")
    print(f"Weekend count: {len(label_map[label_map['is_weekend']==1])}")
    
    # --- B. 篩選前三大熱點並合併標籤 ---
    print("Selecting Top 3 locations...")
    # 找出總人數前三多的地點
    top_3 = df.groupby(['x', 'y'])['number of people'].sum().nlargest(3).reset_index()[['x', 'y']]
    
    # 只保留這三個地點的資料
    result_df = pd.merge(df, top_3, on=['x', 'y'], how='inner')
    
    # 合併 K-Means 產生的標籤
    result_df = pd.merge(result_df, label_map, on='d', how='left')
    
    # --- C. 標準化 (Normalization) ---
    scaler = MinMaxScaler()
    result_df['number_scaled'] = scaler.fit_transform(result_df[['number of people']])
    
    return result_df, scaler

# ==========================================
# 3. 自定義 Dataset (Multivariate)
# ==========================================
class GridTimeSeriesDataset(Dataset):
    def __init__(self, df, group_by_cols, target_col, aux_col, input_seq_len, output_seq_len):
        self.sequences = []
        
        grouped = df.groupby(group_by_cols)
        
        for _, group_df in grouped:
            # 正確的排序方式：按 ['d', 't'] 排序
            group_df = group_df.sort_values(['d', 't'])
            
            target_vals = group_df[target_col].values
            aux_vals = group_df[aux_col].values
            
            total_len = len(target_vals)
            
            # 滑動視窗生成序列
            for i in range(total_len - input_seq_len - output_seq_len + 1):
                in_target = target_vals[i : i + input_seq_len]
                in_aux = aux_vals[i : i + input_seq_len]
                
                # Stack: (Seq_Len, 2)
                input_seq = np.stack((in_target, in_aux), axis=1)
                
                # Output: (Output_Len)
                output_seq = target_vals[i + input_seq_len : i + input_seq_len + output_seq_len]
                
                self.sequences.append((input_seq, output_seq))
                
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, output_seq = self.sequences[idx]
        return torch.FloatTensor(input_seq), torch.FloatTensor(output_seq).unsqueeze(-1)

# ==========================================
# 4. 模型架構 (Seq2Seq)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_len, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len
        self.device = device

    def forward(self, source):
        batch_size = source.shape[0]
        hidden, cell = self.encoder(source)
        
        # Decoder 初始輸入設為 0
        decoder_input = torch.zeros(batch_size, 1, 1).to(self.device)
        
        outputs = []
        for _ in range(self.target_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction)
            decoder_input = prediction 
            
        outputs = torch.cat(outputs, dim=1) 
        return outputs

# ==========================================
# 5. 主執行流程
# ==========================================
if __name__ == "__main__":
    
    # --- Step 1: 準備資料 ---
    df, scaler = load_and_preprocess_data(DATA_PATH)
    
    print("Creating dataset...")
    dataset = GridTimeSeriesDataset(
        df, 
        group_by_cols=['x', 'y'],
        target_col='number_scaled',
        aux_col='is_weekend',
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check input sequence length and data continuity.")
        exit()

    # --- 固定切分：40 天訓練，10 天驗證，25 天測試 ---
    TRAIN_DAYS = 40
    VAL_DAYS = 10
    TEST_DAYS = 25
    TOTAL_DAYS = 75
    
    # 重新建立 Dataset：分別為訓練集、驗證集和測試集
    train_df = df[df['d'] < TRAIN_DAYS]
    val_df = df[(df['d'] >= TRAIN_DAYS) & (df['d'] < TRAIN_DAYS + VAL_DAYS)]
    test_df = df[df['d'] >= TRAIN_DAYS + VAL_DAYS]
    
    print(f"訓練集天數: 0 ~ {TRAIN_DAYS-1} (共 {TRAIN_DAYS} 天)")
    print(f"驗證集天數: {TRAIN_DAYS} ~ {TRAIN_DAYS+VAL_DAYS-1} (共 {VAL_DAYS} 天)")
    print(f"測試集天數: {TRAIN_DAYS+VAL_DAYS} ~ {TOTAL_DAYS-1} (共 {TEST_DAYS} 天)")
    
    train_dataset = GridTimeSeriesDataset(
        train_df, 
        group_by_cols=['x', 'y'],
        target_col='number_scaled',
        aux_col='is_weekend',
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN
    )
    
    val_dataset = GridTimeSeriesDataset(
        val_df, 
        group_by_cols=['x', 'y'],
        target_col='number_scaled',
        aux_col='is_weekend',
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN
    )
    
    test_dataset = GridTimeSeriesDataset(
        test_df, 
        group_by_cols=['x', 'y'],
        target_col='number_scaled',
        aux_col='is_weekend',
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # --- Step 2: 建立模型 ---
    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    decoder = Decoder(OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    model = Seq2Seq(encoder, decoder, OUTPUT_SEQ_LEN, DEVICE).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # --- Step 3: 訓練迴圈 (含 Early Stopping) ---
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 驗證階段：使用驗證集計算 validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                val_loss = criterion(output, y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early Stopping 檢查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ Best model updated (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break
    
    # 載入最佳模型並儲存
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'input_seq_len': INPUT_SEQ_LEN,
            'output_seq_len': OUTPUT_SEQ_LEN
        }
    }, MODEL_SAVE_PATH)
            
    print(f"\nTraining complete. Best model saved to {MODEL_SAVE_PATH}")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # --- Step 4: 評估模型（使用測試集）---
    print("\n--- 模型評估 ---")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            output = model(x)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.numpy())
    
    preds = np.concatenate(all_preds, axis=0).reshape(-1, 1)
    targets = np.concatenate(all_targets, axis=0).reshape(-1, 1)
    
    preds_original = scaler.inverse_transform(preds).flatten()
    targets_original = scaler.inverse_transform(targets).flatten()
    
    mse = mean_squared_error(targets_original, preds_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_original, preds_original)
    
    print(f"📈 模型評估結果 (測試集):")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # 儲存評估結果到 log 檔
    with open(LOG_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("模型: Multivariate Seq2Seq (人數 + is_weekend)\n")
        f.write("=" * 50 + "\n")
        f.write(f"輸入特徵數: {INPUT_SIZE}\n")
        f.write(f"訓練集: 前 50 天 (樣本數: {len(train_dataset)})\n")
        f.write(f"測試集: 後 25 天 (樣本數: {len(test_dataset)})\n")
        f.write("\n--- 測試集評估結果 ---\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write("=" * 50 + "\n")
    print(f"📝 評估結果已儲存至 {LOG_SAVE_PATH}")
    model.eval()
    
    # 智慧搜尋：在測試集中搜尋人流波動明顯的樣本
    total_len = len(test_dataset)
    print(f"測試集總數: {total_len}")
    
    # 搜尋熱門時段 (High-traffic sample)
    target_sample_idx = 0
    found = False
    
    print("正在搜尋人流波動明顯的樣本 (Max > 80)...")
    
    for i in range(total_len):
        _, target_tensor = test_dataset[i]
        # 只取人數部分（第一個特徵）進行反標準化
        temp_val = scaler.inverse_transform(target_tensor.numpy().reshape(-1, 1))
        
        if temp_val.max() > 80:
            target_sample_idx = i
            print(f"\u2705 找到目標樣本 Index: {i} (最大人流: {temp_val.max():.2f})")
            found = True
            break
    
    if not found:
        print("\u26a0\ufe0f 未找到 > 80 的樣本，將使用測試集的第一筆資料繪圖。")
    
    # 取得樣本進行預測
    sample_input, sample_target = test_dataset[target_sample_idx]
    sample_input = sample_input.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        prediction = model(sample_input).cpu().numpy().reshape(-1, 1)
        target = sample_target.numpy().reshape(-1, 1)
        
    pred_orig = scaler.inverse_transform(prediction)
    target_orig = scaler.inverse_transform(target)
    
    plt.figure(figsize=(10, 5))
    plt.plot(target_orig, label='Actual (Ground Truth)', linewidth=2)
    plt.plot(pred_orig, label='Predicted (Multivariate Model)', linestyle='--', color='orange', linewidth=2)
    plt.title(f"Multivariate Seq2Seq Prediction (Sample Index: {target_sample_idx})")
    plt.xlabel("Time Steps (Next 24 Hours)")
    plt.ylabel("Number of People")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{MODEL_SAVE_DIR}/prediction_result_multivariate.png")
    plt.show()
    print(f"Result plot saved to {MODEL_SAVE_DIR}/prediction_result_multivariate.png")