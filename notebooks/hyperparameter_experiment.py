"""
AI 期末作業 - Seq2Seq 超參數實驗
=================================
此腳本基於 Model 1 (Univariate Seq2Seq) 進行系統性的超參數實驗，
測試不同的 Hidden Size、Num Layers、Learning Rate 組合，
並將結果整理成表格和視覺化圖表。
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
import matplotlib

# 設定中文字型支援
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# ==========================================
# 1. 參數設定
# ==========================================
DATA_PATH = "../data/task1_dataset_kotae.csv"
MODEL_SAVE_DIR = "../models"
RESULT_SAVE_DIR = "../models/hyperparameter_results"

# 固定參數
INPUT_SEQ_LEN = 144   # 輸入過去 72 小時
OUTPUT_SEQ_LEN = 48   # 預測未來 24 小時
BATCH_SIZE = 512
INPUT_SIZE = 1
OUTPUT_SIZE = 1

# 資料切分設定
TRAIN_DAYS = 40
VAL_DAYS = 10
TEST_DAYS = 25

# 訓練設定
EPOCHS = 100          # 調低以加速實驗
PATIENCE = 15         # Early Stopping 耐心值

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 超參數實驗配置
# ==========================================
HYPERPARAMETER_GRID = {
    'hidden_size': [64, 128, 256],
    'num_layers': [1, 2, 4],
    'learning_rate': [0.001, 0.0005, 0.0001]
}

# ==========================================
# 3. 資料處理
# ==========================================
def load_and_preprocess_data(path):
    print("Loading raw data...")
    raw_df = pd.read_csv(path)
    
    # 聚合計算人數
    df = raw_df.groupby(['d', 't', 'x', 'y']).size().reset_index(name='number of people')
    print(f"Aggregated data shape: {df.shape}")
    
    # 篩選前三大熱點
    top_3 = df.groupby(['x', 'y'])['number of people'].sum().nlargest(3).reset_index()[['x', 'y']]
    result_df = pd.merge(df, top_3, on=['x', 'y'], how='inner')
    
    # 標準化
    scaler = MinMaxScaler()
    result_df['number_scaled'] = scaler.fit_transform(result_df[['number of people']])
    
    return result_df, scaler

# ==========================================
# 4. Dataset
# ==========================================
class GridTimeSeriesDataset(Dataset):
    def __init__(self, df, group_by_cols, target_col, input_seq_len, output_seq_len):
        self.sequences = []
        grouped = df.groupby(group_by_cols)
        
        for _, group_df in grouped:
            group_df = group_df.sort_values(['d', 't'])
            values = group_df[target_col].values
            total_len = len(values)
            
            for i in range(total_len - input_seq_len - output_seq_len + 1):
                input_seq = values[i : i + input_seq_len]
                output_seq = values[i + input_seq_len : i + input_seq_len + output_seq_len]
                self.sequences.append((input_seq, output_seq))
                
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, output_seq = self.sequences[idx]
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(-1)
        output_tensor = torch.FloatTensor(output_seq).unsqueeze(-1)
        return input_tensor, output_tensor

# ==========================================
# 5. 模型架構
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_len, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len
        self.device = device
        
    def forward(self, src):
        batch_size = src.shape[0]
        output_size = self.decoder.output_size
        
        hidden, cell = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, output_size, device=src.device).float()
        
        outputs = []
        for _ in range(self.target_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction)
            decoder_input = prediction.unsqueeze(1)

        outputs = torch.stack(outputs, dim=1)
        return outputs

# ==========================================
# 6. 訓練函數
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience):
    """訓練模型並回傳最佳驗證 Loss 和訓練歷史"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 訓練階段
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
        
        # 驗證階段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                val_loss = criterion(output, y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, history, epoch + 1

# ==========================================
# 7. 評估函數
# ==========================================
def evaluate_model(model, test_loader, scaler):
    """評估模型並回傳 MSE, RMSE, MAE"""
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
    
    return mse, rmse, mae

# ==========================================
# 8. 主執行流程
# ==========================================
if __name__ == "__main__":
    
    # --- Step 1: 準備資料 ---
    df, scaler = load_and_preprocess_data(DATA_PATH)
    
    train_df = df[df['d'] < TRAIN_DAYS]
    val_df = df[(df['d'] >= TRAIN_DAYS) & (df['d'] < TRAIN_DAYS + VAL_DAYS)]
    test_df = df[df['d'] >= TRAIN_DAYS + VAL_DAYS]
    
    train_dataset = GridTimeSeriesDataset(train_df, ['x', 'y'], 'number_scaled', INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    val_dataset = GridTimeSeriesDataset(val_df, ['x', 'y'], 'number_scaled', INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    test_dataset = GridTimeSeriesDataset(test_df, ['x', 'y'], 'number_scaled', INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"訓練樣本: {len(train_dataset)}, 驗證樣本: {len(val_dataset)}, 測試樣本: {len(test_dataset)}")
    
    # --- Step 2: 超參數實驗 ---
    results = []
    experiment_id = 0
    total_experiments = len(HYPERPARAMETER_GRID['hidden_size']) * len(HYPERPARAMETER_GRID['num_layers']) * len(HYPERPARAMETER_GRID['learning_rate'])
    
    print(f"\n{'='*60}")
    print(f"開始超參數實驗 (共 {total_experiments} 組)")
    print(f"{'='*60}")
    
    for hidden_size in HYPERPARAMETER_GRID['hidden_size']:
        for num_layers in HYPERPARAMETER_GRID['num_layers']:
            for learning_rate in HYPERPARAMETER_GRID['learning_rate']:
                experiment_id += 1
                print(f"\n[{experiment_id}/{total_experiments}] Hidden={hidden_size}, Layers={num_layers}, LR={learning_rate}")
                
                start_time = time.time()
                
                # 建立模型
                encoder = Encoder(INPUT_SIZE, hidden_size, num_layers).to(DEVICE)
                decoder = Decoder(OUTPUT_SIZE, hidden_size, num_layers).to(DEVICE)
                model = Seq2Seq(encoder, decoder, OUTPUT_SEQ_LEN, DEVICE).to(DEVICE)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()
                
                # 訓練
                best_val_loss, history, stopped_epoch = train_model(
                    model, train_loader, val_loader, optimizer, criterion, EPOCHS, PATIENCE
                )
                
                # 評估
                mse, rmse, mae = evaluate_model(model, test_loader, scaler)
                
                elapsed_time = time.time() - start_time
                
                print(f"  → MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f} (Epoch: {stopped_epoch}, Time: {elapsed_time:.1f}s)")
                
                # 儲存結果
                results.append({
                    'Hidden Size': hidden_size,
                    'Num Layers': num_layers,
                    'Learning Rate': learning_rate,
                    'Best Val Loss': best_val_loss,
                    'Test MSE': mse,
                    'Test RMSE': rmse,
                    'Test MAE': mae,
                    'Stopped Epoch': stopped_epoch,
                    'Train Time (s)': elapsed_time
                })
    
    # --- Step 3: 整理結果表格 ---
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test MSE')
    
    print(f"\n{'='*80}")
    print("超參數實驗結果 (按 Test MSE 排序)")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # 儲存表格為 CSV
    csv_path = f"{RESULT_SAVE_DIR}/hyperparameter_results.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📝 結果表格已儲存至 {csv_path}")
    
    # --- Step 4: 視覺化 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 4.1 Hidden Size 對 MSE 的影響
    ax1 = axes[0, 0]
    for lr in HYPERPARAMETER_GRID['learning_rate']:
        subset = results_df[results_df['Learning Rate'] == lr]
        avg_by_hidden = subset.groupby('Hidden Size')['Test MSE'].mean()
        ax1.plot(avg_by_hidden.index, avg_by_hidden.values, marker='o', label=f'LR={lr}')
    ax1.set_xlabel('Hidden Size')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Hidden Size vs Test MSE (by Learning Rate)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 4.2 Num Layers 對 MSE 的影響
    ax2 = axes[0, 1]
    for lr in HYPERPARAMETER_GRID['learning_rate']:
        subset = results_df[results_df['Learning Rate'] == lr]
        avg_by_layers = subset.groupby('Num Layers')['Test MSE'].mean()
        ax2.plot(avg_by_layers.index, avg_by_layers.values, marker='s', label=f'LR={lr}')
    ax2.set_xlabel('Num Layers')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Num Layers vs Test MSE (by Learning Rate)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 4.3 Learning Rate 對 MSE 的影響
    ax3 = axes[1, 0]
    for hs in HYPERPARAMETER_GRID['hidden_size']:
        subset = results_df[results_df['Hidden Size'] == hs]
        avg_by_lr = subset.groupby('Learning Rate')['Test MSE'].mean()
        ax3.plot(avg_by_lr.index, avg_by_lr.values, marker='^', label=f'Hidden={hs}')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Test MSE')
    ax3.set_title('Learning Rate vs Test MSE (by Hidden Size)')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4.4 Top 5 最佳參數組合 (Bar Chart)
    ax4 = axes[1, 1]
    top5 = results_df.head(5)
    labels = [f"H{r['Hidden Size']}-L{r['Num Layers']}-{r['Learning Rate']}" for _, r in top5.iterrows()]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c']
    bars = ax4.barh(labels[::-1], top5['Test MSE'].values[::-1], color=colors[::-1])
    ax4.set_xlabel('Test MSE')
    ax4.set_title('Top 5 超參數組合 (MSE 越低越好)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 在 bar 上顯示數值
    for bar, val in zip(bars, top5['Test MSE'].values[::-1]):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{val:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Seq2Seq 超參數實驗結果', fontsize=14, y=1.02)
    
    fig_path = f"{RESULT_SAVE_DIR}/hyperparameter_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 視覺化圖表已儲存至 {fig_path}")
    
    # --- Step 5: 輸出最佳參數 ---
    best_result = results_df.iloc[0]
    print(f"\n{'='*60}")
    print("🏆 最佳超參數組合")
    print(f"{'='*60}")
    print(f"  Hidden Size:   {best_result['Hidden Size']}")
    print(f"  Num Layers:    {best_result['Num Layers']}")
    print(f"  Learning Rate: {best_result['Learning Rate']}")
    print(f"  Test MSE:      {best_result['Test MSE']:.4f}")
    print(f"  Test RMSE:     {best_result['Test RMSE']:.4f}")
    print(f"  Test MAE:      {best_result['Test MAE']:.4f}")
    print(f"{'='*60}")
    
    # 儲存最佳參數到文字檔
    summary_path = f"{RESULT_SAVE_DIR}/best_hyperparameters.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Seq2Seq 超參數實驗 - 最佳參數\n")
        f.write(f"實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Hidden Size:   {best_result['Hidden Size']}\n")
        f.write(f"Num Layers:    {best_result['Num Layers']}\n")
        f.write(f"Learning Rate: {best_result['Learning Rate']}\n")
        f.write(f"\n--- 測試集評估結果 ---\n")
        f.write(f"MSE:  {best_result['Test MSE']:.4f}\n")
        f.write(f"RMSE: {best_result['Test RMSE']:.4f}\n")
        f.write(f"MAE:  {best_result['Test MAE']:.4f}\n")
        f.write("=" * 50 + "\n")
    print(f"📝 最佳參數已儲存至 {summary_path}")
