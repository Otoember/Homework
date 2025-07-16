# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# %%
# 设置随机种子确保可复现性
import random
import os

def set_all_seeds(seed=42):  # 使用更通用的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# %%
# 加载并预处理数据
def load_and_preprocess_data(file_path):
    # 使用更高效的方式加载大型数据集
    df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['?'])
    
    # 合并日期时间列
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # 处理缺失值 - 使用前向填充+后向填充
    df = df.ffill().bfill()
    
    # 转换为数值类型
    numeric_cols = df.columns.drop('datetime')
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # 添加时间特征
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # 设置索引并按时间排序
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # 移除可能的重复项
    df = df[~df.index.duplicated(keep='first')]
    
    return df

# 加载数据
df = load_and_preprocess_data('data/household_power_consumption.txt')

# %%
# 数据探索性分析
def exploratory_data_analysis(df):
    print("\n=== 数据统计摘要 ===")
    print(df.describe())
    
    print("\n=== 缺失值统计 ===")
    print(df.isnull().sum())
    
    # 绘制目标变量分布
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Global_active_power'], kde=True)
    plt.title('Global Active Power Distribution')
    plt.show()
    
    # 绘制时间序列图
    plt.figure(figsize=(16, 8))
    plt.plot(df['Global_active_power'].resample('D').mean())
    plt.title('Daily Average Global Active Power')
    plt.xlabel('Date')
    plt.ylabel('Power (kW)')
    plt.show()

# 执行EDA
exploratory_data_analysis(df)

# %%
# 划分数据集
def split_time_series(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test

train, val, test = split_time_series(df)

# %%
# 数据标准化 - 使用更稳健的缩放器
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# %%
# 创建序列数据集
def create_sequences(data, seq_length=24, pred_length=1, target_col=0):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+pred_length, target_col]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 超参数配置
SEQ_LENGTH = 72  # 使用72小时(3天)历史数据
PRED_LENGTH = 24  # 预测未来24小时
TARGET_COL = 0   # Global_active_power是第0列

train_X, train_y = create_sequences(train_scaled, SEQ_LENGTH, PRED_LENGTH, TARGET_COL)
val_X, val_y = create_sequences(val_scaled, SEQ_LENGTH, PRED_LENGTH, TARGET_COL)
test_X, test_y = create_sequences(test_scaled, SEQ_LENGTH, PRED_LENGTH, TARGET_COL)

# %%
# 创建PyTorch数据集
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_tensor_datasets(X, y):
    dataset = TensorDataset(
        torch.FloatTensor(X).to(device),
        torch.FloatTensor(y).to(device)
    )
    return dataset

train_dataset = create_tensor_datasets(train_X, train_y)
val_dataset = create_tensor_datasets(val_X, val_y)
test_dataset = create_tensor_datasets(test_X, test_y)

# 使用随机拆分创建更平衡的数据加载器
def create_data_loaders(train_dataset, val_dataset, batch_size=128):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False)
    return train_loader, val_loader

BATCH_SIZE = 128
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, BATCH_SIZE)

# %%
# 改进的LSTM模型
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc(out)
        return out

INPUT_SIZE = train_X.shape[2]
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = PRED_LENGTH

model = TimeSeriesModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE,
    dropout=0.3
).to(device)

# %%
# 训练配置
criterion = nn.HuberLoss()  # 对异常值更鲁棒的损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# %%
# 训练函数
def train_model(model, train_loader, val_loader, epochs=30):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

# 训练模型
EPOCHS = 30
train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS)

# 绘制训练过程
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# %%
# 加载最佳模型进行测试
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# %%
# 创建测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 预测函数
def predict(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    return predictions, actuals

# 在测试集上进行预测
test_preds, test_actuals = predict(model, test_loader)

# %%
# 反标准化函数
def inverse_transform(scaler, data, target_col, n_features):
    # 创建与原始数据相同形状的占位数组
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, target_col] = data
    return scaler.inverse_transform(dummy)[:, target_col]

# 反标准化
n_features = train.shape[1]
test_preds_unscaled = inverse_transform(scaler, test_preds, TARGET_COL, n_features)
test_actuals_unscaled = inverse_transform(scaler, test_actuals, TARGET_COL, n_features)

# 计算评估指标
def calculate_metrics(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.4f}%')
    
    return mae, rmse, mape

# 计算整体指标
print("\n整体性能指标:")
calculate_metrics(test_actuals_unscaled, test_preds_unscaled)

# %%
# 可视化结果 - 选择一周的数据进行展示
def plot_predictions(actuals, predictions, days=7, points_per_day=24):
    n_points = days * points_per_day
    
    plt.figure(figsize=(16, 8))
    plt.plot(actuals[:n_points], label='实际值', linewidth=2)
    plt.plot(predictions[:n_points], label='预测值', linestyle='--', linewidth=1.5)
    
    # 添加每日分隔线
    for i in range(days):
        plt.axvline(x=i*points_per_day, color='gray', linestyle='-', alpha=0.3)
    
    plt.title(f'电力消耗预测对比 - 最近{days}天')
    plt.xlabel('时间 (小时)')
    plt.ylabel('有功功率 (kW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 绘制最近一周的预测结果
plot_predictions(test_actuals_unscaled, test_preds_unscaled, days=7)

# 可视化误差分布
def plot_error_distribution(actuals, predictions):
    errors = actuals - predictions
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, kde=True)
    plt.title('预测误差分布')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.show()

plot_error_distribution(test_actuals_unscaled, test_preds_unscaled)