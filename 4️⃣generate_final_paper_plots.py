# Filename: generate_final_paper_plots.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.metrics import r2_score

# ==============================================================================
# 步驟 1: 在 Colab 中設定繁體中文字型
# ==============================================================================
print("正在下載並設定繁體中文字型...")

# 下載思源黑體
font_path = 'TaipeiSansTCBeta-Regular.ttf'

# 將字型加入 Matplotlib 的字型管理器中
font_manager.fontManager.addfont(font_path)

# 設定 Matplotlib 的 rcParams 來使用新字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("\n字型設定完成。")
print("如果圖表中的中文仍然顯示亂碼，請務必點擊 Colab 選單的『執行階段』->『重新啟動執行階段』，然後再重新運行此儲存格。")


# ==============================================================================
# 步驟 2: 繪圖函式 (已更新為等高線圖)
# ==============================================================================

def plot_prediction_contour(y_true, y_pred, ax):
    """繪製 預測 vs. 真實 的密度等高線圖"""
    sns.kdeplot(x=y_true, y=y_pred, cmap="viridis", fill=True, ax=ax, levels=10, thresh=0.01)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, lw=2, label='完美預測 (y=x)')
    ax.set_title('預測 vs. 真實 (密度等高線圖)', fontsize=16, pad=10)
    ax.set_xlabel('真實 P-value', fontsize=12)
    ax.set_ylabel('預測 P-value', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left')

def plot_residual_contour(y_true, y_pred, ax):
    """繪製殘差的密度等高線圖"""
    residuals = y_true - y_pred
    sns.kdeplot(x=y_pred, y=residuals, cmap="plasma", fill=True, ax=ax, levels=10, thresh=0.01)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8, lw=2, label='零誤差線')
    ax.set_title('殘差密度等高線圖', fontsize=16, pad=10)
    ax.set_xlabel('預測 P-value', fontsize=12)
    ax.set_ylabel('殘差 (真實 - 預測)', fontsize=12)
    ax.legend(loc='upper right')

def plot_distribution(y_true, y_pred, ax):
    """繪製 P-value 的分佈對比圖"""
    sns.kdeplot(y_true, ax=ax, label='真實 P-value 分佈', fill=True, alpha=0.5, lw=2)
    sns.kdeplot(y_pred, ax=ax, label='預測 P-value 分佈', fill=True, alpha=0.5, lw=2)
    ax.set_title('P-value 分佈對比圖', fontsize=16, pad=10)
    ax.set_xlabel('P-value', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.legend(frameon=True)

# ==============================================================================
# 步驟 3: 主生成函式
# ==============================================================================

def generate_full_report(history_obj, y_test_data, y_pred_data, test_names_dict):
    """
    為模型生成一套完整的、使用等高線圖的視覺化分析報告。
    """
    print("\n正在生成論文等級的視覺化圖表...")

    # 繪製總體損失曲線
    plt.figure(figsize=(10, 6))
    ax_loss = plt.gca()
    ax_loss.plot(history_obj.history['loss'], label='訓練損失', lw=2)
    ax_loss.plot(history_obj.history['val_loss'], label='驗證損失', lw=2, linestyle='--')
    ax_loss.set_title('模型訓練過程損失曲線', fontsize=16, pad=10)
    ax_loss.set_xlabel('訓練週期 (Epoch)', fontsize=12)
    ax_loss.set_ylabel('損失值 (Loss)', fontsize=12)
    ax_loss.legend(frameon=True)
    ax_loss.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # 為每個指定的測試項目生成報告
    test_names_list = list(test_names_dict.keys())
    for i, test_name_full in enumerate(test_names_list):
        # 從完整的 Y_test 和 Y_pred 中，根據索引獲取對應的數據
        y_true_single = y_test_data[:, i]
        y_pred_single = y_pred_data[:, i]

        # 計算並顯示該項目的 R² 分數
        r2 = r2_score(y_true_single, y_pred_single)
        print(f"\n--- 正在為『{test_name_full.replace(chr(10), ' ')}』生成圖表 (R² = {r2:.4f}) ---")

        # 創建 1x3 的子圖佈局
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f'對「{test_name_full.replace(chr(10), " ")}」的深度分析', fontsize=20)

        # 繪製三張核心分析圖
        plot_prediction_contour(y_true_single, y_pred_single, axes[0])
        plot_residual_contour(y_true_single, y_pred_single, axes[1])
        plot_distribution(y_true_single.flatten(), y_pred_single.flatten(), axes[2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# ==============================================================================
# 步驟 4: 填入您的真實數據並執行
# ==============================================================================
if __name__ == '__main__':

    # --------------------------------------------------------------------------
    # ⚠️ 請在此處填入您真實的模型結果 ⚠️
    # --------------------------------------------------------------------------

    path = 'AES-CTR_20000'
    
    # --- 這是「模擬」的數據，僅用於讓腳本可以獨立運行 ---
    # 1. 模擬一個 history 物件
    
    class HistoryObject:
        '''
        history = {
            'loss': np.logspace(0, -2, 50),
            'val_loss': np.logspace(0, -1.8, 50) + np.random.rand(50) * 0.05
        }
        '''
        df_history = pd.read_csv(f'./data/{path}/train_history.csv') # <<<<< 替換這裡
        history = df_history.to_dict(orient='list')

    history = HistoryObject()
    
    # 2. 定義您的目標測試列表
    results_df = pd.read_csv(f'./data/{path}/r2_results.csv')
    TARGET_TEST_NAMES_FULL = results_df.set_index('Test')['R2'].to_dict()
    for i in TARGET_TEST_NAMES_FULL.keys():
        TARGET_TEST_NAMES_FULL[i] = round(TARGET_TEST_NAMES_FULL[i], 6)
    '''
    TARGET_TEST_NAMES_FULL = {
        'monobit_test\n(頻率測試)': 0.897090,
        'runs_test\n(遊程測試)': 0.882958,
        'cumulative_sums_test\n(累積和測試)': 0.730935,
        'frequency_within_block_test\n(塊內頻率測試)': 0.630592,
        'serial_test\n(序列測試)': 0.300719,
        'approximate_entropy_test\n(近似熵測試)': 0.299497,
        'longest_run_ones_in_a_block_test\n(塊內最長1遊程)': 0.006391,
        'dft_test\n(光譜測試)': -0.000705,
        'non_overlapping_template_matching_test\n(非重疊模板匹配)': -0.000827,
    }
    '''

    # 3. 模擬 Y_test 和 Y_pred
    # 在您的真實程式碼中，這兩者來自 model.fit() 之後的 train_test_split 和 model.predict()
    num_tests = len(TARGET_TEST_NAMES_FULL)
    num_samples = 50 # 假設測試集有 8000 個樣本

    # 創建一個與真實情況類似的 Y_test
    # 假設 P-value 在 [0,1] 區間內呈現某種非均勻分佈
    Y_test = np.load(f'./data/{path}/y_test.npy') # <<<<< 替換這裡

    # 根據每個測試的 R² 分數，創建一個逼真的 Y_pred
    '''
    Y_pred = np.zeros_like(Y_test)
    mock_r2_scores = list(TARGET_TEST_NAMES_FULL.values())
    for i, r2 in enumerate(mock_r2_scores):
        correlation = np.sqrt(max(0, r2))
        noise_std = 0.15 * np.sqrt(max(0, 1 - max(0, r2)))
        noise = np.random.normal(0, noise_std, num_samples)
        Y_pred[:, i] = correlation * Y_test[:, i] + (1 - correlation) * np.mean(Y_test[:, i]) + noise
    '''
    Y_pred = np.load(f'./data/{path}/y_pred.npy') # <<<<< 替換這裡

    # --- 執行主函式 ---
    generate_full_report(history, Y_test, Y_pred, TARGET_TEST_NAMES_FULL)