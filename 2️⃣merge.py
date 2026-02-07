# ==============================================================================
#  Google Colab 數據集合併與抽樣處理腳本 (單一執行版本)
# ==============================================================================

# --- 步驟 0: 導入函式庫與掛載雲端硬碟 ---
import pandas as pd
import os

# ==============================================================================
#  ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
#  --- 步驟 1: 設定檔案路徑與名稱 (唯一需要修改的區域) ---
#
#  請確保此路徑與您在雲端硬碟中建立的資料夾路徑一致。
#  /content/drive/MyDrive/ 是 Colab 連接後您雲端硬碟的根目錄。
DRIVE_FOLDER_PATH = "/content/drive/MyDrive/國科會超大數據/"

# 請根據您上傳的檔案名稱，修改下方的變數。
aes_file_path = "/home/a0919/Entropy_Analysis/AES-CTR_20000.csv"      # <--- 包含 2500 筆 AES 數據的檔名
rc4_file_1_path = "/home/a0919/Entropy_Analysis/RC4_20000.csv"  # <--- 第一份 500 筆 RC4 的檔名


# 設定最終輸出的檔名與路徑
output_file_path = "AES-CTR+RC4_40000.csv"
#  ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==============================================================================


# --- 步驟 2: 檢查檔案是否存在 ---
print("\n--- 步驟 2: 檢查檔案是否存在 ---")
file_paths = [aes_file_path, rc4_file_1_path]
all_files_exist = all(os.path.exists(p) for p in file_paths)

if not all_files_exist:
    print("❌ 錯誤：找不到一個或多個指定的檔案。請檢查：")
    print("1. 上方 DRIVE_FOLDER_PATH 是否正確。")
    print("2. 檔案名稱是否與您雲端硬碟中的檔案完全相符。")
    print("3. 您是否已將檔案上傳至指定的資料夾。")
    # 如果檔案不存在，則停止執行
    raise SystemExit("檔案檢查失敗，請更正路徑後重試。")
else:
    print("✅ 所有輸入檔案均已找到。")


# --- 步驟 3: 讀取並抽樣 AES 數據 ---
print("\n--- 步驟 3: 處理 AES 數據 ---")
print(f"正在從 '{aes_file_path}' 讀取 AES 數據...")
aes_df = pd.read_csv(aes_file_path)
print(f"原始 AES 數據維度：{aes_df.shape}")

# 從中隨機抽樣 1000 筆
# random_state=42 確保每次執行的抽樣結果都相同，方便重現實驗
print("正在從 AES 數據中隨機抽樣 20000 筆...")
aes_sample_df = aes_df
print(f"抽樣後 AES 數據維度：{aes_sample_df.shape}")


# --- 步驟 4: 讀取並合併 RC4 數據 ---
print("\n--- 步驟 4: 處理 RC4 數據 ---")
print(f"正在讀取 RC4 數據...")
rc4_df1 = pd.read_csv(rc4_file_1_path)
#rc4_df2 = pd.read_csv(rc4_file_2_path)
print(f"一份 RC4 數據維度為: {rc4_df1.shape}")

# 將兩份 RC4 數據合併
print("正在合併兩份 RC4 數據...")
rc4_combined_df = pd.concat([rc4_df1], ignore_index=True)
print(f"合併後 RC4 數據維度：{rc4_combined_df.shape}")


# --- 步驟 5: 最終合併與隨機排序 ---
print("\n--- 步驟 5: 最終合併與排序 ---")
print("正在合併處理好的 AES 和 RC4 數據...")
final_df = pd.concat([aes_sample_df, rc4_combined_df], ignore_index=True)
print(f"合併後維度 (尚未排序): {final_df.shape}")

print("正在隨機排序 (Shuffling) 整個數據集...")
# 使用 .sample(frac=1) 可以有效地打亂整個 DataFrame 的順序
# .reset_index(drop=True) 重設索引
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("✅ 數據集已準備就緒！")


# --- 步驟 6: 驗證並儲存最終數據集 ---
print("\n--- 步驟 6: 驗證與儲存 ---")
print("===== 最終數據集驗證 =====")
print(f"整體維度: {final_df.shape}")
print("\n各類別樣本數：")

# 假設您的類別標籤欄位名稱是 'label'
try:
    print(final_df['label'].value_counts())
    print("\n✅ 驗證成功！數據集已平衡。")
except KeyError:
    print("\n❌ 驗證失敗：找不到名為 'label' 的欄位。請確認您的類別標籤欄位名稱。")

# 儲存檔案
print(f"\n正在將最終數據集儲存至您的 Google 雲端硬碟...")
final_df.to_csv(output_file_path, index=False)


# --- 最終確認 ---
print("\n==============================================================================")
print(f"🎉 處理完成！")
print(f"最終數據集已成功儲存至您的 Google 雲端硬碟路徑：\n{output_file_path}")
print("==============================================================================")