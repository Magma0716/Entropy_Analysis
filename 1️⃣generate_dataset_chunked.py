import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from multiprocessing import Pool, cpu_count
from Crypto.Cipher import AES, ARC4
from Crypto.Random import get_random_bytes
import sys
import time

# =================================================================================
# ─── 1. 參數與分段設定 (主要修改區域) ──────────────────────────────────────────
# =================================================================================

# https://github.com/dj-on-github/sp800_22_tests

# --- 輸出檔案 ---
OUTPUT_CSV_PATH = "AES-CTR_10000.csv"

# --- 演算法選擇 ---
# 在列表中放入您想運行的演算法名稱，可以單選或多選。
# 可選項目: AES-CTR, RC4
ALGORITHMS_TO_RUN = ["AES-CTR"]

# --- 整體樣本數設定 ---
# 這是您「最終」希望每個演算法擁有的總樣本數。
TOTAL_SAMPLES_PER_ALG = 10000

# --- 分段生成設定 (使用 1-based 索引，更直觀) ---
# 設定您「這一次」要生成從第幾筆到第幾筆的數據。
# 例如，第一次運行可設為 START_INDEX = 1, END_INDEX = 1000
# 第二次運行可設為 START_INDEX = 1001, END_INDEX = 2000
# ...以此類推
START_INDEX = 1
END_INDEX = 10000 # 為了示範，先設一個較小的值

# --- 其他參數 ---
# B = 1048576
# tmp = int(MB * 1.5)
mill_bits_into_bytes = 1000000 / 8
REPO_DIR         = "./sp800_22_tests"
STREAM_LEN_BYTES = int(128)


# =================================================================================
# ─── 2. 核心功能函式 (通常無需修改) ──────────────────────────────────────────────
# =================================================================================

def get_sp80022_pvalues(ks_bytes: bytes) -> dict:

    if not os.path.exists(REPO_DIR) or not os.path.exists(os.path.join(REPO_DIR, "sp800_22_tests.py")):
        print(f"錯誤：找不到 SP800-22 測試庫於 '{REPO_DIR}'。請檢查路徑。", file=sys.stderr)
        return {}
    
    with tempfile.NamedTemporaryFile(dir=REPO_DIR, suffix=".bin", delete=False) as f:
        f.write(ks_bytes)
        fname = os.path.basename(f.name)
        
    try:
        proc = subprocess.run(
            ["python", "sp800_22_tests.py", fname],
            cwd=REPO_DIR,
            capture_output=True, text=True, check=False
        )
    finally:
        os.remove(os.path.join(REPO_DIR, fname))
    
    pvals, cur = {}, None
    
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("TEST:"):
            cur = line.split()[1]
        elif line.startswith("P=") and cur:
            pvals[cur] = float(line.split("=",1)[1])
            cur = None
            
    return pvals

# 生成單一樣本，並無限重試直到獲得一個包含所有 p-value 的有效結果。
def process_sample(alg: str, test_names: list):
    while True:
        try:
            key = get_random_bytes(16) # RC4 的 seed (種子) 就是這個 key
            
            if alg == "AES-CTR":
                nonce = get_random_bytes(8)
                ks = AES.new(key, AES.MODE_CTR, nonce=nonce).encrypt(b"\x00"*STREAM_LEN_BYTES)
            elif alg == "RC4":
                ks = ARC4.new(key).encrypt(b"\x00"*STREAM_LEN_BYTES)
            else:
                # 如果未來加入新演算法，這裡可以報錯
                raise ValueError(f"未知的演算法: {alg}")

            pvals = get_sp80022_pvalues(ks)
            
            # 關鍵檢查：確保所有 p-value 都已生成
            pvals_list = [pvals[name] for name in test_names]

            # 將演算法名稱也加入到結果中，方便後續分析
            return list(ks), pvals_list, alg

        except KeyError:
            # 靜默重試不完整的樣本
            time.sleep(0.1) # 避免因過於頻繁的失敗而造成 CPU 空轉
            continue
        
        except Exception as e:
            print(f"在處理樣本時發生意外錯誤: {e}", file=sys.stderr)
            time.sleep(1)
            continue

# =================================================================================
# ─── 3. 主執行流程 ───────────────────────────────────────────────────────────────
# =================================================================================

if __name__ == "__main__":
    
    # 參數驗證
    if START_INDEX <= 0 or END_INDEX < START_INDEX or END_INDEX > TOTAL_SAMPLES_PER_ALG:
        raise ValueError(f"索引設定錯誤！請確保 1 <= START_INDEX <= END_INDEX <= {TOTAL_SAMPLES_PER_ALG}")

    num_samples_this_run = END_INDEX - START_INDEX + 1
    print(f"✅ 設定驗證成功")
    print(f"   - 目標演算法: {ALGORITHMS_TO_RUN}")
    print(f"   - 本次運行區間: 樣本 {START_INDEX} 到 {END_INDEX}")
    print(f"   - 每個演算法將生成 {num_samples_this_run} 筆數據")
    print(f"   - 總計生成 {num_samples_this_run * len(ALGORITHMS_TO_RUN)} 筆數據")
    print(f"   - 輸出檔案: {OUTPUT_CSV_PATH}\n")

    # --- 步驟 3.2: 獲取 SP800-22 測試名稱 ---
    print("正在確定 SP800-22 測試列表...")
    TEST_NAMES = []
    while True:
        _dummy_ks = get_random_bytes(STREAM_LEN_BYTES)
        _pvals = get_sp80022_pvalues(_dummy_ks)
        if len(_pvals) >= 15: # 假設至少有15個標準測試
            TEST_NAMES = sorted(_pvals.keys())
            print("   - 偵測到的測試項目:", TEST_NAMES)
            break
        print("   - 獲取完整測試列表失敗，正在重試...")
        time.sleep(1)

    # --- 步驟 3.3: 準備任務列表 ---
    # 根據選擇的演算法和分段區間，生成任務列表
    tasks = []
    for alg in ALGORITHMS_TO_RUN:
        tasks.extend([alg] * num_samples_this_run)

    # --- 步驟 3.4: 並行生成數據 ---
    print("\n🚀 數據生成已啟動...")
    # 加上一個 'label' 欄位來儲存演算法名稱
    columns = ['label'] + [f"byte_{i}" for i in range(STREAM_LEN_BYTES)] + TEST_NAMES

    results = []
    # 使用 functools.partial 來傳遞固定的 test_names 參數
    from functools import partial

    with Pool(min(12, cpu_count())) as pool:
        # 將固定的 TEST_NAMES 參數傳遞給 process_sample
        process_func = partial(process_sample, test_names=TEST_NAMES)

        for ks_bytes, y, alg_label in tqdm(pool.imap_unordered(process_func, tasks),
                                    total=len(tasks),
                                    desc=f"生成 {START_INDEX}-{END_INDEX} 批次數據"):
            # 將演算法標籤放在第一位
            results.append([alg_label] + ks_bytes + y)

    # --- 步驟 3.5: 儲存或附加到 CSV ---
    print("\n💾 正在儲存數據...")
    df = pd.DataFrame(results, columns=columns)

    # 關鍵：檢查檔案是否存在，以決定是否要寫入標頭
    file_exists = os.path.exists(OUTPUT_CSV_PATH)

    if not file_exists:
        print(f"   - 偵測到檔案 '{OUTPUT_CSV_PATH}' 不存在，將創建新檔案並寫入標頭。")
        df.to_csv(OUTPUT_CSV_PATH, mode='w', index=False, header=True)
    else:
        print(f"   - 偵測到檔案 '{OUTPUT_CSV_PATH}' 已存在，將在檔案末尾附加新數據。")
        df.to_csv(OUTPUT_CSV_PATH, mode='a', index=False, header=False)

    print(f"\n🎉 成功！ {len(results)} 筆新數據已儲存至 {OUTPUT_CSV_PATH}")
    print(f"   - 新增數據維度: {df.shape}")

    # 提示：讀取整個檔案來查看總大小
    try:
        total_df = pd.read_csv(OUTPUT_CSV_PATH)
        print(f"   - 目前檔案總維度: {total_df.shape}")
    except Exception as e:
        print(f"   - 無法讀取最終檔案以確認維度: {e}")