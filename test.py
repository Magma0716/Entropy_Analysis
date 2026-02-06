import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.metrics import r2_score

results_df = pd.read_csv(r'./data/AES-CTR_2000/r2_results.csv')

good = results_df[results_df['R2'] > 0.5].sort_values(by='R2', ascending=False)
bad = results_df[results_df['R2'] <= 0.5].sort_values(by='R2', ascending=False)


print(Fore.GREEN + '\n高相關性:' + Style.RESET_ALL)
print(good.to_string(index=False))

print(Fore.GREEN + '\n低相關性:' + Style.RESET_ALL)
print(bad.to_string(index=False))

y_test = np.load(r'./data/AES-CTR_2000/y_test.npy')
y_pred = np.load(r'./data/AES-CTR_2000/y_pred.npy')

r2 = r2_score(y_test, y_pred)
print(Fore.RED + '總體 R2:' + Style.RESET_ALL)
print(f'{r2:.4f}')