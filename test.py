import pandas as pd
import numpy as np

path = 'AES-CTR_20000'

Y_test = np.load(f'/home/a0919/Entropy_Analysis/data/{path}/y_test.npy')
print(Y_test)