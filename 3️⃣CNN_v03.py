import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 參數
file_name = 'AES-CTR_10000'
byte = 1024
bit = byte * 8

# 讀取csv
df = pd.read_csv(f'{file_name}.csv')

# 輸⼊特徵(X)
cols = [f'byte_{i}' for i in range(byte)]
X_bytes = df[cols].values.astype(np.uint8)
X_bits = np.unpackbits(X_bytes, axis=1).astype('float32')
X = np.expand_dims(X_bits, axis=-1)

# ⽬標標籤(Y)
y = df.iloc[:, -15:].values.astype('float32')
test_names = df.columns[-15:].tolist()

print(f"輸入特徵維度 (X): {X.shape}")
print(f"輸出標籤維度 (y): {y.shape}")

# CNN模型
def residual_block(x, filters, kernel_size=3, l2_reg=1e-6):
    shortcut = x

    # layer 1
    x = layers.Conv1D(
        filters, kernel_size, padding='same', dilation_rate=2
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # layer 2
    x = layers.Conv1D(
        filters, kernel_size, padding='same',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # 輸入 != 輸出維度 -> 調整 shortcut
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, 1, padding='same',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(shortcut)
    
    # 輸入加回輸出
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
    
def CNN(input_shape=(bit, 1), output_dim=15, l2_reg=1e-6):
    inputs = layers.Input(shape=input_shape)
    
    # layer 1
    x = layers.Conv1D(
        64, kernel_size=31, strides=4, padding='same'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # layer 2
    for f in [64, 64, 128, 128, 256]:
        x = residual_block(x, f)
        x = layers.MaxPooling1D(pool_size=2)(x) # 降維
    
    # max-pool layer
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    # output layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x) # 防過擬合
    outputs = layers.Dense(output_dim, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    huber_loss = tf.keras.losses.Huber(delta=0.1)
    model.compile(optimizer=optimizer, loss=huber_loss, metrics=['mae'])
    return model

model = CNN(output_dim=y.shape[1])
model.summary()

# 測試集, 訓練集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## early stop
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 評估
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 1)

results = []
for i, name in enumerate(test_names):
    score = r2_score(y_test[:, i], y_pred[:, i])
    results.append({'Test': name, 'R2': score})
results_df = pd.DataFrame(results)

good = results_df[results_df['R2'] > 0.5].sort_values(by='R2', ascending=False)
bad = results_df[results_df['R2'] <= 0.5].sort_values(by='R2', ascending=False)

# 成果
print(Fore.GREEN + '\n高相關性:' + Style.RESET_ALL)
print(good.to_string(index=False))

print(Fore.GREEN + '\n低相關性:' + Style.RESET_ALL)
print(bad.to_string(index=False))

r2 = r2_score(y_test, y_pred)
print(Fore.RED + '總體 R2:' + Style.RESET_ALL)
print(f'{r2:.4f}')

# 儲存
history_df = pd.DataFrame(history.history)
history_df.to_csv('train_history.csv', index=False)

np.save('y_test.npy', y_test)
np.save('y_pred.npy', y_pred)

results_df.to_csv('r2_results.csv', index=False)

model.save(f'{file_name}.keras')

print('\n資料儲存成功')

'''
1. byte to bits
2. kernal + resnet
3. pool layer (0~1)
'''