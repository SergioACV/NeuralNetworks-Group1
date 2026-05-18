#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configurada correctamente: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("Advertencia: No se detectó GPU. El entrenamiento será más lento.")

# Load Dataset
df = pd.read_csv("./data/dataset_lstm.csv")

# Split dataset into training and validation
def normalize_and_split_study_period(df, train_days=750):
    df_seq = df
    unique_dates = sorted(df_seq['date'].unique())
    split_idx = min(train_days - 1, len(unique_dates) - 1)
    split_date = unique_dates[split_idx]
    
    train_df = df_seq[df_seq['date'] <= split_date].copy()
    test_df = df_seq[df_seq['date'] > split_date].copy()
    
    return_cols = [col for col in df_seq.columns if col.startswith('R_')]
    return train_df[return_cols + ["target"]], test_df[return_cols + ["target"]]

return_cols = [col for col in df.columns if col.startswith('R_')]
train_df, test_df = normalize_and_split_study_period(df)

X_train = train_df[return_cols]
y_train = train_df["target"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# Reshape data for LSTM (samples, timesteps, features)
X_tr = X_tr.values.reshape(-1, 240, 1)
X_val = X_val.values.reshape(-1, 240, 1)

# Build and train model
with tf.device('/GPU:0'):
    model = Sequential()
    model.add(LSTM(units=25, input_shape=(240, 1), dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(
        optimizer=RMSprop(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_tr,
        y_tr,
        epochs=1000,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        shuffle=True
    )

model.summary()

# Save model
os.makedirs('./models', exist_ok=True)
model.save('./models/lstm_model.h5')
print("Modelo guardado en ./models/lstm_model.h5")