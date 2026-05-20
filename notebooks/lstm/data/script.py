# ============================================================
# BAYESIAN-OPTIMIZED LSTM FOR STOCK PRICE PREDICTION
# Based on:
# "Bayesian-Optimized LSTM Framework for Accurate Stock Price Prediction"
# ============================================================

# ============================================================
# INSTALL (RUN ONLY ONCE)
# ============================================================
# pip install pandas numpy scikit-learn matplotlib tensorflow bayesian-optimization

# ============================================================
# IMPORTS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from bayes_opt import BayesianOptimization

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "../../../data/processed/dataset_with_returns.csv"   # CHANGE THIS

WINDOW_SIZE = 30

EPOCHS_BO = 10
EPOCHS_FINAL = 50

TARGET = "adjusted_price"

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d"
]

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...\n")

df = pd.read_csv(CSV_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["Name", "date"])

print(df.head())

# ============================================================
# OPTIONAL:
# TRAIN ONLY ONE STOCK
# ============================================================

# Example:
# df = df[df["Name"] == "AAPL"]

# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna()

# ============================================================
# SCALE FEATURES
# ============================================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df[FEATURES] = feature_scaler.fit_transform(df[FEATURES])

df[[TARGET]] = target_scaler.fit_transform(df[[TARGET]])

# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(dataframe, features, target, window):

    X = []
    y = []

    values = dataframe[features + [target]].values

    for i in range(window, len(values)):

        X.append(values[i-window:i, :-1])
        y.append(values[i, -1])

    return np.array(X), np.array(y)

X, y = create_sequences(
    df,
    FEATURES,
    TARGET,
    WINDOW_SIZE
)

print("\nSequence shapes:")
print("X:", X.shape)
print("y:", y.shape)

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print("\nDataset split:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ============================================================
# MODEL FUNCTION
# ============================================================

def build_model(
    units=64,
    dropout=0.2,
    learning_rate=0.001
):

    model = Sequential()

    model.add(
        LSTM(
            units=int(units),
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        )
    )

    model.add(Dropout(dropout))

    model.add(Dense(1))

    optimizer = Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model

# ============================================================
# BAYESIAN OPTIMIZATION OBJECTIVE FUNCTION
# ============================================================

def objective(
    units,
    dropout,
    learning_rate,
    batch_size
):

    units = int(units)
    batch_size = int(batch_size)

    model = build_model(
        units=units,
        dropout=dropout,
        learning_rate=learning_rate
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_BO,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    predictions = model.predict(
        X_val,
        verbose=0
    )

    mse = mean_squared_error(
        y_val,
        predictions
    )

    return -mse

# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================

print("\nStarting Bayesian Optimization...\n")

pbounds = {
    "units": (32, 256),
    "dropout": (0.1, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "batch_size": (16, 128)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=SEED,
    verbose=2
)

optimizer.maximize(
    init_points=5,
    n_iter=10
)

# ============================================================
# BEST PARAMETERS
# ============================================================

best_params = optimizer.max["params"]

best_units = int(best_params["units"])
best_dropout = best_params["dropout"]
best_learning_rate = best_params["learning_rate"]
best_batch_size = int(best_params["batch_size"])

print("\nBest Parameters:")
print(best_params)

# ============================================================
# BUILD FINAL MODEL
# ============================================================

print("\nTraining final model...\n")

final_model = build_model(
    units=best_units,
    dropout=best_dropout,
    learning_rate=best_learning_rate
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

history = final_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FINAL,
    batch_size=best_batch_size,
    verbose=1,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...\n")

predictions = final_model.predict(
    X_test,
    verbose=0
)

# ============================================================
# INVERSE SCALE
# ============================================================

predictions_rescaled = target_scaler.inverse_transform(
    predictions
)

y_test_rescaled = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# ============================================================
# METRICS
# ============================================================

mape = mean_absolute_percentage_error(
    y_test_rescaled,
    predictions_rescaled
)

mse = mean_squared_error(
    y_test_rescaled,
    predictions_rescaled
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test_rescaled,
    predictions_rescaled
)

r2 = r2_score(
    y_test_rescaled,
    predictions_rescaled
)

# ============================================================
# RESULTS
# ============================================================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(f"MAPE : {mape:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")

# ============================================================
# PLOT TRAINING HISTORY
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training History")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()

# ============================================================
# PLOT PREDICTIONS
# ============================================================

plt.figure(figsize=(16, 7))

plt.plot(
    y_test_rescaled,
    label="Real Price"
)

plt.plot(
    predictions_rescaled,
    label="Predicted Price"
)

plt.title("Real vs Predicted Stock Prices")

plt.xlabel("Time")

plt.ylabel("Price")

plt.legend()

plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

final_model.save("bayesian_lstm_stock_model.h5")

print("\nModel saved as:")
print("bayesian_lstm_stock_model.h5")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = pd.DataFrame({
    "Real": y_test_rescaled.flatten(),
    "Predicted": predictions_rescaled.flatten()
})

results_df.to_csv(
    "predictions.csv",
    index=False
)

print("\nPredictions saved as:")
print("predictions.csv")

# ============================================================
# OPTIONAL:
# NEXT DAY PREDICTION
# ============================================================

last_window = X[-1]

last_window = np.expand_dims(
    last_window,
    axis=0
)

next_prediction = final_model.predict(
    last_window,
    verbose=0
)

next_prediction = target_scaler.inverse_transform(
    next_prediction
)

print("\nNext Predicted Price:")
print(next_prediction[0][0])

# ============================================================
# END
# ============================================================
