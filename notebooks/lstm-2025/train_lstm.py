import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Attention,
    Add,
    LayerNormalization,
    PReLU
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import (
    LecunUniform,
    Orthogonal,
    Zeros
)

# =====================================================
# GPU CONFIG
# =====================================================

print("TensorFlow version:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

physical_gpus = tf.config.list_physical_devices('GPU')

for gpu in physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# =====================================================
# LOAD DATA
# =====================================================

DATA_PATH = './data/financial_dataset.npz'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"No se encontró el dataset en: {DATA_PATH}"
    )

print("\nCargando dataset...")

data = np.load(DATA_PATH)

X = data['X']
y = data['y']
sequence_dates = data['sequence_dates']

print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================================
# MODEL
# =====================================================


def build_attention_lstm_model(
    seq_len=20,
    n_features=5
):

    inputs = Input(
        shape=(seq_len, n_features)
    )

    x = LSTM(
        units=64,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.2,
        recurrent_dropout=0.2,
        kernel_regularizer=l2(1e-6),
        recurrent_regularizer=l2(1e-6),
        kernel_initializer=LecunUniform(),
        recurrent_initializer=Orthogonal(),
        bias_initializer=Zeros()
    )(inputs)

    attention_output = Attention()([x, x])

    x = Add()([x, attention_output])

    x = LayerNormalization()(x)

    x = LSTM(
        units=32,
        return_sequences=False,
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=0.2,
        kernel_regularizer=l2(1e-6),
        recurrent_regularizer=l2(1e-6),
        kernel_initializer=LecunUniform(),
        recurrent_initializer=Orthogonal(),
        bias_initializer=Zeros()
    )(x)

    x = Dense(16)(x)
    x = PReLU()(x)

    x = Dense(8)(x)
    x = PReLU()(x)

    outputs = Dense(
        1,
        activation='linear'
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    return model

# =====================================================
# BUILD MODEL
# =====================================================

SEQ_LEN = X.shape[1]
N_FEATURES = X.shape[2]

model = build_attention_lstm_model(
    seq_len=SEQ_LEN,
    n_features=N_FEATURES
)

model.summary()

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss='mse',
    metrics=['mae']
)

# =====================================================
# TRAIN / VAL / TEST SPLIT
# =====================================================

train_mask = sequence_dates < np.datetime64('2017-01-01')

val_mask = (
    (sequence_dates >= np.datetime64('2017-01-01')) &
    (sequence_dates < np.datetime64('2018-01-01'))
)

test_mask = sequence_dates >= np.datetime64('2018-01-01')

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\n====================")
print("TRAIN")
print("====================")
print(X_train.shape)
print(y_train.shape)

print("\n====================")
print("VALIDATION")
print("====================")
print(X_val.shape)
print(y_val.shape)

print("\n====================")
print("TEST")
print("====================")
print(X_test.shape)
print(y_test.shape)

# =====================================================
# TRAINING
# =====================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    shuffle=False,
    verbose=1
)

# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluando modelo...")

loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {loss:.6f}")
print(f"Test MAE : {mae:.6f}")

# =====================================================
# SAVE MODEL
# =====================================================

SAVE_PATH = './models/attention_lstm_model.keras'

model.save(SAVE_PATH)

print(f"\nModelo guardado en: {SAVE_PATH}")

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

predictions = model.predict(X_test[:10])

print("\nPredicciones ejemplo:")
print(predictions.flatten())

print("\nValores reales:")
print(y_test[:10].flatten())

# =====================================================
# REQUIREMENTS
# =====================================================

# pip install tensorflow numpy
