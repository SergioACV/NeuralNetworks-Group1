import os
import random
from pathlib import Path

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import (
    LecunUniform,
    Orthogonal,
    Zeros
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'financial_dataset.npz'
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'attention_lstm_classifier.keras'
SEED = 42


os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def load_data(data_path):
    data = np.load(data_path)

    X = data['X']
    y_classification = data['y_classification']
    sequence_dates = data['sequence_dates']

    return X, y_classification, sequence_dates


def split_data(X, y_classification, sequence_dates):
    train_mask = sequence_dates < np.datetime64('2017-01-01')

    val_mask = (
        (sequence_dates >= np.datetime64('2017-01-01')) &
        (sequence_dates < np.datetime64('2018-01-01'))
    )

    test_mask = sequence_dates >= np.datetime64('2018-01-01')

    return {
        'train': (
            X[train_mask],
            y_classification[train_mask]
        ),
        'validation': (
            X[val_mask],
            y_classification[val_mask]
        ),
        'test': (
            X[test_mask],
            y_classification[test_mask]
        )
    }


def print_split_info(split_name, X_split, y_split):
    print(f"\n====================\n{split_name.upper()}\n====================")
    print("X:", X_split.shape)
    print("y:", y_split.shape)


def build_attention_lstm_classifier(
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

    output = Dense(
        1,
        activation='sigmoid',
        name='classification_output'
    )(x)

    model = Model(
        inputs=inputs,
        outputs=output
    )

    return model


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val
):

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        callbacks=[
            early_stopping,
            checkpoint
        ],
        epochs=15,
        batch_size=256,
        shuffle=False,
        verbose=1
    )

    return history


def save_model(model, model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)


def main():

    X, y_classification, sequence_dates = load_data(DATA_PATH)

    splits = split_data(
        X,
        y_classification,
        sequence_dates
    )

    X_train, y_train = splits['train']
    X_val, y_val = splits['validation']
    X_test, y_test = splits['test']

    print_split_info(
        'train',
        X_train,
        y_train
    )

    print_split_info(
        'validation',
        X_val,
        y_val
    )

    print_split_info(
        'test',
        X_test,
        y_test
    )

    model = build_attention_lstm_classifier(
        seq_len=X.shape[1],
        n_features=X.shape[2]
    )

    compile_model(model)

    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    save_model(
        model,
        MODEL_PATH
    )


if __name__ == '__main__':
    main()