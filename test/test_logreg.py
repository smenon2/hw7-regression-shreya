"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import numpy as np


# (you will probably need to import more things here)

def test_prediction():
    # Loading data as in main.py
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.

    # Set the random seed to 0 so that this is the same every time.
    np.random.seed(0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
    num_feats = 6
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

    # We also want to just check if the function is working so I'm setting the weights to zero
    log_model.W = np.zeros(num_feats + 1).flatten()
    log_model.train_model(X_train, y_train, X_val, y_val)

    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    preds = log_model.make_prediction(X_val)[0:10]
    preds_check = np.array([0.5002052, 0.49996008, 0.50020019, 0.49984917, 0.50000198,
                            0.50005586, 0.49976388, 0.50029581, 0.49986729, 0.50018047])

    assert np.allclose(preds, preds_check)


def test_loss_function():
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.

    # Set the random seed to 0 so that this is the same every time.
    np.random.seed(0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
    num_feats = 6
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)

    # We also want to just check if the function is working so I'm setting the weights to zero
    log_model.W = np.zeros(num_feats + 1).flatten()
    log_model.train_model(X_train, y_train, X_val, y_val)

    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    val_loss = log_model.loss_function(y_val, log_model.make_prediction(X_val))

    assert val_loss == 0.003431961882810284


def test_gradient():
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    np.random.seed(0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
    num_feats = 6
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
    log_model.W = np.zeros(num_feats + 1).flatten()
    log_model.train_model(X_train, y_train, X_val, y_val)

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
    np.random.shuffle(shuffle_arr)
    X_train = shuffle_arr[:, :-1]
    y_train = shuffle_arr[:, -1].flatten()
    num_feats = 6
    log_model.W = np.random.randn(num_feats + 1).flatten()
    grad = log_model.calculate_gradient(y_train, X_train)

    assert np.allclose(grad, np.array([-0.33097326, -0.42723908, -0.48789866, -0, -0,
                                       0.11667581, 0.04846649]))


def test_training():
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    np.random.seed(0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.
    num_feats = 6
    log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
    log_model.W = np.zeros(num_feats + 1).flatten()
    log_model.train_model(X_train, y_train, X_val, y_val)

    assert np.allclose(log_model.W, np.array([3.76866092e-04, 3.90501165e-04, 3.79532496e-04, 0.00000000e+00,
                                              0.00000000e+00, -1.26317298e-05, -7.77714080e-07]))
