# -*- coding: utf-8 -*-
from cointegration import EGTest
from util import *

from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import random

def gen_data():
    df = merge_prices(get_price('000008'), get_price('600446'))
    train_df, test_df = split_by_date(df, '2015-12-31')
    pvalue, params = EGTest(train_df.close_x, train_df.close_y)
    train_spread = train_df.close_y - params.close_x * train_df.close_x - params.const
    train = train_df.assign(spread=train_spread)
    test_spread = test_df.close_y - params.close_x * test_df.close_x - params.const
    test = test_df.assign(spread=test_spread)
    return train, test

def preprocess(train, window):
    length = len(train) - window
    spread_array = np.copy(train.spread.values)
    train_array = train.drop('date', 1).drop('spread', 1).values
    X_train = np.zeros((length, window, train_array.shape[1]), dtype=np.float64)
    for i in range(length):
        X_train[i] = train_array[i:i+window]
    y_train = np.diff(spread_array[window-1:]) > 0
    return X_train, y_train, np.array([train_array[length:]])

def train_model(train, epochs=1000, window=30):
    X_train, y_train, _ = preprocess(train, window)
    model = Sequential()
    model.add(LSTM(32, input_dim=X_train.shape[2], return_sequences=False))
    # model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['binary_accuracy'])
    history = model.fit(X_train, y_train, batch_size=100, verbose=0, nb_epoch=epochs)
    print('Trainging done:', model.evaluate(X_train, y_train, verbose=0))

    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    return model

def predict_next(model, train, window=30):
    if model is None:
        # return 0.5
        return random.uniform(0, 1)
    X_train, y_train, X_test = preprocess(train, window)
    # model.fit(X_train, y_train, verbose=0, nb_epoch=1)
    return model.predict(X_test, verbose=0)[0, 0]

def eval_model(model, train, test):
    model = train_model(train)

def main():
    train, test = gen_data()
    X_train, y_train, _ = preprocess(train, 30)
    X_test, y_test, _ = preprocess(test, 30)
    
    model = train_model(train)

    y_true = np.diff(np.insert(test.spread.values, 0, train.spread.values[-1])) > 0
    y_pred = np.zeros(len(y_true))

    index = 0
    for i, row in test.iterrows():
        y_pred[index] = predict_next(model, train)
        # print(index, y_pred[index])
        # return
        train = train.append(row, ignore_index=True)
        index += 1

    print('Accuracy:', np.mean(y_true == (y_pred > 0.5)))
    # print('<0.8:', np.sum(y_pred > 0.8))
    # print('<0.2:', np.sum(y_pred < 0.2))

    # print('Plotting Results')
    # plt.subplot(2, 1, 1)
    # plt.plot(y_true)
    # plt.title('Expected')
    # plt.subplot(2, 1, 2)
    # plt.plot(y_pred)
    # plt.title('Predicted')
    # plt.show()

if __name__ == '__main__':
    pass
    # main()