# -*- coding: utf-8 -*-
from cointegration import EGTest
from util import *

from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random

def gen_data(code1, code2):
    df = merge_prices(get_price(code1), get_price(code2))
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

def train_lstm(X_train, y_train, window, epochs=50):
    model = Sequential()
    model.add(LSTM(32, input_dim=X_train.shape[2], return_sequences=True))
    model.add(LSTM(32)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    history = model.fit(X_train, y_train, batch_size=100, verbose=0, nb_epoch=epochs)
    print('Trainging done:', model.evaluate(X_train, y_train, verbose=0))
    # print(history.history.keys())
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()
    return model

def predict_next(model, train, window):
    if model is None:
        return 0.5
        # return np.random.uniform()
    X_train, y_train, X_nest = preprocess(train, window)
    return model.predict(X_nest, verbose=0)[0, 0]

def train_model(train, window, test=None):
    X_train, y_train, _ = preprocess(train, window)
    model = train_lstm(X_train, y_train, window)
    if test is None:
        return model

    X_test, y_test, _ = preprocess(test, window)
    y_true = np.diff(np.insert(test.spread.values, 0, train.spread.values[-1])) > 0
    y_pred = np.zeros(len(y_true))
    index = 0
    for i, row in test.iterrows():
        y_pred[index] = predict_next(model, train, window)
        train = train.append(row, ignore_index=True)
        index += 1
    # print(y_pred)
    return model, np.mean(y_true == (y_pred > 0.5))

def main():
    train, test = gen_data('000008', '600446')
    _, accuracy = train_model(train, 10, test)
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    # pass
    main()