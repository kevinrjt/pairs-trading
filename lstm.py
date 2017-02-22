from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

def predict_prob():
    return 0.5

def gen_data(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing
    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos

def main():
    # since we are using stateful rnn tsteps can be set to 1
    tsteps = 1
    batch_size = 25
    epochs = 1
    # number of elements ahead that are used to make the prediction
    lahead = 1

    print('Generating Data')
    X = gen_data()
    print('Input shape:', X.shape)

    y = np.zeros((len(X), 1))
    for i in range(len(X) - lahead):
        y[i, 0] = np.mean(X[i + 1:i + lahead + 1])
    print('Output shape')
    print(y.shape)

    X_train = X[:40000]
    y_train = y[:40000]
    X_val = X[40000:45000]
    y_val = y[40000:45000]
    X_test = X[45000:]
    y_test = y[45000:]

    print('Creating Model')
    model = Sequential()
    model.add(LSTM(50,
                   batch_input_shape=(batch_size, tsteps, 1),
                   return_sequences=True,
                   stateful=True))
    model.add(LSTM(50,
                   return_sequences=False,
                   stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='rmsprop')

    print('Training')
    for i in range(epochs):
        print('Epoch', i, '/', epochs)
        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  verbose=0,
                  nb_epoch=1,
                  shuffle=False)
        model.reset_states()

    print('Predicting')
    y_val_pred = model.predict(X_val, batch_size=batch_size)

    print('Plotting Results')
    plt.subplot(2, 1, 1)
    plt.plot(y_val)
    plt.title('Expected')
    plt.subplot(2, 1, 2)
    plt.plot(y_val_pred)
    plt.title('Predicted')
    plt.show()

if __name__ == '__main__':
  main()