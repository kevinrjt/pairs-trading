from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cointegration import EGTest
from lstm import predict_prob
from util import *
from volatility import predict_volatility

def backtest(train_df, test_df, params, init_money=1, m=1.5, n=0.5, s=0.9, t=0.8):
    train_spread = train_df.close_y - params.close_x * train_df.close_x - params.const
    test_spread = test_df.close_y - params.close_x * test_df.close_x - params.const
    train = train_df.assign(spread=train_spread)
    test = test_df.assign(spread=test_spread)
    money = init_money
    transactions = []
    v_start, v_buy, v_volume = None, None, None
    d_start, d_buy, d_volume = None, None, None
    for i, row in test.iterrows():
        cost = row.close_x + np.abs(params.close_x) * row.close_y
        sigma = predict_volatility(train.spread, method='garch0')
        k = row.spread / sigma
        if v_start is None:
            if np.abs(k) > m:
                if d_start is not None:
                    money += (row.spread - d_buy) * d_volume
                    transactions.append((d_buy, row.spread, d_volume, d_start, row.date, 'd'))
                    d_start = None
                sign = np.abs(k) / k
                v_buy = row.spread
                v_volume = -sign * money / cost
                v_start = row.date
            else:
                prob = predict_prob()
                if d_start is None:
                    if prob > s or prob < 1 - s:
                        sign = 1 if prob > s else -1
                        d_buy = row.spread
                        d_volume = sign * money / cost
                        d_start = row.date
                elif (prob < 1 - t and d_volume > 0) or (prob > t and d_volume < 0):
                    money += (row.spread - d_buy) * d_volume
                    transactions.append((d_buy, row.spread, d_volume, d_start, row.date, 'd'))
                    d_start = None
        elif np.abs(k) < n:
            money += (row.spread - v_buy) * v_volume
            transactions.append((v_buy, row.spread, v_volume, v_start, row.date, 'v'))
            v_start = None
        train = train.append(row, ignore_index=True)
    end, sell = test.iloc[-1].date, test.iloc[-1].spread
    if v_start is not None:
        money += (sell - v_buy) * v_volume
        transactions.append((v_buy, sell, v_volume, v_start, end, 'v'))
    elif d_start is not None:
        money += (sell - d_buy) * d_volume
        transactions.append((d_buy, sell, d_volume, d_start, end, 'd'))
    return transactions, (money - init_money) / init_money

def main():
    df = merge_prices(get_price('000661'), get_price('000826'))
    train_df, test_df = split_by_date(df, '2015-12-31')

    pvalue, params = EGTest(train_df.close_x, train_df.close_y)
    print(pvalue)
    
    transactions, return_rate = backtest(train_df, test_df, params)
    pprint(transactions)
    print('Annual return: %.2f%%' % (return_rate * 100))

if __name__ == '__main__':
    main()