from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cointegration import EGTest
from lstm import predict_prob
from util import *
from volatility import predict_volatility

def backtest(train_df, test_df, params, init_money=1, m=1.5, n=0.5, r=0.7, s=0.6):
    train_spread = train_df.close_y - params.close_x * train_df.close_x - params.const
    train = train_df.assign(spread=train_spread)
    test_spread = test_df.close_y - params.close_x * test_df.close_x - params.const
    test = test_df.assign(spread=test_spread)
    money = init_money
    transactions = []
    v_start, d_start = None, None
    cost, buy, sign = None, None, None
    for i, row in test.iterrows():
        sigma = predict_volatility(train.spread, method='garch0')
        k = row.spread / sigma
        if v_start is None:
            if np.abs(k) > m:
                if d_start is not None:
                    rate = (row.spread - buy) * sign / cost
                    transactions.append((rate, d_start, row.date, 'd'))
                    d_start = None
                cost = row.close_y + np.abs(params.close_x) * row.close_x
                sign = 1 if k < 0 else -1
                buy = row.spread
                v_start = row.date
            else:
                prob = predict_prob()
                if d_start is None:
                    if prob > r or prob < 1 - r:
                        cost = row.close_y + np.abs(params.close_x) * row.close_x
                        sign = 1 if prob > r else -1
                        buy = row.spread
                        d_start = row.date
                elif (prob < 1 - s and sign > 0) or (prob > s and sign < 0):
                    profit = (row.spread - buy) * sign
                    transactions.append((rate, d_start, row.date, 'd'))
                    d_start = None
        elif np.abs(k) < n:
            rate = (row.spread - buy) * sign / cost
            transactions.append((rate, v_start, row.date, 'v'))
            v_start = None
        train = train.append(row, ignore_index=True)
    end, sell = test.iloc[-1].date, test.iloc[-1].spread
    if v_start is not None:
        rate = (sell - buy) * sign / cost
        transactions.append((rate, v_start, end, 'v'))
    elif d_start is not None:
        rate = (sell - buy) * sign / cost
        transactions.append((rate, d_start, end, 'd'))
    return transactions

def eval_returns(transactions):
    returns = []
    for rate, start, end, _ in transactions:
        days = calculate_days(start, end)
        returns.append(np.log(1 + rate) / days)
    print(returns)
    return np.mean(returns) / np.std(returns)

def main():
    df = merge_prices(get_price('000661'), get_price('000826'))
    train_df, test_df = split_by_date(df, '2015-12-31')

    pvalue, params = EGTest(train_df.close_x, train_df.close_y)
    print(pvalue)
    
    transactions = backtest(train_df, test_df, params)
    pprint(transactions)

    print('Average Daily return: %.2f' % eval_returns(transactions))

if __name__ == '__main__':
    main()