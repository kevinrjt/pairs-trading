# -*- coding: utf-8 -*-
from cointegration import EGTest
from lstm import train_model, predict_next
from util import *
from volatility import predict_volatility

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def backtest(train_df, test_df, params, m=1.5, n=0.5, r=0.8, s=0.6):
    train_spread = train_df.close_y - params.close_x * train_df.close_x - params.const
    train = train_df.assign(spread=train_spread)
    test_spread = test_df.close_y - params.close_x * test_df.close_x - params.const
    test = test_df.assign(spread=test_spread)
    transactions = []
    v_start, d_start = None, None
    cost, buy, sign = None, None, None
    model = None
    model = train_model(train)
    for i, row in test.iterrows():
        sigma = predict_volatility(train.spread, method='garch')
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
                prob = predict_next(model, train)
                if d_start is None:
                    if prob > r or prob < 1 - r:
                        cost = row.close_y + np.abs(params.close_x) * row.close_x
                        sign = 1 if prob > r else -1
                        buy = row.spread
                        d_start = row.date
                elif (prob < s and sign > 0) or (prob > 1 - s and sign < 0):
                    rate = (row.spread - buy) * sign / cost
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
    money = 1
    total_days = 0
    rets = []
    for rate, start, end, _ in transactions:
        days = calculate_days(start, end)
        total_days += days
        money *= 1 + rate
        rets += [np.power(1 + rate, 1 / days) - 1] * days
    # print(rets)
    return money - 1, (money - 1) * 365 / total_days, np.mean(rets) / np.std(rets)

def main():
    # df = merge_prices(get_price('000008'), get_price('600446')) #GOOD
    df = merge_prices(get_price('000008'), get_price('600446'))
    train_df, test_df = split_by_date(df, '2015-12-31')

    pvalue, params = EGTest(train_df.close_x, train_df.close_y)
    print(pvalue)

    transactions = backtest(train_df, test_df, params)
    pprint(transactions)

    # transactions = [(0.080167086151783556, '2016-01-25', '2016-01-27', 'v'),
    #                 (0.068495327325024558, '2016-01-28', '2016-02-26', 'd'),
    #                 (-0.00056971374473218706, '2016-02-26', '2016-03-22', 'v'), 
    #                 (0.036164446581118947, '2016-03-28', '2016-04-05', 'v'), 
    #                 (0.15764461837606525, '2016-04-13', '2016-05-13', 'd'), 
    #                 (-0.0091411001116070336, '2016-07-05', '2016-07-14', 'd'), 
    #                 (-0.02382616826889639, '2016-07-29', '2016-08-11', 'd'), 
    #                 (0.0067729583143217404, '2016-08-23', '2016-09-12', 'd'), 
    #                 (0.019087847354597114, '2016-09-19', '2016-09-27', 'd'), 
    #                 (0.0039180094149781656, '2016-10-24', '2016-12-30', 'v')]
    print('%.4f, %.4f, %.4f' % eval_returns(transactions))

if __name__ == '__main__':
    pass
    # main()