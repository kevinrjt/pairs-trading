# -*- coding: utf-8 -*-
from cointegration import EGTest
from lstm import train_model, predict_next
from util import *
from volatility import predict_volatility

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def backtest(train, test, params, window, model=None, long_term='garch', short_term='', m=1.5, n=0.5, r=0.8, s=0.6):
    transactions = []
    l_start, s_start = None, None
    cost, buy, sign = None, None, None
    if short_term != 'model':
        model = None
    for i, row in test.iterrows():
        sigma = predict_volatility(train.spread, method=long_term)
        k = row.spread / sigma
        if l_start is None:
            if np.abs(k) > m:
                if s_start is not None:
                    rate = (row.spread - buy) * sign / cost
                    transactions.append((rate, s_start, row.date, 's'))
                    s_start = None
                cost = row.close_y + np.abs(params.close_x) * row.close_x
                sign = 1 if k < 0 else -1
                buy = row.spread
                l_start = row.date
            else:
                prob = predict_next(model, train, window)
                if s_start is None:
                    if prob > r or prob < 1 - r:
                        cost = row.close_y + np.abs(params.close_x) * row.close_x
                        sign = 1 if prob > r else -1
                        buy = row.spread
                        s_start = row.date
                elif (prob < s and sign > 0) or (prob > 1 - s and sign < 0):
                    rate = (row.spread - buy) * sign / cost
                    transactions.append((rate, s_start, row.date, 's'))
                    s_start = None
        elif np.abs(k) < n:
            rate = (row.spread - buy) * sign / cost
            transactions.append((rate, l_start, row.date, 'l'))
            l_start = None
        train = train.append(row, ignore_index=True)
    end, sell = test.iloc[-1].date, test.iloc[-1].spread
    if l_start is not None:
        rate = (sell - buy) * sign / cost
        transactions.append((rate, l_start, end, 'l'))
    elif s_start is not None:
        rate = (sell - buy) * sign / cost
        transactions.append((rate, s_start, end, 's'))
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
    totol = money - 1
    annual = 0 if total_days == 0 else (money - 1) * 365 / total_days
    sharp = np.mean(rets) / np.sqrt(np.var(rets) + 1e-8)
    return totol, annual, sharp

def trade(code1, code2, window=100):
    df = merge_prices(get_price(code1), get_price(code2))
    train_df, test_df = split_by_date(df, '2015-12-31')
    pvalue, params = EGTest(train_df.close_x, train_df.close_y)
    if pvalue > 0.01:
        return
    train_spread = train_df.close_y - params.close_x * train_df.close_x - params.const
    test_spread = test_df.close_y - params.close_x * test_df.close_x - params.const
    train = train_df.assign(spread=train_spread)
    test = test_df.assign(spread=test_spread)

    result = []
    pair = code1 + '&' + code2
    strategies = [('history', ''), ('garch', ''), ('history', 'model'), ('garch', 'model')]
    model, accuracy = train_model(train, window, test)
    for long_term, short_term in strategies:
        strategy = long_term + '+' + short_term
        transactions = backtest(train, test, params, window, model, long_term, short_term)
        totol, annual, sharp = eval_returns(transactions)
        result.append([pair, strategy, accuracy, transactions, totol, annual, sharp])
    return result

def test():
    transactions = [(0.080167086151783556, '2016-01-25', '2016-01-27', 'l'),
                    (0.068495327325024558, '2016-01-28', '2016-02-26', 's'),
                    (-0.00056971374473218706, '2016-02-26', '2016-03-22', 'l'), 
                    (0.036164446581118947, '2016-03-28', '2016-04-05', 'l'), 
                    (0.15764461837606525, '2016-04-13', '2016-05-13', 's'), 
                    (-0.0091411001116070336, '2016-07-05', '2016-07-14', 's'), 
                    (-0.02382616826889639, '2016-07-29', '2016-08-11', 's'), 
                    (0.0067729583143217404, '2016-08-23', '2016-09-12', 's'), 
                    (0.019087847354597114, '2016-09-19', '2016-09-27', 's'), 
                    (0.0039180094149781656, '2016-10-24', '2016-12-30', 'l')]
    print('%.4f, %.4f, %.4f' % eval_returns(transactions))

def main():
    results = []
    columns = ['pair','strategy','accuracy', 'transactions', 'totol', 'annual', 'sharp']
    counter = 0
    pairs = get_pairs()
    for code1, code2 in pairs:
        print(code1, code2)
        try:
            result = trade(code1, code2)
            if result is not None:
                results += result
                counter += 1
                if counter % 1 == 0:
                    pd.DataFrame(results, columns=columns).to_csv('results.csv', index=False)
        except Exception:
            pass
    pd.DataFrame(results, columns=columns).to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()