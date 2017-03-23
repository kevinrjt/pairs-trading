# -*- coding: utf-8 -*-
from datetime import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOWNLOAD_LOG = './log/download.log'
CORRELATION_LOG = './log/correlation.log'
CORRELATION_DATA = './data/correlation.csv'

def calculate_days(start, end):
    start_date = datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.strptime(end,'%Y-%m-%d')
    return (end_date - start_date).days

def get_codes():
    with open('./data/A_codes.txt') as codes:
        return [line.strip() for line in codes]

def get_data_file(code):
    return './data/A/%s.csv' % code

def get_price(code):
    data_file = get_data_file(code)
    if os.path.exists(data_file):
        return pd.read_csv(data_file).sort_values(by='date')
    return None

def split_by_date(p, date):
    return [g for k,g in p.groupby(p["date"] > date)]

def merge_prices(p1, p2):
    return p1.merge(p2, left_on='date', right_on='date', how='inner') 

def cal_correlation(p1, p2):
    df = merge_prices(p1, p2)
    return df.close_x.corr(df.close_y)

def plot_series(s, xlabel, ylabel):
    dates = [mdates.datestr2num(d) for d in s.date]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gcf().autofmt_xdate()
    plt.plot(dates, s)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_prices(code1, code2):
    p1 = get_price(code1)
    p2 = get_price(code2)
    ps = merge_prices(p1, p2)
    ps = split_by_date(ps, '2015-12-31')[0]
    x = ps.close_x
    y = ps.close_y
    dates = [mdates.datestr2num(d) for d in ps.date]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gcf().autofmt_xdate()
    plt.plot(dates, x)
    plt.plot(dates, y)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def test():
    plot_prices('600033', '601188')
    p1 = get_price('000661')
    p2 = get_price('000826')
    df = merge_prices(p1, p2)
    df1, df2 = split_by_date(df, '2015-12-31')
    print(len(df), len(df1), len(df2))

if __name__ == '__main__':
    test()