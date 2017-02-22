import pandas as pd

DOWNLOAD_LOG = './log/download.log'
CORRELATION_LOG = './log/correlation.log'
CORRELATION_DATA = './data/correlation.csv'

def get_codes():
    with open('./data/A_codes.txt') as codes:
        return [line.strip() for line in codes]

def get_data_file(code):
    return './data/A/%s.csv' % code

def get_price(code):
    return pd.read_csv(get_data_file(code)).sort_values(by='date')

def split_by_date(p, date):
    return [g for k,g in p.groupby(p["date"] > date)]

def merge_prices(p1, p2):
    return p1.merge(p2, left_on='date', right_on='date', how='inner') 

def cal_correlation(p1, p2):
    df = merge_prices(p1, p2)
    return df.close_x.corr(df.close_y)

def test():
    p1 = get_price('000661')
    p2 = get_price('000826')
    df = merge_prices(p1, p2)
    df1, df2 = split_by_date(df, '2016-12-29')
    print(len(df), len(df1), len(df2))

if __name__ == '__main__':
    test()