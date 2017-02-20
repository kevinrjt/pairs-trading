from util import *
import logging
import os
import tushare as ts

START = '1990-01-01'
END = '2016-12-31'
# STOCK_BASICS = ts.get_stock_basics()
# STOCK_BASICS.ix[code]['timeToMarket']

def download_data():
    codes = get_codes()
    for i, code in enumerate(codes):
        try:
            data_file = get_data_file(code)
            if not os.path.exists(data_file):
                h_data = ts.get_h_data(code, start=START, end=END)
                h_data.to_csv(data_file)
            logging.info('Stock %s(%s) downloaded.' % (i + 1, code))
        except Exception as e:
            logging.error('Stock %s(%s) error: %s.' % (i + 1, code, repr(e)))

def main():
    logging.basicConfig(filename=DOWNLOAD_LOG, 
                        level=logging.DEBUG,
                        filemode='w',
                        format='%(asctime)s[%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    download_data()

if __name__ == '__main__':
    main()
