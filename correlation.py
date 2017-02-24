import logging
import os

import pandas as pd

from util import *

def load_data():
    codes = get_codes()
    data = {}
    for code in codes:
        price = get_price(code)
        if price is not None and len(price) >= 1202:
            data[code] = price
    return data

def main():
    logging.basicConfig(filename=CORRELATION_LOG, 
                        level=logging.DEBUG,
                        filemode='w',
                        format='%(asctime)s[%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')

    data = load_data()
    stocks_num = len(data)
    print(stocks_num)
    return

    logging.info('%s stocks prices loaded.' % stocks_num)
    total = stocks_num * (stocks_num - 1) / 2

    if not os.path.exists(CORRELATION_DATA):
        corrs = pd.DataFrame(columns=('code1', 'code2', 'corr'))
        corrs.to_csv(CORRELATION_DATA, index=False)
    else:
        corrs = pd.read_csv(CORRELATION_DATA)
    length = len(corrs.index)
    logging.info('%s stock pairs loaded.' % length)

    counter = 0
    start_index = 0
    addend = stocks_num - 1
    while counter + addend <= length:
        counter += addend
        addend -= 1
        start_index += 1

    codes = sorted(data.keys())
    for i in range(start_index, stocks_num):
        for j in range(i + 1, stocks_num):
            corr = cal_correlation(data[codes[i]], data[codes[j]])
            corrs.loc[counter] = (codes[i], codes[j], corr)
            counter += 1
        corrs.to_csv(CORRELATION_DATA, index=False)
        logging.info('Stock %s\t%.2f.' % (codes[i], 100 * counter / total))

if __name__ == '__main__':
    main()