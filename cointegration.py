# -*- coding: utf-8 -*-
from util import *

from arch.unitroot import ADF
import numpy as np
import statsmodels.api as sm

def EGTest(x, y):
    modol = sm.OLS(y, sm.add_constant(x))
    result = modol.fit()
    residual = result.resid
    adf = ADF(residual)
    return adf.pvalue, result.params

def test():
    df = merge_prices(get_price('000008'), get_price('600446'))
    train_df, test_df = split_by_date(df, '2015-12-31')

    x, y = train_df.close_x, train_df.close_y
    # print(ADF(x).summary())
    # print(ADF(y).summary())
    # print(ADF(np.diff(x)).summary())
    # print(ADF(np.diff(y)).summary())
    modol = sm.OLS(y, sm.add_constant(x))
    result = modol.fit()
    # print(result.summary())
    residual = result.resid
    adf = ADF(residual)
    print(adf.summary())

if __name__ == '__main__':
    test()