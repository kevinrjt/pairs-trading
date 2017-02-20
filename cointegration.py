from arch.unitroot import ADF
from util import *
from volatility import *
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def EGTest(x, y):
    modol = sm.OLS(y, sm.add_constant(x))
    results = modol.fit()
    residuals = results.resid
    adf = ADF(residuals)
    return adf.pvalue, results.params, residuals

def main():
    # df = merge_prices(get_price('000001'), get_price('000008'))
    df = merge_prices(get_price('000661'), get_price('000826'))
    x = df.close_x
    y = df.close_y
    pvalue, params, residuals = EGTest(x, y)
    print(pvalue)
    print(params)
    # plt.plot(residuals)
    # plt.show()
    print(predict_volatility(residuals))
    
if __name__ == '__main__':
    main()