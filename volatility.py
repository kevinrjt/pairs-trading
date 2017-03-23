# -*- coding: utf-8 -*-
from util import *

from arch import arch_model
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题  

def predict_volatility(x, method='garch'):
    if method == 'garch':
        am = arch_model(x, mean='Zero')
        result = am.fit(disp='off')
        return (am.forecast(result.params).variance).iloc[-1, 0]**0.5
    return np.std(x)

def test():
    df = merge_prices(get_price('000008'), get_price('600446'))
    train_df, test_df = split_by_date(df, '2015-12-31')
    x, y = train_df.close_x, train_df.close_y
    modol = sm.OLS(y, sm.add_constant(x))
    result = modol.fit()
    residual = result.resid

    am = arch_model(x, mean='Zero')
    res = am.fit(disp='off')
    print(res.summary())
    
    dates = [mdates.datestr2num(d) for d in train_df.date]
    plt.xlabel('Date')
    plt.subplot(2, 1, 1)
    plt.plot(dates, residual)
    plt.title('价差')
    plt.subplot(2, 1, 2)
    plt.plot(dates, res.conditional_volatility)
    plt.title('波动率')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == '__main__':
    pass
    # test()