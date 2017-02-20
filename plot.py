from util import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_prices(code1, code2):
    p1 = get_price(code1)
    p2 = get_price(code2)
    ps = merge_prices(p1, p2)
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
    plot_prices('000755', '600660')

if __name__ == '__main__':
    test()
