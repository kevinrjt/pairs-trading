from arch import arch_model

def predict_volatility(x):
    am = arch_model(x, mean='Zero')
    res = am.fit(disp='off')
    return (am.forecast(res.params).variance).iloc[-1]**0.5

# def predict(arch_res):
#     omega = arch_res.params['omega']
#     alpha = arch_res.params['alpha[1]']
#     beta = arch_res.params['beta[1]']
#     cur_resid = arch_res.resid.iloc[-1]
#     cur_volatility = arch_res.conditional_volatility.iloc[-1]
#     return (omega + alpha * (cur_resid**2) + beta * (cur_volatility**2))**0.5

def test():
    pass

if __name__ == '__main__':
    test()