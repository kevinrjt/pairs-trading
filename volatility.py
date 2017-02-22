from arch import arch_model
import numpy as np

def predict_volatility(x, method='garch'):
    if method == 'garch':
        am = arch_model(x, mean='Zero')
        result = am.fit(disp='off')
        return (am.forecast(result.params).variance).iloc[-1, 0]**0.5
    return np.std(x)