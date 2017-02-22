from arch.unitroot import ADF
import statsmodels.api as sm

def EGTest(x, y):
    modol = sm.OLS(y, sm.add_constant(x))
    result = modol.fit()
    residual = result.resid
    adf = ADF(residual)
    return adf.pvalue, result.params
