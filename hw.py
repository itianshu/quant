import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm


#------------------提取数据------------------------
p = r'C:\Users\DELL\Desktop\DATA.xlsx'
data = pd.read_excel(p)
data = np.array(data)
hs = data[0, 2:2433]
stocks = data[1:51, 2:2433]

days = stocks.shape[1]
n_1 = np.zeros(50)
#----------------------遍历50支股票days里的数据---------------------
for i in range(50):
    for j in range(days):
        if np.isnan(stocks[i, j]):
            stocks[i, j] = 0
            n_1[i] += 1

mu = np.zeros(50)
sumup = np.sumup(stocks, axis=1)
for i in range(50):
    mu[i] = sumup[i]/(days-n_1[i])
for i in range(50):
    for j in range(days):
        if stocks[i, j] == 0:
            stocks[i, j] = mu[i]

def mean_cov(stocks, k, duration):
    stock_past = stocks[0:50, k-duration:k]
    shift = np.zeros((50, duration))
    for i in range(50):
        for j in range(0, duration-1):
            shift[i, j] = stock_past[i, j+1] / stock_past[i, j] - 1
    mean = np.mean(shift, 1)*365
    cov = np.cov(shift)
    return mean, cov

def objective(w, stocks, i, duration):
    _, cov = mean_cov(stocks, i, duration)
    return w.T@cov@w
def constrain(w, stocks, i, duration):
    mean,cov = mean_cov(stocks, i, duration)
    return w.T@(mean-0.025)-0.08+0.025

def analyze(hs, stocks, duration, start, end, rf, mu):
    k = (end-start)//60
    l = end-start
    r = np.zeros(l-1)
    y = np.zeros(l-1)
    ww = np.zeros((k, 50))
    for j in range(k):
        mean, cov = mean_cov(stocks, start + j * 60, duration)
        w0 = np.zeros(50)
        w0[0] = (mu - rf) / (mean - rf)[0]
        w = (mu - rf) / ((mean - rf).T @ np.linalg.inv(cov) @ (mean - rf)) * np.linalg.inv(cov) @ (mean - rf)
        cons = dict(type='eq', fun=constrain, args=(stocks, start + j * 60, duration))
        result = minimize(objective, w0, args=(stocks, start + j * 60, duration), constraints=[cons])
        ww[j] = result.x

    hsshift = np.zeros(2430)
    for j in range(0, 2430):
        hsshift[j] = np.log(hs[j + 1]) - np.log(hs[j])
    for i in range(0, l - 1):
        y[i] = np.sum((hsshift[start + i - 240:start + i]))
    shift = np.zeros((50, 2430))
    for i in range(50):
        for j in range(0, 2430):
            shift[i, j] = np.log(stocks[i, j + 1]) - np.log(stocks[i, j])
    shift = shift.T
    for i in range(0, l - 1):
        for j in range(k):
            if start + j * 60 <= i + start < start + (j + 1) * 60:
                break
        r[i] = np.sum(shift[start + i - 240:start + i], axis=0) @ ww[j].T + (1 - ww[j].sum()) * 0.025
    x = np.arange(start, end - 1)
    plt.plot(x, y, label='HS300')
    plt.plot(x, r, label='50 stocks')
    plt.xlabel('time')
    plt.ylabel('log-return')
    plt.title('log-return of hs300 & 50 stocks from 2015-2019')
    plt.legend()
    plt.show()


analyze(hs, stocks, 1200, 1213, 2433, 0.025, 0.08)

