import itertools
import pandas
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pylab import rcParams

data = pandas.read_csv('train.csv')

print(data.head())

extracted_ATM = data[data['ATM_ID'] == 3].copy()

extracted_ATM.loc[:, 'DATE'] = pandas.to_datetime(extracted_ATM['DATE'])
print(extracted_ATM['DATE'].max() - extracted_ATM['DATE'].min())

extracted_ATM = extracted_ATM.set_index('DATE')
y = extracted_ATM['CLIENT_OUT'].resample('Q').mean()

print(y)

# plt.plot(y)
# plt.show()

# rcParams['figure.figsize'] = 18, 8
# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.show()

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]

order = 0
seasonal_order = 0

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}4 - AIC:{}'.format(param, param_seasonal, results.aic))
            order = param
            seasonal_order = param_seasonal
        except:
            continue

print(order)
print(seasonal_order)

mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

pred = results.get_prediction(start=pandas.to_datetime('2017-03-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2015':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('CLIENT_OUT')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2017-03-31':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(numpy.sqrt(mse), 2)))
#
# pred_uc = results.get_forecast(steps=2)
# pred_ci = pred_uc.conf_int()
# ax = y.plot(label='observed', figsize=(14, 7))
# pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color='k', alpha=.25)
# ax.set_xlabel('Date')
# ax.set_ylabel('CLIENT_OUT')
# plt.legend()
# plt.show()
