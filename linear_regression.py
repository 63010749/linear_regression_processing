import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import model_selection
from sklearn import linear_model
from IPython.display import display

weather_data = pd.read_csv('North.csv',parse_dates=['datetime']
                     , infer_datetime_format=True)

temp_t = weather_data["T_mu"]
temp_df = weather_data[["datetime","T_mu"]]
temp_df.head(10)

mask = (temp_df['datetime'] >= '15/02/2016') & (temp_df['datetime'] <= '15/05/2016')
temp_df = temp_df.loc[mask]
temp_df.set_index("datetime", inplace=True)

y = [e for e in temp_df.T_mu]
x = list(range(1,len(y)+1))   #[f for f in temp_df.datetime]
df = pd.DataFrame(
     {'x': x,
      'y': y} )

xmean = np.mean(x)
ymean = np.mean(y)

df['xycov'] = (df['x'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['x'] - xmean)**2

beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
print(f'alpha = {alpha:.3f}')
print(f'beta = {beta:.3f}')
print(f'y(predict) = {beta:.3f}x + {alpha:.3f}')

ypred = [alpha + beta*e for e in x]

corr_matrix = np.corrcoef(y, ypred)
corr = corr_matrix[0,1]
R_sq = corr**2
print(f'R-square = {R_sq:.3f}')
MSE = np.square(np.subtract(y,ypred)).mean()
print(f'MSE = {MSE:.3f}')

yText=min(y)

fig = plt.figure(figsize=(12, 6))

plt.plot(x, y)# scatter plot showing actual data
plt.plot(x, ypred)# regression line   
plt.title('North 2021')
plt.xlabel('days')
plt.ylabel('temperature(C)')
plt.annotate(f'y(predict) = {beta:.3f}x + {alpha:.3f}',(270,yText+2))
plt.annotate(f'R-square = {R_sq:.3f}', (270, yText+1))
plt.annotate(f'MSE = {MSE:.3f}',(270,yText))
plt.show()

'''
#### Covariance ####

mask1 = (temp_df['datetime'] >= '15/02/2016') & (temp_df['datetime'] <= '15/05/2016')
mask2 = (temp_df['datetime'] >= '15/02/2017') & (temp_df['datetime'] <= '15/05/2017')
mask3 = (temp_df['datetime'] >= '15/02/2018') & (temp_df['datetime'] <= '15/05/2018')
mask4 = (temp_df['datetime'] >= '15/02/2019') & (temp_df['datetime'] <= '15/05/2019')
mask5 = (temp_df['datetime'] >= '15/02/2020') & (temp_df['datetime'] <= '15/05/2020')

temp_df1 = temp_df.loc[mask1]
temp_df2 = temp_df.loc[mask2]
temp_df3 = temp_df.loc[mask3]
temp_df4 = temp_df.loc[mask4]
temp_df5 = temp_df.loc[mask5]

temp_df1.set_index("datetime", inplace=True)
temp_df2.set_index("datetime", inplace=True)
temp_df3.set_index("datetime", inplace=True)
temp_df4.set_index("datetime", inplace=True)
temp_df5.set_index("datetime", inplace=True)

y1 = [e for e in temp_df1.T_mu]
y2 = [e for e in temp_df2.T_mu]
y3 = [e for e in temp_df3.T_mu]
y4 = [e for e in temp_df4.T_mu]
y5 = [e for e in temp_df5.T_mu]
# print(f'{y1}\n\n{y2}\n\n{y3}\n\n{y4}\n\n{y5}\n')
print(f'2016 : {len(y1)}\n2017 : {len(y2)}\n2018 : {len(y3)}\n2019 : {len(y4)}\n2020 : {len(y5)}\n')

data = {'2016': y1,
        '2017': y2,
        '2018': y3,
        '2019': y4,
        '2020': y5
        }
df = pd.DataFrame(data,columns=['2016','2017','2018','2019','2020'])
covMatrix = pd.DataFrame.cov(df)
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()'''