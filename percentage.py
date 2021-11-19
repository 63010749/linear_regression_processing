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

# Import the data
weather_data = pd.read_csv('Central_diff.csv',parse_dates=['datetime']
                     , infer_datetime_format=True)

temp_df = weather_data[["datetime","temp_diff"]]
temp_df.head(10)

mask = (temp_df['datetime'] >= '02/01/2021') & (temp_df['datetime'] <= '20/10/2021')
temp_df = temp_df.loc[mask]
temp_df.set_index("datetime", inplace=True)

count_pos = 0
count_neg = 0
count_z = 0
for d in temp_df.temp_diff:
    if d > 0: count_pos += 1
    elif d < 0: count_neg += 1
    else: count_z += 1

count_all = count_pos+count_neg+count_z
percent_pos = (count_pos/count_all)*100
percent_neg = (count_neg/count_all)*100
percent_z = (count_z/count_all)*100
lst_perc = []
lst_perc.append(f'{percent_pos:.1f}')
lst_perc.append(f'{percent_neg:.1f}')
lst_perc.append(f'{percent_z:.1f}')
# print(f'positive = {percent_pos:.1f}')
# print(f'negative = {percent_neg:.1f}')
# print(f'zero = {percent_z:.1f}')

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = [f'Positive : {percent_pos:.1f}',
          f'Negative : {percent_neg:.1f}',
          f'Zero : {percent_z:.1f}'
          ]


wedges, texts = ax.pie(lst_perc, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Percentage of Central tempearature difference between 2021 and 2020")

# plt.show()
# plt.pie(pie_var,labels=pie_var)
# plt.title('Percentage of Central tempearature difference between 2021 and 2021')
plt.show()