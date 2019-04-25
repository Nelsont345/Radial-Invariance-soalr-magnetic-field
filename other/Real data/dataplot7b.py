# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:45:45 2018

@author: nmt115
"""
#3 with slice


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


datalist2 = pd.read_csv('Juno_2015.csv')
datalist = datalist2.set_index('DECIMALDAY')
#datalist = datalist[158800:188800]
#datalist = datalist[:-104302]   #284796 maximum, 237.650169


datalist['Distance from Sun(AU)'] = (datalist['POSNX(km)']/(1.496e+8))


datalist['Magnitude (nT)R^2'] = np.abs((datalist['BX(nT)']))


missing = np.linspace(1, 366, 366)
#datalist = datalist.reindex(datalist['DECIMALDAY'])
datalist = datalist.reindex(datalist.index.union(missing))


#%%
'''
datalist.to_csv('withmagnitude2')
'''

#%%
df = pd.DataFrame({'day': datalist['DOY'],
                       'hour': datalist['HOUR'], 'Magnitude (nT)R^2': datalist['Magnitude (nT)R^2']})
df = df.drop_duplicates(subset =['day','hour'], keep='first')
df =df.set_index(['day', 'hour'])


df.plot(legend = True)
plt.xlabel("day")
plt.ylabel("Magnitude (nT) R^2 (AU^2)")
plt.show()

#%%
datalist3 = datalist.reset_index()
datalist3.plot(x ='index', y ='Magnitude (nT)R^2', title = 'Magnitude vs. time', legend = True)
plt.xlabel("Decimal Day")
plt.ylabel("Magnitude (nT) R^2 (AU^2)")
plt.show()
#%%
'''
datalist.plot(x ='Distance from Sun(AU)', y ='Magnitude (nT)', title = 'Magnitude vs Distance', legend = True)
plt.xlabel("Distance from Sun(AU)")
plt.ylabel("Magnitude (nT)")
plt.show()
'''
#%%
'''
datalist4 = pd.Series(datalist['Distance from Sun(AU)'],datalist['Magnitude (nT)R^2' ], datalist['DOY'])
data;ist4 = datalist4.set_index('DOY').drop_duplicates(keep = 'first')
datalist4.drop_duplicates(plot(title = 'Magnitude vs Distance', legend = True)
plt.xlabel("Distance from Sun(AU)")
plt.ylabel("Magnitude (nT)r^2")
plt.show()
'''
#%%
datalist3.plot(x ='index', y ='Distance from Sun(AU)', title = 'Distance vs. Time', legend = True)
plt.xlabel("Decimal Day")
plt.ylabel("Distance from Sun(AU)")
plt.show()
