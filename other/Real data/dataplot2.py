# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:45:45 2018

@author: nmt115
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


datalist = pd.read_csv('2012.csv')
datalist = datalist[:-104302]


def magnitude(X, Y, Z):
    return np.sqrt(X*X + Y*Y + Z*Z)
 
datalist['Magnitude (nT)'] = magnitude(datalist['BX(nT)'], 
                                       datalist['BY(nT)'], 
                                       datalist['BZ(nT)'])



datalist['Distance from Sun(AU)'] = magnitude(datalist['POSNX(km)'], 
                                              datalist['POSNY(km)'],
                                              datalist['POSNZ(km)'])/(1.496e+8)

#%%
#datalist.to_csv('withmagnitude')


#%%
datalist.plot(x ='DECIMALDAY', y ='Magnitude (nT)', title = 'Magnitude vs. time', legend = True)
plt.xlabel("Decimal Day")
plt.ylabel("Magnitude (nT)")
plt.show()
#%%
datalist.plot(x ='Distance from Sun(AU)', y ='Magnitude (nT)', title = 'Magnitude vs Distance', legend = True)
plt.xlabel("Distance from Sun(AU)")
plt.ylabel("Magnitude (nT)")
plt.show()
#%%
datalist.plot(x ='DECIMALDAY', y ='Distance from Sun(AU)', title = 'Distance vs. Time', legend = True)
plt.xlabel("Decimal Day")
plt.ylabel("Distance from Sun(AU)")
plt.show()
