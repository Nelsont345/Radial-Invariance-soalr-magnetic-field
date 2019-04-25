# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:45:45 2018

@author: nmt115
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


datalist = pd.read_csv('2012.csv')



def magnitude(X, Y, Z):
    return np.sqrt(X*X + Y*Y + Z*Z)
 
datalist['Magnitude (nT)'] = magnitude(datalist['BX(nT)'], datalist['BY(nT)'], datalist['BZ(nT)'])
datalist.to_csv('withmagnitude')

datalist.plot(x ='DECIMALDAY', y ='Magnitude (nT)')


plt.legend()
plt.title("MAgnitude over time")
plt.xlabel("Decimal Day")
plt.ylabel("Magnitude (nT)")
plt.show()