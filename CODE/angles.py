# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 21:35:20 2018

@author: nmt115
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sea
import scipy.stats as st

datalist2 = pd.read_csv('JUno_2012.csv')

plt.style.use('ggplot')
plt.rc('axes', titlesize= 30, labelsize = 25)
plt.rc('xtick',labelsize = 25)
plt.rc('ytick',labelsize = 25)
plt.rc('legend',fontsize = 25)


def absolute(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def extracolumns(x):    # Juno set up 
    x['Distance from Sun(AU)'] = absolute(x['POSNX(km)'], x['POSNY(km)'], x['POSNZ(km)'])/(1.496e+8)

def angularcorrection(frame):
    frame['angle'] = (180/(np.pi))*np.arctan((frame['Distance from Sun(AU)']*(1.5e8*2*np.pi)/(27*86400*400)))
    frame['error'] = 5



extracolumns(datalist2)
angularcorrection(datalist2)
datalist2 = datalist2.set_index(['Distance from Sun(AU)']).drop_duplicates()


plt.polar( datalist2.index, datalist2['angle'])
plt.fill_between(datalist2.index, datalist2['angle']+datalist2['error'], datalist2['angle']-datalist2['error'], color='g', alpha=0.5)
plt.xlabel('Distance from Sun(AU)')
plt.ylabel('Angle of Spiral')



datalist2.plot(y='DECIMALDAY')