# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:10:06 2018

@author: nmt115
"""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sea
import pandas as pd
import matplotlib as mat

'''
Data From
http://www.sws.bom.gov.au/Solar/1/6
'''


sunspots= [95.5, 98.2, 98.3, 95.1, 90.9, 86.6, 84.5, 85.1, 85.3, 85.8, 87.7, 88.1]
month = [1,2,3,4,5,6,7,8,9,10,11,12]
data = pd.DataFrame({'Sunspots':np.array(sunspots)/max(sunspots), 'month':month})
sea.regplot(x= 'month', y ='Sunspots', data=data,ci=0)
plt.rc('axes', titlesize= 20, labelsize = 20)
plt.rc('xtick',labelsize = 12)
plt.rc('ytick',labelsize = 12)
plt.xlabel('Month')
plt.ylabel('Relative number of sunspots')
plt.title('Regressive plot of Number of Sunspots')