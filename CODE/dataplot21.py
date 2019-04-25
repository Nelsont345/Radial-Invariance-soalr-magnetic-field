# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:45:45 2018

@author: nmt115
"""
#3 with slice


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


Year = 2015



if Year == 2012:
    datalist = pd.read_csv('ACE_2012.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2012.csv')      #create functiosn for 2 datasets
elif Year == 2015:
    datalist = pd.read_csv('ACE_2015.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2015.csv')   
elif Year == 2016:
    datalist = pd.read_csv('ACE_2012.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2016.csv') 

def removedoubles(x):
    return x.drop_duplicates(subset =['DOY','HOUR', 'MIN'], keep='first') #required as 2/388098 are duplicates...

datalist = removedoubles(datalist)
datalist2 = removedoubles(datalist2)


def absolute(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def extracolumns(x):    
    x['Distance from Sun(AU)'] = absolute(x['POSNX(km)'], x['POSNY(km)'], x['POSNZ(km)'])/(1.496e+8)
    x['XCOM(nT)R^2']     = (x['BX(nT)']*(x['Distance from Sun(AU)']**2))
    x['Magnitude'] = absolute(x['XCOM(nT)R^2'], x['BY(nT)'], x['BZ(nT)'])
    return x


datalist = extracolumns(datalist)
datalist2 = extracolumns(datalist2)

def setindices(x):
    return x.set_index(['DOY', 'HOUR', 'MIN'])

datalist = setindices(datalist)
datalist2 = setindices(datalist2)


 
#missing1 = np.linspace(1, 366, 8761)
missing1 = np.linspace(1, 366, 366)
missing2 = np.linspace(0, 23, 24)
missing3 = np.linspace(0, 59, 60)
missing = []
missing4=[]
for i in (missing1):
    for j in (missing2):
        missing4.append([i, j])
        for k in (missing3):
            missing.append([i, j, k])
            
def lostminutes(x, d):
    return x.reindex(x.index.union(d)) 
datalist = lostminutes(datalist, missing) #add days to dataframe
datalist2 = lostminutes(datalist2, missing) #add days to dataframe


datalist = datalist.reset_index()
datalist2 = datalist2.reset_index()
datalist = datalist.set_index(['DOY', 'HOUR'])
datalist2 = datalist2.set_index(['DOY', 'HOUR'])


#%%
            
def reverse(y, x): 
    try:
        q = np.arctan2(y, x)*(180/np.pi)
    except:
        q = 'NaN'
    return q
    
def arctancolumn(x):
    x['atan2(y/x)'] = reverse(x['BY(nT)'], x['BX(nT)'])

arctancolumn(datalist)
arctancolumn(datalist2)


def angularcorrection(frame):
    u = 0
    frame['angle'] = (frame['Distance from Sun(AU)']*1.5e8*2*np.pi*180)/(27*86400*400)
    frame['atan2(y/x)'] = frame['atan2(y/x)']+frame['angle']+90   
    list = []
    for k in frame['atan2(y/x)']:
        if k>180:
            u=u+1
            a = -(360-k)
        else:
            a=k
        list.append(a)
    newangles =  pd.Series(list, index = frame.index)
    print (u)
    return newangles

datalist['CorrectedAngle'] =angularcorrection(datalist)
datalist2['CorrectedAngle'] =angularcorrection(datalist)

#%%    Daily & Hourly Average  ####        NEW DATAFRAMES




def newdataframesdaily(x):
    y = pd.DataFrame(x['Magnitude'].mean(level = 0))  #default skip NaN
    y['angle'] = x['CorrectedAngle'].mean(level = 0)
    return y

listDailyaverages = newdataframesdaily(datalist)
listDailyaverages2 = newdataframesdaily(datalist2)

#Dataframe1
def newdataframeshourly(x):
    y = pd.DataFrame(x['Magnitude'].mean(level = [0, 1]))
    y['angle'] = x['CorrectedAngle'].mean(level = [0, 1])
    return y

listHourlyaverages =newdataframeshourly(datalist)
listHourlyaverages2 =newdataframeshourly(datalist2)
    

#%%    MAGNITUDE ONLY 

def plottingHourly(x):
    q = np.sqrt(x['Magnitude'].var())
    x.plot(y = 'Magnitude',title = 'Hourly Average vs. Time', legend = True, ylim = (0., 40.))
    plt.xlabel("Day, Hour")
    plt.ylabel("Magnitude")
    plt.fill_between(np.arange(0, 8784), x['Magnitude']+q, x['Magnitude']-q, color='r', alpha=0.5)
def plottingDaily(x):
    q = np.sqrt(x['Magnitude'].var())
    x.plot(y = 'Magnitude',title = 'Daily Average vs. Time', legend = True)
    plt.xlabel("Day")
    plt.ylabel("Magnitude")
    plt.fill_between(x.index, x['Magnitude']+q, x['Magnitude']-q, color='g', alpha=0.5)
    plt.show()
    
plottingHourly(listHourlyaverages)
#plottingDaily(listDailyaverages)

#plottingHourly(listHourlyaverages2)
#plottingDaily(listDailyaverages2)
'''
w [np.arange(len(listHourlyaverages)), list(listHourlyaverages['angle'])]
t = np.fft.rfft(listDailyaverages, 7)

plt.plot(np.fft.irfft(t, 7))
'''


q =listHourlyaverages['angle']

#%%
lengthlimit = 5
Flucuations = 15
s=[]
c = []
g =[]
b= 0
counter =0
validsequence = []
allsequences = [[0, 1]]
Flag = 0
counter2 = 0
Flagappend =[]
for i in q:
    counter = counter+1
    if b*i <0 and counter >lengthlimit and (Flag % 2) == 0:          #if seqence is long enough append, starts new sequence
        allsequences.append(c) 
        counter2 = counter2+1
    
    elif b*i <0 and counter >lengthlimit and (Flag % 2) == 1:     #True indicates previous was short
        for j in c:                                     #doesn't start new squence, adds to previous
            allsequences[-1].append(j)  
        Flag = 0
        
    
    elif b*i < 0  and counter <= lengthlimit and Flag > Flucuations:   #now change to include multiple sign changes
        for j in np.arange(counter):                   
            g.append(np.nan)
        allsequences.append(g)
        counter =0
        g=[]
        counter2 = counter2+1
    
    elif b*i < 0  and counter <= lengthlimit and Flag < Flucuations+1:               #else append equivalent in 
        for j in np.arange(counter):
            allsequences[-1].append(allsequences[-1][-1])   #replaces with last of next sequence
        Flag = Flag + 1                                #use odd or even to determine psitive or negative
        Flagappend.append(Flag)                 
        
    if b*i < 0:
        counter =0
        c = []
    
            
            
    c.append(i)    #need dummy array
    b = i   #where b is previous i
            
for i in np.arange(len(allsequences)):
    for j in allsequences[i]:
        s.append(j)        
        
'''  
c = []
r=[]
b= 0
counter =0
validsequence = []
allsequences = []
for i in q:
    counter = counter+1
    if b*i <0 and counter >24:                  #if seqence is long enough append
        allsequences.append(c) 
        for i in c:             
            validsequence.append(i)
        counter = 0 
        c=[]
    elif b*i < 0  and counter <= 24:               #else append equivalent in 
        for j in np.arange(counter):
            validsequence.append(b)
            r.append(np.nan)
        allsequences.append(r)
        r=[]
        counter =0
        c = []
    c.append(i)    #need dummy array
    b = i   #where b is previous i
'''

#%%  SIgn of Magnitude, creates new dataframes


def sign(frame):
    list = []
    for k in frame:
        if k>0:
            a = np.float64(1)
        elif k<0:
            a = np.float64(-1)
        elif k == np.nan:
            a = np.nan
        elif k == 0:
            a = np.float64(0)
        list.append(a)
    sign_frame =  pd.Series(list, index = frame.index)
    return sign_frame
     
def signgroups2(v, u):
    v['M'] = v['angle']
    v['Magnitude_sign'] = sign(v.M)  
    w= v.groupby(['Magnitude_sign']).get_group(-1.0)
    z= v.groupby(['Magnitude_sign']).get_group(1.0)
    u['M'] = u['angle']
    u['Magnitude_sign'] = sign(u.M)  
    a= u.groupby(['Magnitude_sign']).get_group(-1.0)
    s= u.groupby(['Magnitude_sign']).get_group(1.0)
    return w, z, a, s
    
    
def missingvalues(d, w, z, a, s): 
    w = w.reindex(w.index.union(d))
    z = z.reindex(z.index.union(d))
    a = a.reindex(a.index.union(d)) 
    s = s.reindex(s.index.union(d))     
    return w, z, a, s
    
def plots(w, z, a, s):   #plots for positive and negative
    d = w.plot(y= 'Magnitude')
    d =z.plot(y= 'Magnitude', ax=d)
    a.plot(y= 'Magnitude', ax=d)
    s.plot(y= 'Magnitude', ax=d)
    d.legend(["ACE Negative","ACE Positive","Juno Negative","Juno Positive"])

Hourgroupminus1, Hourgroup1 , Hourgroupminus2, Hourgroup2 = signgroups2(listHourlyaverages, listHourlyaverages2)
Hourgroupminus1, Hourgroup1 , Hourgroupminus2, Hourgroup2 = missingvalues(missing4, Hourgroupminus1, Hourgroup1, Hourgroupminus2, Hourgroup2)
#plots(Hourgroupminus1, Hourgroup1, Hourgroupminus2, Hourgroup2 )


missing5 = np.arange(0.0, 366.0)
Daygroupminus1, Daygroup1 , Daygroupminus2, Daygroup2 = signgroups2(listDailyaverages, listDailyaverages2)
Daygroupminus1, Daygroup1 , Daygroupminus2, Daygroup2 = missingvalues(missing5, Daygroupminus1, Daygroup1, Daygroupminus2, Daygroup2)
plots(Daygroupminus1, Daygroup1, Daygroupminus2, Daygroup2)

#%%
averagingperiod = 5

x = np.arange(1, 366, averagingperiod)

def createdataset(y, e):
    y['Magnitudeandsign']= y['Magnitude']*y['Magnitude_sign']
    e['Magnitudeandsign']= e['Magnitude']*e['Magnitude_sign']
    error1 = np.sqrt(y['Magnitude'].var())
    error2 = np.sqrt(e['Magnitude'].var())
    z1 = []
    z2 = []
    for i in x:
        negative = y['Magnitudeandsign'].loc[i:i+averagingperiod].mean()
        positive = e['Magnitudeandsign'].loc[i:i+averagingperiod].mean()
        #z1.append(negative)
        #z2.append(positive)
        
        if positive < -1*negative:
            z1.append(negative)
            z2.append(np.NaN)
        elif positive > -1*negative:
            z2.append(positive)
            z1.append(np.NaN)
        else:
            z1.append(np.NaN)
            z2.append(np.NaN)
       

    w = pd.DataFrame({'DOY':x, 'Magnitude':z1, 'var' :error1})
    #w = w.set_index('DOY')
    r = pd.DataFrame({'DOY':x, 'Magnitude':z2, 'var' :error2})
    #r = r.set_index('DOY')
    return w, r

DataFrame1, DataFrame3 = createdataset(Daygroupminus1, Daygroup1)
DataFrame2, DataFrame4 = createdataset(Daygroupminus2, Daygroup2)

        
d = DataFrame1.plot(x = 'DOY', y ='Magnitude',  color='Green', yerr = 'var', capsize=4)
DataFrame2.plot(x = 'DOY', y ='Magnitude' , color='Purple', ax =d, yerr = 'var', capsize=4)
DataFrame3.plot(x = 'DOY', y ='Magnitude' , color='Red', ax =d, yerr = 'var', capsize=4)
DataFrame4.plot(x = 'DOY', y ='Magnitude' , color='Blue', ax =d, yerr = 'var', capsize=4)
plt.title(averagingperiod)
d.legend(["ACE Negative","Juno Negative", "ACE Positive","Juno Positive"])
plt.xlabel("Day of Year")
plt.ylabel("Magntiude, nT (R^2)")




#%%
#listDailyaverages.to_csv('dailyav')
#listHourlyaverages.to_csv('dailyav2')