# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:45:45 2018

@author: nmt115
"""
#3 with slice


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sea
import scipy.stats as st


plt.style.use('ggplot')
plt.rc('axes', titlesize= 30, labelsize = 25)
plt.rc('xtick',labelsize = 25)
plt.rc('ytick',labelsize = 25)

Year = 2016
if Year == 2012:
    datalist = pd.read_csv('ACE_2012.csv')                 #read in data
    datalist2 = pd.read_csv('JUno_2012.csv')      #create functiosn for 2 datasets    
    data = pd.read_csv('2012sunspot.csv')
elif Year == 2015:
    datalist = pd.read_csv('ACE_2015.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2015.csv')    
    data = pd.read_csv('2015sunspot.csv')
elif Year == 2016:
    datalist = pd.read_csv('ACE_2016.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2016.csv')    
    data = pd.read_csv('2016sunspot.csv')

def removedoubles(x):
    return x.drop_duplicates(subset =['DOY','HOUR', 'MIN'], keep='first') #required as 2/388098 are duplicates...

datalist = removedoubles(datalist)
datalist2 = removedoubles(datalist2)



def absolute(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def extracolumns(x):    # Juno set up 
    x['Distance from Sun(AU)'] = absolute(x['POSNX(km)'], x['POSNY(km)'], x['POSNZ(km)'])/(1.496e+8)
    x['XCOM(nT)R^2']     = (x['BX(nT)']*(x['Distance from Sun(AU)']**2))
    x['Magnitude'] = abs(x['XCOM(nT)R^2'])   #R^2 Br
    return x

def extracolumns2(x):    #ACE set up 
    x['Distance from Sun(AU)'] = (1.496e+8 - x['POSNX(km)'])/(1.496e+8)  #approx
    x['XCOM(nT)R^2']     = (x['BX(nT)']*(x['Distance from Sun(AU)']**2))
    x['Magnitude'] = abs(x['XCOM(nT)R^2'] )    #R^2 Br
    x['OverallMagnitude'] = abs(x['XCOM(nT)R^2'] )    #R^2 Br    
    return x

datalist = extracolumns2(datalist)    #ACE is distance from Sun
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
    q = np.arctan2(y, x)*(180/np.pi)        
    return q
    
def arctancolumn(x):
    x['atan2(y/x)'] = reverse(x['BY(nT)'], x['BX(nT)'])

arctancolumn(datalist)
arctancolumn(datalist2)


def angularcorrection(frame):
    u = 0
    frame['angle'] = (180/(np.pi))*np.arctan((frame['Distance from Sun(AU)']*(1.5e8*2*np.pi)/(27*86400*400)))
    frame['originalangle'] = frame['atan2(y/x)']  
    frame['atan2(y/x)'] = frame['atan2(y/x)']+frame['angle']+90  #90 added to get postive, negative split
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
datalist2['CorrectedAngle'] =angularcorrection(datalist2)


def plotanglechange(x, d, f):
    if f == ax[0]:
        c = pd.DataFrame(x['angle']).dropna(axis=0)
    elif f == ax[1]:
        c= pd.DataFrame(x['angle'])
    c.plot(ax =f)      
    f.legend(d)
    
    
datalist['Showoriginalangle'] = -datalist['originalangle']

opacity = 0.8
axs = datalist.hist('originalangle', bins=45 , normed=True,alpha=opacity,color = 'Red')
datalist.hist('CorrectedAngle', bins=45, ax = axs , normed=True, alpha=opacity, color = 'Blue')
n = ('$\phi_P$')
plt.legend(['$\phi_B$','$\phi_B-\phi_P$'])
plt.axvline(x= -50.1, linewidth=2, color='g')
plt.axvline(x= -5.2, linewidth=2, color='g')
plt.title('Angle frequency')
plt.ylabel('Normalised frequency')
plt.xlabel('Angle, $\phi^{\circ}$')

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

def plottingHourly(x, d):
    q = np.sqrt(x['Magnitude'].var())
    x.plot(y = 'Magnitude',title = 'Hourly Average vs. Time', legend = True, ylim = (0., 40.))
    plt.xlabel("Day, Hour")
    plt.ylabel("R^2 Br")
    plt.title(d)
    plt.fill_between(np.arange(0, 8784), x['Magnitude']+q, x['Magnitude']-q, color='r', alpha=0.5)
def plottingDaily(x, d, axs):
    q = np.sqrt(x['Magnitude'].var())
    x.plot(y = 'Magnitude', ax = axs, sharex = True, sharey = True)
    axs.set_xlabel("Day")
    axs.set_ylabel("R^2 Br (AU^2 nT)")
    ax[0].set_title('Daily Average vs. Time, 2012')
    axs.legend(d)
    axs.fill_between(x.index, x['Magnitude']+q, x['Magnitude']-q, color='g', alpha=0.5)
    plt.show()
    

fig, ax = plt.subplots(2,1)    
'''
plottingHourly(listHourlyaverages, 'ACE')
plottingHourly(listHourlyaverages2, 'Juno')
'''


plottingDaily(listDailyaverages2, 'Juno', ax[1])
plottingDaily(listDailyaverages, 'ACE', ax[0])
#%%  magntiude adjustment, output's incorrect angles, off by 8

lengthlimit = 6  #in hours, minimum length of sequence
Flucuations = 6
removal_length = 6

def reduceflucuations(direction, size, d):
    s=[]
    corsize = []
    c = []
    c2=[]
    total=[]
    b= 0
    allsequences = [[0]]
    sizesequences = [[0]]
    Flag = 0
    for i in range(len(direction)):
        if  direction[i] == np.nan and b!= np.nan:
            allsequences.append(c)
            sizesequences.append(c2)
            
            
            #if previous and current are NaN keep sequence of NaN going
        if  direction[i] == np.nan and b == np.nan:
            for j in range(len(c)-1):                                   
                allsequences[-1].append(c[j])
                sizesequences[-1].append(c2[j])                        
            
        if b*direction[i] <=0:
            total.append(c)
            if len(c) >lengthlimit:          
                if (Flag % 2) == 0:          #if seqence is long enough append, starts new sequence
                    allsequences.append(c)
                    sizesequences.append(c2)
            
   
                elif (Flag % 2) == 1:     #Odd indicates previous was short
                    for j in range(len(c)):                                     #doesn't start new squence, adds to previous
                        allsequences[-1].append(c[j])
                        sizesequences[-1].append(c2[j])
                    Flag = 0
 
            elif len(c) <= lengthlimit:
                if Flag > Flucuations and len(allsequences[-1]) >= Flag :   #now changed to include multiple sign changes, Flucuation set 
                    for j in (-np.arange(Flag)-1):      #counting backwards from Flag value, changing to NaN
                        allsequences[-1][j] = np.nan
                        sizesequences[-1][j] = np.nan   
                            
                    Flag = 0
       
                elif Flag < Flucuations+1:               #else append equivalent in previous sequence 
                    
                    for j in np.arange(len(c)):
                        if j < lengthlimit:
                            allsequences[-1].append(allsequences[-1][-1])   #replaces with last of next sequence
                            sizesequences[-1].append(sizesequences[-1][-1]) 
                                                                                    
                    Flag = Flag + 1                                #use odd or even to determine psitive or negative
                  
                    
                    
                    
            c = []
            c2 = [] 
        
        c.append(direction[i])    #need dummy array
        c2.append(size[i]) 
        b = direction[i]   #where b is previous i
        print (i)
        
        
    del(allsequences[0][0])
    del(sizesequences[0][0])
    for i in range(len(allsequences)):   #replace all sequnces less than 100hrs with nan values              
        if len(allsequences[i]) < removal_length:                        
            for j in range(len(allsequences[i])):     #i is the sequence                
                allsequences[i][j] = np.nan
                sizesequences[i][j] = np.nan
                

                
    
    for i in np.arange(len(allsequences)):
        for j in range(len(allsequences[i])):
            s.append(allsequences[i][j])
            corsize.append(sizesequences[i][j])                 

    if len(s) < 8784:
        for i in np.arange(8784-len(s)):
            s.append(0)
            corsize.append(0)
    
    elif len(s) > 8784:
        for i in np.arange(len(s)-8784):
            del(s[-i])
            del(corsize[-i])
        
    
    plt.plot(np.arange(len(s))/24, s, label='all the angles')
    plt.title(d)
    return s, corsize, allsequences, total
            
plt.figure()       
listHourlyaverages['Final_angle'], listHourlyaverages['Final_Magnitude'], histogramACE, total = reduceflucuations(
                                                                    np.array(listHourlyaverages['angle']),
                                                                    np.array(listHourlyaverages['Magnitude']),                                                                    
                                                                    'ACE')
'''
listHourlyaverages2['Final_angle'], listHourlyaverages2['Final_Magnitude'], histogramJuno,total2  = reduceflucuations(
                                                                       np.array(listHourlyaverages2['angle']),
                                                                       np.array(listHourlyaverages2['Magnitude']),
                                                                       'Juno')
'''

plt.xlabel('Day')
plt.ylabel('Angle')
plt.legend()

def histogramplot(x,d, axs):
    histogram = []
    for i in x:
        histogram.append(len(i))
    axs.hist(histogram, bins = max(histogram))
#    axs.set_xlim([0, 200])
    ax[0].set_xlabel(' ')
    axs.set_ylabel('Frequency')
    axs.set_xlim([0,200])
    axs.legend()
    
fig, ax = plt.subplots(2,1, sharey=False, sharex=True)
ax[1].set_xlabel('Sequence length, Hours')
ax[0].set_title('Histogram of sequences lengths, 2012')
histogramplot(total, 'total', ax[0])
histogramplot(histogramACE, 'ACE', ax[1])
#histogramplot(histogramJuno, 'Juno', ax[1])



listHourlyaverages['Final_angle'], listHourlyaverages['Final_Magnitude'] = listHourlyaverages['angle'], listHourlyaverages['Magnitude']
listHourlyaverages2['Final_angle'], listHourlyaverages2['Final_Magnitude'] = listHourlyaverages2['angle'], listHourlyaverages2['Magnitude']

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
        else:
            a = np.nan
        list.append(a)
    sign_frame =  pd.Series(list, index = frame.index)
    return sign_frame
     
def signgroups2(v, u):
    v['Magnitude_sign'] = sign(v.Final_angle)  
    w= v.groupby(['Magnitude_sign']).get_group(-1.0)
    z= v.groupby(['Magnitude_sign']).get_group(1.0)
    u['Magnitude_sign'] = sign(u.Final_angle)  
    a= u.groupby(['Magnitude_sign']).get_group(-1.0)
    s= u.groupby(['Magnitude_sign']).get_group(1.0)
    return v, u, w, z, a, s
    
    
def missingvalues(d, w, z, a, s): 
    w = w.reindex(w.index.union(d))
    z = z.reindex(z.index.union(d))
    a = a.reindex(a.index.union(d)) 
    s = s.reindex(s.index.union(d))     
    return w, z, a, s
    
def plots(w, z, ax1, legend1, colour1, colour2):   #plots for positive and negative    
    w.plot(ax = ax1, y= 'Magnitude', color = colour1)
    z.plot(y= 'Magnitude', ax = ax1, color = colour2)    
    ax1.legend(legend1)

OverallHour1, OverallHour2, Hourgroupminus1, Hourgroup1 , Hourgroupminus2, Hourgroup2 = signgroups2(listHourlyaverages, listHourlyaverages2)
Hourgroupminus1, Hourgroup1 , Hourgroupminus2, Hourgroup2 = missingvalues(missing4, Hourgroupminus1, Hourgroup1, Hourgroupminus2, Hourgroup2)

fig, ax = plt.subplots(2,1) 
plots(Hourgroupminus1, Hourgroup1, ax[0], ["ACE Negative","ACE Positive"], 'Blue', 'Green')
plots(Hourgroupminus2, Hourgroup2, ax[1],["Juno Negative","Juno Positive"], 'Orange', 'Red')

#signafter = OverallHour1.plot(y = 'Magnitude_sign')
#OverallHour2.plot(y = 'Magnitude_sign', ax = signafter)

#%%
Hourgroupminus1['Magnitude2'] = Hourgroupminus1['Magnitude']*-1
Hourgroupminus2['Magnitude2'] = Hourgroupminus2['Magnitude']*-1
q = Hourgroup1.plot(y = 'Magnitude', x = Hourgroup1.index)
Hourgroupminus1.plot(y = 'Magnitude2', x = Hourgroupminus1.index, ax = q)
#plt.xticks(np.arange(0,366,50))

#Hourgroup2.plot(y = 'Magnitude', x = Hourgroup2.index, ax = q)
#Hourgroupminus2.plot(y = 'Magnitude2', x = Hourgroup2.index, ax = q)

plt.title('Negative and Positive Sectors, ACE')
plt.ylabel("R^2 Br, nT (AU^2)")

#%% NEw averaging from manipulated data

listDailyaverages['Final_angle'] = listHourlyaverages['Final_angle'].mean(level = 0)
listDailyaverages2['Final_angle'] = listHourlyaverages2['Final_angle'].mean(level = 0)
listDailyaverages['Final_Magnitude'] = listHourlyaverages['Final_Magnitude'].mean(level = 0)
listDailyaverages2['Final_Magnitude'] = listHourlyaverages2['Final_Magnitude'].mean(level = 0)  


missing5 = np.arange(0.0, 366.0)
OverallDay1, OverallDay2, Daygroupminus1, Daygroup1 , Daygroupminus2, Daygroup2 = signgroups2(listDailyaverages, listDailyaverages2)
Daygroupminus1, Daygroup1 , Daygroupminus2, Daygroup2 = missingvalues(missing5, Daygroupminus1, Daygroup1, Daygroupminus2, Daygroup2)

fig, ax = plt.subplots(2,1, sharey =True) 
plots(Daygroupminus1*-1, Daygroup1, ax[0], ["ACE Negative","ACE Positive"], 'Blue', 'Green')
plots(Daygroupminus2*-1, Daygroup2,ax[1],["Juno Negative","Juno Positive"], 'Orange', 'Red')
ax[0].set_title('2012, ACE and Juno Daily Averages')
ax[1].set_ylabel("                                         $R^{2} B_{r}$ $(AU^{2}$ $nT)$")
ax[1].set_xlabel('Day of year')
ax[0].set_yticks([-5, 0, 5])
ax[1].set_yticks([-5, 0, 5])
#OverallDay2.plot(y = 'Magnitude_sign')
Daygroup2['Final_Magnitude'] = Daygroup2['Final_Magnitude']

#%%

averagingperiodACE = 27*2
averagingperiodJuno = 25*2

def createdataset(averagingperiod, y, e):
    x = np.arange(1, 366, averagingperiod)
    y['Magnitudeandsign']= y['Final_Magnitude']*y['Magnitude_sign']
    e['Magnitudeandsign']= e['Final_Magnitude']*e['Magnitude_sign']
    error1 = np.sqrt(y['Final_Magnitude'].var())
    error2 = np.sqrt(e['Final_Magnitude'].var())
    z1 = []
    z2 = []
    for i in x:        
        
        negative = y['Magnitudeandsign'].loc[i:i+averagingperiod].mean()
        positive = e['Magnitudeandsign'].loc[i:i+averagingperiod].mean()
        
        z1.append(negative)
        z2.append(positive)
        

            
            
    w = pd.DataFrame({'DOY': np.arange(len(z1))*(366/len(z1)), 'Magnitude':z1, 'var' :error1})
    #w = w.set_index('DOY')
    r = pd.DataFrame({'DOY':np.arange(len(z2))*(366/len(z2)), 'Magnitude':z2, 'var' :error2})
    #r = r.set_index('DOY')
    
    return w, r

DataFrame1, DataFrame3 = createdataset(averagingperiodACE, Daygroupminus1, Daygroup1)
DataFrame2, DataFrame4 = createdataset(averagingperiodJuno, Daygroupminus2, Daygroup2)

 
 
ax[0] = DataFrame1.plot(x = 'DOY', y ='Magnitude',  color='Blue', linestyle = '--', yerr = 'var')#, capsize=4)
DataFrame2.plot(x = 'DOY', y ='Magnitude' , color='Orange', ax = ax[0], linestyle = '--',yerr = 'var')#, capsize=4)
DataFrame3.plot(x = 'DOY', y ='Magnitude' , color='Green', ax =ax[0],linestyle = '--', yerr = 'var')#, capsize=4)
DataFrame4.plot(x = 'DOY', y ='Magnitude' , color='Red', ax =ax[0], linestyle = '--',yerr = 'var')#, capsize=4)
    

#DataFrame2.append(pd.DataFrame([[366.0, 4.4, 1.3]], columns=list({'DOY', 'Magnitude', 'var'})), ignore_index = True)

DataFrame1.plot.scatter(x = 'DOY', y ='Magnitude',  color='Blue',ax =ax[0])#, capsize=4)
DataFrame2.plot.scatter(x = 'DOY', y ='Magnitude' , color='Orange', ax = ax[0])#, capsize=4)
DataFrame3.plot.scatter(x = 'DOY', y ='Magnitude' , color='Green', ax =ax[0])#, capsize=4)
DataFrame4.plot.scatter(x = 'DOY', y ='Magnitude' , color='Red', ax =ax[0])#, capsize=4)


g = ('ACE:', averagingperiodACE, 'days', 'Juno:', averagingperiodJuno, 'days', Year)
plt.title(g)
plt.legend(["ACE Negative","Juno Negative", "ACE Positive","Juno Positive"])
plt.xlabel("Day of Year")
plt.ylabel("$R^{2} B_{r}$ $(AU^{2}$ $nT)$")


#%%

#data = data.set_index('Day') 
sun1 = pd.read_csv('2012sunspot.csv')
sun2 = pd.read_csv('2015sunspot.csv')
sun3 = pd.read_csv('2016sunspot.csv')

ACE1 = pd.read_csv('DaiyACE2012.csv')
ACE2 = pd.read_csv('DaiyACE2015.csv')
ACE3 = pd.read_csv('DaiyACE2016.csv')

Juno1 = pd.read_csv('DaiyJuno2012.csv')
Juno2 = pd.read_csv('DaiyJuno2015.csv')
Juno3 = pd.read_csv('DaiyJuno2016.csv')
  
plt.rc('legend', fontsize=16)
plt.rc('axes', titlesize= 30, labelsize = 25)
plt.rc('xtick',labelsize = 25)
plt.rc('ytick',labelsize = 20)


period = 20
periodB = 2
range1 = np.arange(0, 366, period)
range2 = np.arange(0, 366, periodB)

def magav(x):
    day = 0
    days = []
    value= []
    for i in range1:        
            averaged = x['Magnitude'].loc[i:i+period].mean()
            value.append(averaged)
            days.append(day)
            day = day+period
    return value, days

def magav2(x):
    value= []
    for i in range2:        
            averaged = x.loc[i:i+periodB].mean()
            value.append(averaged)
    value = pd.DataFrame(value)
    return value

sun1 = magav2(sun1)
sun2 = magav2(sun2)
sun3 = magav2(sun3)

        
value1, days1 = magav(ACE1)
value2, days2 = magav(Juno1)

value3, days3 = magav(ACE2)
value4, days4 = magav(Juno2)

value5, days5 = magav(ACE3)
value6, days6 = magav(Juno3)

listDailyaverages = newdataframesdaily(datalist)
listDailyaverages2 = newdataframesdaily(datalist2)


q= np.arange(3, 6, 1)
opacity = 0.3

fig, ax = plt.subplots(3,1)  
ax[0].cla
ax[0].set_ylabel(' ', color = 'Red')
ax[0].set_title('|B| and sunspot number')
ax[0].plot(days2, value2)
ax[0].legend(['2012'])
ax[0].set_yticks(q)
ax[0] = ax[0].twinx()
ax[0].bar(range2, sun1['Dailysunspotno'],  width = period, alpha=opacity, color = 'Blue')
ax[0].set_yticks(np.arange(0., 200., 50.))
 
ax[1].plot(days4, value4)   
ax[1].legend(['2015'])
ax[1].set_yticks(q)
ax[1] = ax[1].twinx()
ax[1].bar(range2, sun2['Dailysunspotno'], width = period, alpha=opacity, color = 'Blue')
ax[1].set_yticks(np.arange(0., 200., 50.))


ax[2].plot(days6, value6) 
ax[2].set_ylabel('                                     $|B|$ $(AU^{2}$ $nT)$', color = 'Red')
ax[2].set_xlabel('Day')
ax[2].legend(['2016'])
ax[2].set_yticks(q)
ax[2] = ax[2].twinx()
ax[2].set_ylabel('                                  Daily count', color = 'Blue')
ax[2].bar(range2, sun3['Dailysunspotno'],width = period, alpha=opacity, color = 'Blue')
ax[2].set_yticks(np.arange(0., 200., 50.))


ax[0].get_shared_x_axes().join(ax[0], ax[1], ax[2])
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])

#%%

t =datalist2.mean(level = [0, 1])
#t.to_csv('2012CIR.csv')
import random

t12 = pd.read_csv('2012CIR.csv')
t15 = pd.read_csv('2015CIR.csv')
t16 = pd.read_csv('2016CIR.csv')

n=0
f = np.random.normal(1.56, 0.5, 9000)
g = absolute(t12['BX(nT)'], t12['BY(nT)'], t12['BZ(nT)'])
for i in range(len(g)):   
    if g[i] > 4:  
        t12['BX(nT)'].loc[i] = f[i]
        t12['BY(nT)'].loc[i] = f[i]
        t12['BZ(nT)'].loc[i] = f[i]
        n = n+1
print (n)

fig, ax = plt.subplots(5,1) 
fig.subplots_adjust(hspace=0.1)
ax[0].plot(t12['DECIMALDAY'], absolute(t12['BX(nT)'], t12['BY(nT)'], t12['BZ(nT)']))
ax[1].plot(t12['DECIMALDAY'], absolute(t12['BX(nT)'], t12['BY(nT)'], t12['BZ(nT)']))
ax[2].plot(t15['DECIMALDAY'], absolute(t12['BX(nT)'], t12['BY(nT)'], t12['BZ(nT)']))
ax[3].plot(t15['DECIMALDAY'], absolute(t15['BX(nT)'], t15['BY(nT)'], t15['BZ(nT)']))
ax[4].plot(t16['DECIMALDAY'], absolute(t16['BX(nT)'], t16['BY(nT)'], t16['BZ(nT)'])) 



ax[0].set_title('|B| signals measured by Juno')
ax[4].set_xlabel('Relative Day')
ax[2].set_ylabel('|B| nT')

ax[0].set_xlim([0, 180])
ax[1].set_xlim([180, 360])
ax[2].set_xlim([0, 180])
ax[3].set_xlim([180, 360])
ax[4].set_xlim([0, 180])

ax[0].set_ylim([0, 6.5])
ax[1].set_ylim([0, 6.5])
ax[2].set_ylim([0, 6.5])
ax[3].set_ylim([0, 2.8])
ax[4].set_ylim([0, 2.8])

ax[0].get_shared_x_axes().join(ax[0], ax[1], ax[2], ax[3], ax[4])
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[3].set_xticklabels([])

z=2

ax[0].legend(['2012, days 0 to 180'], loc=z)
ax[1].legend(['2012, days 180 to 366'], loc=z)
ax[2].legend(['2015, days 0 to 180'], loc=z)
ax[3].legend(['2015, days 180 to 365'], loc=z)
ax[4].legend(['2016, days 0 to 180'], loc=z)

