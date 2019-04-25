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
plt.rc('xtick',labelsize = 12)
plt.rc('ytick',labelsize = 12)

Year = 2012


if Year == 2012:
    datalist = pd.read_csv('ACE_2012.csv')                 #read in data
    datalist2 = pd.read_csv('JUno_2012.csv')      #create functiosn for 2 datasets
elif Year == 2015:
    datalist = pd.read_csv('ACE_2015.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2015.csv')   
elif Year == 2016:
    datalist = pd.read_csv('ACE_2016.csv')                 #read in data
    datalist2 = pd.read_csv('Juno_2016.csv') 

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
    frame['angle'] = -(180/(np.pi))*np.arctan((frame['Distance from Sun(AU)']*(1.5e8*2*np.pi)/(27*86400*400)))
    frame['originalangle'] = frame['atan2(y/x)']  
    frame['atan2(y/x)'] = frame['atan2(y/x)']-frame['angle']   
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
    
    
'''
fig, ax = plt.subplots(2,1)   
plotanglechange(datalist, 'ACE', ax[0])
plotanglechange(datalist2, 'Juno', ax[1])
ax[0].set_xlabel('Time: Day, Hour')
ax[1].set_xlabel('Time: Day, Hour')
ax[0].set_ylabel('Angle (degrees)')
ax[1].set_ylabel('Angle (degrees)')
ax[0].set_title('Graph of IMF angle to normal')
'''

#most useful

axs = datalist.hist('CorrectedAngle', bins=360)
datalist.hist('originalangle', bins=360, ax = axs)

'''
plt.figure()
plt.polar(datalist['CorrectedAngle'].mean(level =[0,1]), 
          absolute(datalist['BY(nT)'],datalist['BX(nT)'], 0).mean(level =[0,1]),  '.')
'''          
          
#plt.xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
#plt.absolute_importthetalim(-np.pi, np.pi)

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
    x.plot(y = 'Magnitude', ax = axs)
    axs.set_xlabel("Day")
    axs.set_ylabel("R^2 Br (AU^2 nT)")
    ax[0].set_title('Daily Average vs. Time')
    axs.legend(d)
    axs.fill_between(x.index, x['Magnitude']+q, x['Magnitude']-q, color='g', alpha=0.5)
    plt.show()
    

fig, ax = plt.subplots(2,1)    
plottingHourly(listHourlyaverages, 'ACE')
plottingDaily(listDailyaverages, 'ACE', ax[0])


plottingHourly(listHourlyaverages2, 'Juno')
plottingDaily(listDailyaverages2, 'Juno', ax[1])

#%%  magntiude adjustment, output's incorrect angles, off by 8

lengthlimit = 6   #in hours, minimum length of sequence
Flucuations = 12
removal_length = 12

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

listHourlyaverages2['Final_angle'], listHourlyaverages2['Final_Magnitude'], histogramJuno,total2  = reduceflucuations(
                                                                       np.array(listHourlyaverages2['angle']),
                                                                       np.array(listHourlyaverages2['Magnitude']),
                                                                       'Juno')


plt.xlabel('Day')
plt.ylabel('Angle')
plt.legend()

def histogramplot(x,d, axs):
    histogram = []
    for i in x:
        histogram.append(len(i))
    axs.hist(histogram, bins = max(histogram))
    axs.set_xlim([0, 200])
    ax[0].set_xlabel(' ')
    axs.set_ylabel('Number of Sequences')
#    plt.xlim([0,300])
    axs.set_title(['Histogram of sequences lengths', d, Year])
    
fig, ax = plt.subplots(2,1)
ax[1].set_xlabel('Length')
histogramplot(total, 'total', ax[0])
histogramplot(histogramACE, 'ACE', ax[1])
#histogramplot(histogramJuno, 'Juno', ax[1])


'''
listHourlyaverages['Final_angle'], listHourlyaverages['Final_Magnitude'] = listHourlyaverages['angle'], listHourlyaverages['Magnitude']
listHourlyaverages2['Final_angle'], listHourlyaverages2['Final_Magnitude'] = listHourlyaverages2['angle'], listHourlyaverages2['Magnitude']
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
    
def plots(w, z, a, s):   #plots for positive and negative
    d = w.plot(y= 'Magnitude')
    d =z.plot(y= 'Magnitude', ax=d)
    a.plot(y= 'Magnitude', ax=d)
    s.plot(y= 'Magnitude', color = 'green', ax=d)
    d.legend(["ACE Negative","ACE Positive","Juno Negative","Juno Positive"])

OverallHour1, OverallHour2, Hourgroupminus1, Hourgroup1 , Hourgroupminus2, Hourgroup2 = signgroups2(listHourlyaverages, listHourlyaverages2)
Hourgroupminus1, Hourgroup1 , Hourgroupminus2, Hourgroup2 = missingvalues(missing4, Hourgroupminus1, Hourgroup1, Hourgroupminus2, Hourgroup2)
plots(Hourgroupminus1, Hourgroup1, Hourgroupminus2, Hourgroup2 )

signafter = OverallHour1.plot(y = 'Magnitude_sign')
#OverallHour2.plot(y = 'Magnitude_sign', ax = signafter)

#%%
Hourgroupminus1['Magnitude2'] = Hourgroupminus1['Magnitude']*-1
q = Hourgroup1.plot(y = 'Magnitude', x = Hourgroup1.index)
Hourgroupminus1.plot(y = 'Magnitude2',  ax = q)
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
plots(Daygroupminus1*-1, Daygroup1, Daygroupminus2*-1, Daygroup2)

#OverallDay2.plot(y = 'Magnitude_sign')

#%%
theta = np.arange(-np.pi, np.pi*(6/5), np.pi/5)
r = np.arange(0, 11, 1)
#plt.polar(theta, r, 'bo')
plt.polar(listHourlyaverages['Final_angle']*(np.pi/180.),listHourlyaverages['Final_Magnitude'],  '.')
#%%

averagingperiodACE = 54
averagingperiodJuno = 50.6

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
plt.ylabel("R^2 Br, (AU^2 nT)")
plt.ylabel("R^2 Br, (AU^2 nT)")
