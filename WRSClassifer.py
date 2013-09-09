import ApplianceFeatures as AF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
import csv
from datetime import timedelta
import copy
import cPickle
import multiprocessing
import scipy.signal as signal

def CompareStripped(appl1,appl2,l1,l2):
    timeAvg = '500L'
    num=24
    #feature = (appl1.resample(rule=timeAvg).values[:,:num],appl2.resample(rule=timeAvg).values[:,:num])
    #ts = (l1.resample(rule=timeAvg).values[:,:num],l2.resample(rule=timeAvg).values[:,:num])
    #timeIdx = (l1.resample(rule=timeAvg).index[:-len(feature[0])+1],l2.resample(rule=timeAvg).index[:-len(feature[1])+1])

    feature = (appl1.values[:,:num],appl2.values[:,:num])
    ts = (l1.values[:,:num],l2.values[:,:num])
    timeIdx = (l1.index[:-len(feature[0])+1],l2.index[:-len(feature[1])+1])
    #l1Comp = pd.Series([np.mean([np.linalg.norm(feature[0][col].values-ts[0][col][ts[0].index[ii]:ts[0].index[ii+len(feature[0])-1]].values)**2/len(feature) for col in ts[0].columns]) for ii in xrange(len(ts[0].index)-len(feature[0])+1)],ts[0].index[:-len(feature[0])+1])
    #l2Comp = pd.Series([np.mean([np.linalg.norm(feature[1][col].values-ts[1][col][ts[1].index[ii]:ts[1].index[ii+len(feature[1])-1]].values)**2/len(feature) for col in ts[1].columns]) for ii in xrange(len(ts[1].index)-len(feature[1])+1)],ts[1].index[:-len(feature[1])+1])
    l1Comp = pd.Series((np.linalg.norm(feature[0])**2+np.linalg.norm(ts[0])**2
                               -2*signal.convolve2d(ts[0],feature[0][::-1,::-1],mode='valid')[:,0])/(feature[0].shape[0]*feature[0].shape[1]),timeIdx[0])
    l2Comp = pd.Series((np.linalg.norm(feature[1])**2+np.linalg.norm(ts[1])**2
                               -2*signal.convolve2d(ts[1],feature[1][::-1,::-1],mode='valid')[:,0])/(feature[1].shape[0]*feature[1].shape[1]),timeIdx[1])
    return pd.DataFrame({'l1Comp':l1Comp,'l2Comp':l2Comp})

def CompareToFeature(appl1,appl2,ts,start,stop,FeatureType):
    prebaseTime = pd.to_datetime(start)-timedelta(seconds=5),pd.to_datetime(start)-timedelta(seconds=0)
    postbaseTime = pd.to_datetime(stop)+timedelta(seconds=0),pd.to_datetime(stop)+timedelta(seconds=5)
    l1pre = (ts[0].index > prebaseTime[0]) & (ts[0].index < prebaseTime[1])
    l2pre = (ts[1].index > prebaseTime[0]) & (ts[1].index < prebaseTime[1])
    l1post = (ts[0].index > postbaseTime[0]) & (ts[0].index < postbaseTime[1])
    l2post = (ts[1].index > postbaseTime[0]) & (ts[1].index < postbaseTime[1])
    if FeatureType == 'on':
        baseline1 = ts[0].loc[l1pre].median()
        baseline2 = ts[1].loc[l2pre].median()
    elif FeatureType == 'off':
        baseline1 = ts[0].loc[l1post].median()
        baseline2 = ts[1].loc[l2post].median()
    elif FeatureType == 'ss':
        baseline1 = (ts[0].loc[l1post].median()
                     +ts[0].loc[l1pre].median())/2
        baseline2 = (ts[1].loc[l2post].median()
                     +ts[1].loc[l2pre].median())/2
    sl1 = ts[0].loc[(ts[0].index > pd.to_datetime(start)) & (ts[0].index <pd.to_datetime(stop))]
    sl2 = ts[1].loc[(ts[1].index > pd.to_datetime(start)) & (ts[1].index <pd.to_datetime(stop))]
    sl1 -= baseline1
    sl2 -= baseline2
    return CompareStripped(appl1,appl2,sl1,sl2)

def DeviceFitValue(appl1,appl2,ts,start,stop,FeatureType):
    data = CompareToFeature(appl1,appl2,ts,start,stop,FeatureType)
    return np.amin(data.l1Comp)+np.amin(data.l2Comp)

def MatchEvent((time,appliances,ts)):
    num = 24
    applist = []
    jumpDet = 5
    for appl in appliances:
        if time[3] == 'on':
            dOn = np.absolute((np.median(appl[0].values[-jumpDet:,:num],axis=0)-np.median(appl[0].values[:jumpDet,:num],axis=0)).sum())+np.absolute((np.median(appl[1].values[-jumpDet:,:num],axis=0)-np.median(appl[1].values[:jumpDet,:num],axis=0)).sum())
            Comp = DeviceFitValue(appl[0],appl[1],ts,time[0],time[1],'on')/dOn
        else:
            dOff = np.absolute((np.median(appl[2].values[-jumpDet:,:num],axis=0)-np.median(appl[2].values[:jumpDet,:num],axis=0)).sum())+np.absolute((np.median(appl[3].values[-jumpDet:,:num],axis=0)-np.median(appl[3].values[:jumpDet,:num],axis=0)).sum())
            Comp = DeviceFitValue(appl[2],appl[3],ts,time[0],time[1],'off')/dOff
        if np.isnan(Comp):
            print 'nan found'
        else:
            applist.append([appl[4],time[3],Comp])
    applist = sorted(applist,key=lambda ap:ap[-1])
    num = 5
    if len(applist) < 5:
        num = len(applist)
    ret = []
    idList = []
    n=0
    ii=0
    while n < num:
        inList = False
        for iden in idList:
            if applist[ii][0] == iden:
                inList = True
        if inList == False:
            ret.append(applist[ii])
            idList.append(applist[ii][0])
            n += 1
        ii += 1
        
    return [time[2].value,appliances[0][5],ret]

def Process(chunk):
    output = []
    for data in chunk:
        output.append(MatchEvent(data[0],data[1],data[2]))
    return output

def main():
    featureLength = 20
    tdelta = 20

    f2Classify = 'H4_09-19_testing.h5'
    ts = AF.ElectricTimeStream('data/hdf5storage/'+f2Classify)
    ts.ExtractComponents()

    temp = []
    with open('classified_series/simple_features_'+f2Classify[:-2]+'csv','rb') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            temp.append(row)
    temp = temp[1:]
    times = [[pd.to_datetime(int(float(time[0]))-tdelta,unit='s'),pd.to_datetime(int(float(time[0]))+tdelta,unit='s'),pd.to_datetime(int(float(time[0]))),('on' if (float(time[1])+float(time[3])) > 0 else 'off')] for time in temp]

    numAppl=-1
    applist = []
    with open('data/eventtimes.csv') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            applist.append(row)
            numAppl +=1
    applist = applist[1:]
    appsInHouse = []
    for ii,app in enumerate(applist):
       if app[0] == f2Classify[:2]:
           appsInHouse.append(ii)

    appliances = [AF.Appliance(ii) for ii in appsInHouse]
    for app in appliances:
        app.getOn(featureLength)
        app.getOff(featureLength)

    #print 'Creating streams'
    #streams = [(ts.l1.loc[(ts.l1.index > pd.to_datetime(time[0])-timedelta(seconds=30)) & (ts.l1.index < pd.to_datetime(time[1])+timedelta(seconds=30))],ts.l2.loc[(ts.l2.index < pd.to_datetime(time[1])+timedelta(seconds=30)) & (ts.l2.index < pd.to_datetime(time[1])+timedelta(seconds=30))]) for time in times]
    print 'Creating features'
    features = [(app.l1_on,app.l2_on,app.l1_off,app.l2_off,app.id,app.house) for app in appliances]

    print 'Classifying features.'
    output = []
    #inputs = [(group[0],features,group[1]) for group in zip(times,streams)]

    #pool = multiprocessing.Pool()
    #number = len(inputs)
    #div = multiprocessing.cpu_count()
    #iterable = []
    #for i in xrange(div):
    #    iterable.append(inputs[int(number*i/float(div)):int(number*(i+1)/float(div))])
    #outputs = pool.map(Process,iterable)
    #output = []
    #for out in outputs:
    #    for data in out:
    #        output.append(data)
    for ii,time in enumerate(times):
        print ii
        #if ii >= 10:
        #    break
        stream = (ts.l1.loc[(ts.l1.index > pd.to_datetime(time[0])-timedelta(seconds=30)) & (ts.l1.index < pd.to_datetime(time[1])+timedelta(seconds=30))],ts.l2.loc[(ts.l2.index < pd.to_datetime(time[1])+timedelta(seconds=30)) & (ts.l2.index < pd.to_datetime(time[1])+timedelta(seconds=30))])
        inp = (time,features,stream)
        output.append(MatchEvent(inp))
    with open(f2Classify[:-2]+'pickle','w+b') as f:
        cPickle.dump(output,f)


if __name__ == '__main__':
    main()
