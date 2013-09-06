import ApplianceFeatures as AF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
import csv
from datetime import timedelta
import copy
import cPickle
from multiprocessing import Pool
import scipy.signal as signal

def CompareStripped(appl,timestream,FeatureType=None):
    if FeatureType is None:
        FeatureType='on'
    timeAvg = '1S'
    num=12
    if FeatureType == 'on':
        feature = (appl.l1_on.resample(rule=timeAvg).values[:,:12],appl.l2_on.resample(rule=timeAvg).values[:,:12])
    elif FeatureType == 'off':
        feature = (appl.l1_off.resample(rule=timeAvg).values[:,:12],appl.l2_off.resample(rule=timeAvg).values[:,:12])
    elif FeatureType == 'ss':
        feature = (appl.l1_ss,appl.l2_ss)
    else:
        raise Exception('FeatureType not understood')
    ts = (timestream.l1.resample(rule=timeAvg).values[:,:12],timestream.l2.resample(rule=timeAvg).values[:,:12])
    timeIdx = (timestream.l1.resample(rule=timeAvg).index[:-len(feature[0])+1],timestream.l2.resample(rule=timeAvg).index[:-len(feature[1])+1])
    #l1Comp = pd.Series([np.mean([np.linalg.norm(feature[0][col].values-ts[0][col][ts[0].index[ii]:ts[0].index[ii+len(feature[0])-1]].values)**2/len(feature) for col in ts[0].columns]) for ii in xrange(len(ts[0].index)-len(feature[0])+1)],ts[0].index[:-len(feature[0])+1])
    #l2Comp = pd.Series([np.mean([np.linalg.norm(feature[1][col].values-ts[1][col][ts[1].index[ii]:ts[1].index[ii+len(feature[1])-1]].values)**2/len(feature) for col in ts[1].columns]) for ii in xrange(len(ts[1].index)-len(feature[1])+1)],ts[1].index[:-len(feature[1])+1])

    print 'Values'
    print ts[0].shape,feature[0].shape
    print ts[1].shape,feature[1].shape
    l1Comp = pd.Series((signal.convolve2d(feature[0],feature[0],mode='valid')[0]+signal.convolve2d(ts[0],ts[0],mode='valid')[0]
                               -2*signal.convolve2d(ts[0],feature[0],mode='valid')[0])/(feature[0].shape[0]*feature[0].shape[1]),timeIdx[0])
    l2Comp = pd.Series((signal.convolve2d(feature[1],feature[1],mode='valid')[0]+signal.convolve2d(ts[1],ts[1],mode='valid')[0]
                               -2*signal.convolve2d(ts[1],feature[1],mode='valid')[0])/(feature[1].shape[0]*feature[1].shape[1]),timeIdx[1])
    return pd.DataFrame({'l1Comp':l1Comp,'l2Comp':l2Comp})

def CompareToFeature(appl,timestream,start,stop,FeatureType=None):
    if FeatureType is None:
        FeatureType='on'
    prebaseTime = pd.to_datetime(start)-timedelta(seconds=10),pd.to_datetime(start)-timedelta(seconds=5)
    postbaseTime = pd.to_datetime(stop)+timedelta(seconds=5),pd.to_datetime(stop)+timedelta(seconds=10)
    l1pre = (timestream.l1.index > prebaseTime[0]) & (timestream.l1.index < prebaseTime[1])
    l2pre = (timestream.l2.index > prebaseTime[0]) & (timestream.l2.index < prebaseTime[1])
    l1post = (timestream.l1.index > postbaseTime[0]) & (timestream.l1.index < postbaseTime[1])
    l2post = (timestream.l2.index > postbaseTime[0]) & (timestream.l2.index < postbaseTime[1])
    if FeatureType == 'on':
        baseline1 = timestream.l1.loc[l1pre].sum()/len(timestream.l1.loc[l1pre])
        baseline2 = timestream.l2.loc[l2pre].sum()/len(timestream.l2.loc[l2pre])
    elif FeatureType == 'off':
        baseline1 = timestream.l1.loc[l1post].sum()/len(timestream.l1.loc[l1post])
        baseline2 = timestream.l2.loc[l2post].sum()/len(timestream.l2.loc[l2post])
    elif FeatureType == 'ss':
        baseline1 = (timestream.l1.loc[l1post].sum()/len(timestream.l1.loc[l1post])
                     +timestream.l1.loc[l1pre].sum()/len(timestream.l1.loc[l1pre]))/2
        baseline2 = (timestream.l2.loc[l2post].sum()/len(timestream.l2.loc[l2post])
                     +timestream.l2.loc[l2pre].sum()/len(timestream.l2.loc[l2pre]))/2
    send = copy.deepcopy(timestream)
    send.TruncateToWindow(start,stop)
    print 'send'
    print send.l1.index[0],send.l1.index[-1]
    print send.l2.values.shape,appl.l2.values.shape
    send.l1 -= baseline1
    send.l2 -= baseline2
    return CompareStripped(appl,send,FeatureType)

def DeviceFitValue(appl,timestream,start,stop,FeatureType=None):
    if FeatureType is None:
        FeatureType='on'
    data = CompareToFeature(appl,timestream,start,stop,FeatureType)
    return np.amin(data.l1Comp)+np.amin(data.l2Comp)

def MatchEvent((time,appliances,ts)):
    applist = []
    for appl in appliances:
        onComp = DeviceFitValue(appl,ts,time[0],time[1],FeatureType='on')
        offComp = DeviceFitValue(appl,ts,time[0],time[1],FeatureType='off')
        applist.append([appl.id,('on' if onComp <= offComp else 'off'),(onComp if onComp <= offComp else offComp)])
    applist = sorted(applist,key=lambda ap:ap[-1])
    num = 5
    if len(applist) < 5:
        num = len(applist)
    return [time[2].value,appliances[0].house,applist[:num]]

def main():
    featureLength = 15
    tdelta = 20

    f2Classify = 'H4_09-18_testing.h5'
    ts = AF.ElectricTimeStream('data/hdf5storage/'+f2Classify)
    ts.ExtractComponents()

    temp = []
    with open('simple_features/simple_features_'+f2Classify[:-2]+'csv','rb') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            temp.append(row[0])
    temp = temp[1:]
    times = [[pd.to_datetime(int(float(time))-tdelta,unit='s'),pd.to_datetime(int(float(time))+tdelta,unit='s'),pd.to_datetime(int(float(time)))] for time in temp]

    numAppl=-1
    with open('data/eventtimes.csv') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            numAppl +=1

    appliances = [AF.Appliance(ii) for ii in xrange(9)]
    for app in appliances:
        app.getOn(featureLength)
        app.getOff(featureLength)

    appliances2 = appliances[:10]
    times2 = times[:10]

    print 'Classifying features.'
    output = []
    #pool = Pool()
    inputs = [(time,appliances2,ts) for time in times2]
    #output = pool.map(MatchEvent,inputs)
    for inp in inputs:
        output.append(MatchEvent(inp))
    with open(f2Classify[:-2]+'pickle','w+b') as f:
        cPickle.dump(output,f)


if __name__ == '__main__':
    main()
