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

def CompareStripped(appl,timestream,FeatureType=None):
    if FeatureType is None:
        FeatureType='on'
    timeAvg = '2S'
    if FeatureType == 'on':
        feature = (appl.l1_on.resample(rule=timeAvg),appl.l2_on.resample(rule=timeAvg))
    elif FeatureType == 'off':
        feature = (appl.l1_off.resample(rule=timeAvg),appl.l2_off.resample(rule=timeAvg))
    elif FeatureType == 'ss':
        feature = (appl.l1_ss,appl.l2_ss)
    else:
        raise Exception
    ts = (timestream.l1.resample(rule=timeAvg),timestream.l2.resample(rule=timeAvg))
    l1Comp = pd.Series([np.mean([np.linalg.norm(feature[0][col].values-ts[0][col][ts[0].index[ii]:ts[0].index[ii+len(feature[0])-1]].values)**2 for col in ts[0].columns]) for ii in xrange(len(ts[0].index)-len(feature[0])+1)],ts[0].index[:-len(feature[0])+1])
    l2Comp = pd.Series([np.mean([np.linalg.norm(feature[1][col].values-ts[1][col][ts[1].index[ii]:ts[1].index[ii+len(feature[1])-1]].values)**2 for col in ts[1].columns]) for ii in xrange(len(ts[1].index)-len(feature[1])+1)],ts[1].index[:-len(feature[1])+1])
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
    send.l1 -= baseline1
    send.l2 -= baseline2
    return CompareStripped(appl,send,FeatureType)

def DeviceFitValue(appl,timestream,start,stop,FeatureType=None):
    if FeatureType is None:
        FeatureType='on'
    data = CompareToFeature(appl,timestream,start,stop,FeatureType)
    return np.amin(data.l1Comp)+np.amin(data.l2Comp)

def MatchEvent(args):
    time = args[0]
    appliances = args[1]
    ts = args[2]
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
    delta = 15

    f2Classify = 'H1_07-09_testing.h5'
    ts = AF.ElectricTimeStream('data/hdf5storage/'+f2Classify)
    ts.ExtractComponents()

    temp = []
    with open('simple_features_'+f2Classify[:-2]+'csv','rb') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            temp.append(row[0])
    temp = temp[1:]
    times = [[pd.to_datetime(int(float(time))-delta,unit='s'),pd.to_datetime(int(float(time))+delta,unit='s'),pd.to_datetime(int(float(time)))] for time in temp]

    numAppl=-1
    with open('data/eventtimes.csv') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            numAppl +=1

    appliances = [AF.Appliance(ii) for ii in xrange(97)]
    for app in appliances:
        app.getOn(featureLength)
        app.getOff(featureLength)

    appliances2 = appliances[:10]
    times2 = times[:10]

    print 'Classifying features.'
    output = []
    pool = Pool()
    inputs = [(time,appliances2,ts) for time in times2]
    output = pool.map(MatchEvent,inputs,chunksize=1)
    #for inp in inputs:
    #    output.append(MatchEvent(inp))
    with open(f2Classify[:-2]+'pickle','w+b') as f:
        cPickle.dump(output,f)


if __name__ == '__main__':
    main()
