#!/usr/bin/env python
# encoding: utf-8
"""
ApplianceFeatures.py

Extract features from Belkin energy training data.

Created by Adam N Morgan on 2013-08-25.
"""

import sys
import os
import unittest
import numpy as np
from matplotlib.pylab import *
from scipy import io
from cmath import phase
import sys
import cPickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import gc

def _updateFileTimes():
    '''
    Loop through the HDF5 training files and build a database of the start and
    stop times of each, to allow for lookup of the relevant data file when 
    investigating an individual appliance instance.  Save them as a csv file.
    '''
    hdf5list = glob.glob("data/hdf5storage/H?_??-??.h5")
    startlist = []
    stoplist = []
    matlist = []
    matgloblist = glob.glob("data/H?/Tagged_Training*.mat")
    for hdf5file in hdf5list:
        mat_matches = []
        housedatestr = hdf5file.split('/')[-1].replace('.h5','')
        housestr,datestr = housedatestr.split('_')
        for matfile in matgloblist:
            if (housestr in matfile) and (datestr.replace('-','_') in matfile):
                mat_matches.append(matfile)
        assert len(mat_matches) == 1
        matlist.append(mat_matches[0]) 
        dfl1, dfl2, dfhf = LoadHDF5(hdf5file)
        # populate the start and stop times of each file by adding a 2 minute
        # window to the min and max index value for one of the data frames
        # (the resolution isnt high enough for it to matter which one)
        startlist.append(pd.to_datetime(dfhf.index.min()-timedelta(minutes=2)))
        stoplist.append(pd.to_datetime(dfhf.index.max()+timedelta(minutes=2)))
    filedf = pd.DataFrame({
    "filename":hdf5list,
    "matfile":matlist,
    "filestart":startlist,
    "filestop":stoplist
    })
    
    filedf.to_csv("data/filetimes.csv")
    print "Saved to data/filetimes.csv"
    # outh5 = "data/tagging.h5"
    # print "Saving data frames to hdf5: {}".format(outh5)
    # store = pd.HDFStore(outh5)
    # store['filetimes'] = filedf
    # store.close()
    return filedf
    
def _updateEventTimes():
    '''
    Grab the start/stop times of each training event from allevents.csv;
    Store as a pandas dataframe, and store it in the tagging.h5 file
    
    No longer implemented. the hdf5 files were too large. just sticking with csv.
    
    '''
    raise NotImplementedError
    eventdf=pd.read_csv("data/allevents.csv",skiprows=4,names=['house','id','name','start','stop'])
    outh5 = "data/tagging.h5"
    print "Saving data frames to hdf5: {}".format(outh5)
    store = pd.HDFStore(outh5)
    store['eventtimes'] = eventdf
    store.close()
    return eventdf
    # grab the file names from the pkl
    
def LoopAndStore(globpath,clobber=False,save_all=False):
    '''
    Given a globpath such as 'data/H?/Tagged_Training_*.mat', loop through 
    each file and store it as an HDF5 file, if it has not been already.
    
    Other keywords are the same as in GetDataFrames.
    '''
    matlist = glob.glob(globpath)
    for mat in matlist:
        a=0 
        b=0 
        c=0
        a,b,c = GetDataFrames(mat,clobber=clobber,save_all=save_all)
        gc.collect()

def LoadHDF5(hdf5file):
    '''
    hdf5file is the name of the .h5 data file generated by GetDataFrames 
    '''
    if not os.path.exists(hdf5file):
        print "Cannot find {}. Returning nothing.".format(hdf5file)
        return
    if not hdf5file[-3:] == '.h5':
        print "File {} does not end in .h5; Returning nothing.".format(hdf5file)
    dfl1 = pd.read_hdf(hdf5file,'dfl1')
    dfl2 = pd.read_hdf(hdf5file,'dfl2')
    dfhf = pd.read_hdf(hdf5file,'dfhf')
    return dfl1, dfl2, dfhf
    
def GetDataFrames(filename,clobber=False,save_all=False):
    '''
    Load the dataframes for a particular house/date.
    Given the matlab file name, load up the data frames from data/hdf5storage/
    
    If the data have not been saved in hdf5 format yet, then this will 
    convert the matlab data into hdf5 files. 
    
    If clobber == True, then it will reload and resave as HDF5 even if the 
    HDF5 file already exists.
    
    If it is tagged training data and save_all = False, then it will strip 
    off any data < 1000s before the first tagged start time and any data 
    > 1000s after the last tagged end time
    
    '''
    gc.collect()
    
    if not os.path.exists(filename):
        raise IOError('Cannot find the specified filename')
    
    # os.path.basename("data/H1/Tagged_Training_04_13_1334300401.mat").split('_')
    # ['Tagged', 'Training', '04', '13', '1334300401.mat']        
    pathlist = os.path.basename(filename).split('_')
    
    # determine whether we are testing or training data, or neither
    if str(pathlist[0]) == "Testing":
        teststr = "_testing"
        date_str = str(pathlist[1]) + '-' + str(pathlist[2])
        if not save_all:
            print "{} does not appear to be a training data set. Not truncating.".format(filename)
            save_all = True
    elif str(pathlist[0]) == "Tagged":
        teststr = ''
        date_str = str(pathlist[2]) + '-' + str(pathlist[3])
    else:
        raise IOError("Cannot parse the file {}. Is it a matlab file?".format(filename))
        
    # assuming directory structure data/H1/Tagged_Training_04_13_1334300401.mat
    house_id = os.path.dirname(filename).split('/')[-1]
    # output 
    outdir = os.path.dirname(filename).rstrip(house_id) + 'hdf5storage/'
    # data/buffers
    if not os.path.exists(outdir):
        try:
            os.mkdir(outdir)
        except:
            print "Cannot create directory for hdf5 files."
            return None
    
    outh5 = outdir + house_id + '_' + date_str + teststr + '.h5'
    # outh5l2 = outdir + house_id + '_' + date_str + 'l2.h5'
    # outh5hf = outdir + house_id + '_' + date_str + 'hf.h5'
    
    if not os.path.exists(outh5) or clobber == True:
        # if the buffer isn't already saved as a pickle file, or if you're 
        # reloading it, reload the buffer
        # Read in the matlab datafile
        if not os.path.exists(outh5):
            print "Saved data does not exist for {}: Converting from matlab binary".format(outh5)
        elif clobber:
            print "Saved data already exists for {} but clobber == True; overwriting".format(outh5)
        buf = io.loadmat(filename)['Buffer']
    else:
        print "Saved data for {} already exists.\n>Loading {}".format(filename,outh5)
        return LoadHDF5(outh5)


    # These are the voltage and current time-series FFT fundamental and first 5
    # harmonics measurements.
    LF1V = buf['LF1V'][0][0]                    # Nx6, N is number of time steps
    LF1I = buf['LF1I'][0][0]                    # Nx6, N is number of time steps
    LF2V = buf['LF2V'][0][0]                    # Nx6, N is number of time steps
    LF2I = buf['LF2I'][0][0]                    # Nx6, N is number of time steps

    # Time stamps, these are floats
    L1_TimeTicks = buf['TimeTicks1'][0][0].flatten()
    L2_TimeTicks = buf['TimeTicks2'][0][0].flatten()

    # Spectrogram of high frequency noise. Each FFT vector is computed every 1.0667 secs
    HF = buf['HF'][0][0]                        # 4096xN, N is number of FFT vectors
    # Time stamps for the HF spectrogram, also floats
    HF_TimeTicks = buf['TimeTicksHF'][0][0].flatten()


    # make the MultiIndex; 6 instances each of v and i
    typelist = ['v']*6 + ['i']*6
    numlist =['0','1','2','3','4','5']*2
    arrays = [typelist,numlist]
    tuples = zip(*arrays)
    compindex = pd.MultiIndex.from_tuples(tuples,names=['component','harmonic'])
    
    # build up multiindex data frame 
    dfL1fullarr = np.zeros((len(L1_TimeTicks),12),dtype="complex64")
    dfL1fullarr[:,0:6] = LF1V
    dfL1fullarr[:,6:12] = LF1I
    dfL1 = pd.DataFrame(dfL1fullarr,index = (L1_TimeTicks*1e9).astype('datetime64[ns]'), columns = compindex)

    dfL2fullarr = np.zeros((len(L2_TimeTicks),12),dtype="complex64")
    dfL2fullarr[:,0:6] = LF2V
    dfL2fullarr[:,6:12] = LF2I
    dfL2 = pd.DataFrame(dfL2fullarr,index = (L2_TimeTicks*1e9).astype('datetime64[ns]'), columns = compindex)
    
    # 4096 columns, rows indexed by time
    dfHF=pd.DataFrame(HF.transpose(),index=(HF_TimeTicks*1e9).astype('datetime64[ns]'),columns=np.arange(4096))


    # dfL1.real.plot()
    # raise Exception
    # # plot lines showing the start/stop times
    # for appliance_id in taggingInfo_dict:
    #     appliance_name = taggingInfo_dict[appliance_id]['ApplianceName']
    #     for interval in taggingInfo_dict[appliance_id]['OnOffSeq']:
    #         plt.vlines(interval[0],0,10000,color='red')
    #         plt.vlines(interval[1],0,10000,color='blue')
    #         plt.text(interval[0],100,appliance_name+'+',color='red')
    #         plt.text(interval[1],500,appliance_name+'-',color='blue')
    # plt.show()

    # Truncate the data down to get rid of unnecessary info    
    if not save_all:
        # Parse the tagged info for appliance identification
        # save them in a pkl file for later usage
        dictpkl = 'file_specific_tags.pkl'
        tagdict = load(dictpkl)
        if tagdict == None:
            tagdict = {}
        taggingInfo_dict = parse_tagging_info(buf["TaggingInfo"][0][0])
        
        tagdict.update({filename:taggingInfo_dict})
        save(tagdict,dictpkl,clobber=True)
        
        # Loop through the tagging info to identify the time window of relevance 
        #  in order to see if we can truncate the data for storage 
        interval_starts = []
        interval_stops = []
        for appliance_id in taggingInfo_dict:
            appliance_name = taggingInfo_dict[appliance_id]['ApplianceName']
            for interval in taggingInfo_dict[appliance_id]['OnOffSeq']:
                interval_starts.append(interval[0])
                interval_stops.append(interval[1])
    
        truncate_start = np.min(interval_starts) - 1000
        truncate_stop = np.max(interval_stops) + 1000
        # MUST convert these to the same datetime in pandas; having weird issues
        # with timezones if i did it the old way.
        truncate_start = pd.to_datetime([truncate_start],unit='s')
        truncate_stop = pd.to_datetime([truncate_stop],unit='s')

        
        print "Truncating the data to the time windows of relevance: {}-{}".format(str(truncate_start),str(truncate_stop))
        
        dfL1 = dfL1.loc[(dfL1.index < truncate_stop) & (dfL1.index > truncate_start)]
        dfL2 = dfL2.loc[(dfL2.index < truncate_stop) & (dfL2.index > truncate_start)]
        dfHF = dfHF.loc[(dfHF.index < truncate_stop) & (dfHF.index > truncate_start)]
        
        
    print "Saving data frames to hdf5: {}".format(outh5)
    store = pd.HDFStore(outh5)
    store['dfl1'] = dfL1
    store['dfl2'] = dfL2
    store['dfhf'] = dfHF
    
    store.close()
    
    return dfL1, dfL2, dfHF

def rollingavg(arr,binsize):    
    n_indices = len(arr) - binsize
    outavg = np.array(n_indices)
    first_ind = 0 
    last_ind = binsize-1
    current_sum = np.sum(arr[0:binsize])
    for ind in np.arange(n_indices):
        current_sum = current_sum - arr[first_ind] + arr[last_ind]
        outavg[ind] = current_sum
        first_ind += 1
        last_ind += 1


class ElectricTimeStream:
    '''
    Class for any timeseries of electrical data saved to an HDF5 file using
    GetDataFrames().  The nominal data frames are the 6 voltage and current 
    components for the two channels, stored in self.dfl1 and self.dfl2, and 
    the high frequency information, stored in self.dfhf.
    
    To extract the real, imaginary, amplitude, and pf components, run
    ExtractComponents(). This will create two new data frames, self.l1 and
    self.l2, which contain these components. You can index them as follows:
    
    import ApplianceFeatures as ap
    # Read in Electric Time Stream
    datafilename = "H1_07-09_testing.h5"
    ts = ap.ElectricTimeStream("data/hdf5storage/" + datafilename)
    # Extract Components on the ap object
    ts.ExtractComponents()
    
    # Get the timeseries of the 6 real components from channel l1:
    ts.l1['real']  # or ts.l1.real
    
    # Get the timeseries of the 3rd imaginary component from l2:
    ts.l2['imag']['2']
    
    # Get the timeseries of the real components between 9:30 and 9:48 pm
    subts = ts.l1['real'].loc[(ts.l1.index > "2012-10-22 21:30:00") & (ts.l1.index < "2012-10-22 21:48:00")]
    
    # plot all of the components of a data frame
    subts.plot()
    
    '''
    def __init__(self,hdf5file=None):
        self.hdf5file = hdf5file
    
    def _loadHDF5(self):
        if self.hdf5file == None:
            print "Need to assign location of self.hdf5 to load it."
        elif not os.path.exists(self.hdf5file):
            print "Path to hdf5 file {} does not exist. Not loading.".format({self.hdf5file})
        else:
            dfL1, dfL2, dfHF = LoadHDF5(self.hdf5file)
            self.dfl1 = dfL1
            self.dfl2 = dfL2
            self.dfhf = dfHF
            print "Loaded {}".format(self.hdf5file)

        
    def ExtractComponents(self):
        
        if 'i0' in self.dfl1:
            print "You have an outdated HDF5 data file. Grab the new one from Adam when you can."
            print "Still extracting components.."
            LF1I = np.array(self.dfl1.loc[:,['i0','i1','i2','i3','i4','i5']],dtype="complex64")
            LF1V = np.array(self.dfl1.loc[:,['v0','v1','v2','v3','v4','v5']],dtype="complex64")

            LF2I = np.array(self.dfl2.loc[:,['i0','i1','i2','i3','i4','i5']],dtype="complex64")
            LF2V = np.array(self.dfl2.loc[:,['v0','v1','v2','v3','v4','v5']],dtype="complex64")
        elif 'i' in self.dfl1:
            print "Extracting components."
            LF1I = np.array(self.dfl1['i'])
            LF1V = np.array(self.dfl1['v'])
            
            LF2I = np.array(self.dfl2['i'])
            LF2V = np.array(self.dfl2['v'])
            
        else:
            print "Malformed hdf5 file. Cannot extract data."
            return    
        

        # Calculate power (by convolution)
        L1_P = LF1V * LF1I.conjugate()      # Nx6, N is number of time steps
        L2_P = LF2V * LF2I.conjugate()
        # #
        L1_ComplexPower = L1_P#.sum(axis=1)          # length N, N is number of time steps
        L2_ComplexPower = L2_P#.sum(axis=1)          # length N, N is number of time steps
        # 
        # # Extract components
        L1_Real = L1_ComplexPower.real
        L1_Imag = L1_ComplexPower.imag
        L1_Amp  = abs(L1_ComplexPower)
        L2_Real = L2_ComplexPower.real
        L2_Imag = L2_ComplexPower.imag
        L2_Amp  = abs(L2_ComplexPower)
        # # Power Factor 
        L1_Pf = np.cos(np.angle(L1_P))
        L2_Pf = np.cos(np.angle(L2_P))
        
        # build up the multiindex; 6 instances each of real, image, amp, pf
        typelist=['real']*6 + ['imag']*6 + ['amp']*6 + ['pf']*6
        numlist =['0','1','2','3','4','5']*4
        arrays = [typelist,numlist]
        tuples = zip(*arrays)
        compindex = pd.MultiIndex.from_tuples(tuples,names=['component','harmonic'])
        
        # build up multiindex data frame 
        dfl1fullarr = np.zeros((len(self.dfl1.index),24),dtype='float64')
        dfl1fullarr[:,0:6] = L1_Real
        dfl1fullarr[:,6:12] = L1_Imag
        dfl1fullarr[:,12:18] = L1_Amp
        dfl1fullarr[:,18:24] = L1_Pf
        self.l1 = pd.DataFrame(dfl1fullarr,index = self.dfl1.index, columns = compindex)
      
        dfl2fullarr = np.zeros((len(self.dfl2.index),24),dtype='float64')
        dfl2fullarr[:,0:6] = L2_Real
        dfl2fullarr[:,6:12] = L2_Imag
        dfl2fullarr[:,12:18] = L2_Amp
        dfl2fullarr[:,18:24] = L2_Pf
        self.l2 = pd.DataFrame(dfl2fullarr,index = self.dfl2.index, columns = compindex)
        
        
    def Truncate(self,newstart,newstop):
        newstart = pd.to_datetime(newstart)
        newstop = pd.to_datetime(newstop)
        dflist = ['dfl1','dfl2','dfhf','l1','l2']
        print "Truncating data frames to new window: [{},{}]".format(newstart,newstop)
        for att in dflist:
            try:
                truncated_df = getattr(self,att)
                truncated_df = truncated_df.loc[(truncated_df.index >= newstart) & (truncated_df.index <= newstop)]
                setattr(self,att,truncated_df)
                print "Truncated {}.".format(att)
                truncated_df = None
            except:
                print "Instance of this object does not (yet) have {}. Not Truncating it.".format(att)

class Appliance(ElectricTimeStream): 
    '''
    Appliance class, a subclass of the ElectricTimeStream.  Given the ID (row) 
    number of the tagged instance in the eventtimes.csv file, it will identify 
    the hdf5file containing the data for this appliance, load it, truncate it 
    down to its start/stop times, and then extract its components."
    
    Sample Usage:
    
    In [239]: import ApplianceFeatures as af

    In [240]: app=af.Appliance(3)
    Loaded data/hdf5storage/H1_04-13.h5
    Truncating data frames to new window: [2012-04-13 22:24:26.990000,2012-04-13 22:26:34.870000]
    Truncated dfl1.
    Truncated dfl2.
    Truncated dfhf.
    Instance of this object does not have l1. Not Truncating it.
    Instance of this object does not have l2. Not Truncating it.
    Extracting components.
    Finished loading the instance of Washer for house H1 starting at 2012-04-13 22:24:26.990000.

    In [242]: app.house
    Out[242]: 'H1'

    In [243]: app.name
    Out[243]: 'Washer'

    In [244]: app.l1.real['0'].plot() # plot 0th component of the real timestream for l1
    Out[244]: <matplotlib.axes.AxesSubplot at 0x1ae45ef0>
    
    '''
    def __init__(self,eventid):    
        # load the times for each training file
        filedf = pd.read_csv('data/filetimes.csv',parse_dates=['filestart','filestop'],index_col=0)
        # load the start and stop times for each training event
        eventdf = pd.read_csv('data/eventtimes.csv')

        # find the hdf5 file that contains the training event
        self.start = pd.to_datetime(eventdf.loc[eventid]['start'],unit='s')
        self.stop = pd.to_datetime(eventdf.loc[eventid]['stop'],unit='s')
        self.house = eventdf.loc[eventid]['house']
        self.name = eventdf.loc[eventid]['name']
        
        row = filedf.loc[(filedf.filestart < self.start) & (filedf.filestop > self.stop)]
        if len(row) == 0:
            print "Could not find this training event in a datafile time window!"
        elif len(row) > 1:
            print "Found this training event in multiple datafile time windows; something is wrong"
        else:
            self.hdf5file = row.loc[:,'filename'].values[0]
            self._loadHDF5()
            self.Truncate(self.start,self.stop)
            self.ExtractComponents()
            print "Finished loading the instance of {} for house {} starting at {}.".format(self.name,self.house,self.start)
        
    def blah(self):
        # going to make class Appliance utilizing chris's event detection
        raise NotImplementedError
        # plot lines showing the start/stop times
        for appliance_id in taggingInfo_dict:
            appliance_name = taggingInfo_dict[appliance_id]['ApplianceName']
            for interval in taggingInfo_dict[appliance_id]['OnOffSeq']:
                plt.vlines(interval[0],0,10000,color='red')
                plt.vlines(interval[1],0,10000,color='blue')
                plt.text(interval[0],100,appliance_name+'+',color='red')
                plt.text(interval[1],500,appliance_name+'-',color='blue')
        plt.show()
        raise Exception 
    
        for appliance_id in taggingInfo_dict.keys():
            appliance_name = taggingInfo_dict[appliance_id]['ApplianceName']
            for interval in taggingInfo_dict[appliance_id]['OnOffSeq']:
            
                # grab the manually logged info for when the start/stop interval was
                # probably cannot trust this too much
                start_time = interval[0] - 300
                end_time = interval[1] + 300
            
                # determine the indices that contain the interval start_time to end_time
                a = HF_TimeTicks>=start_time
                b = HF_TimeTicks<=end_time
                HF_TimeTicks_indices = a*b # boolean of true/false values 
            
                HF_start_index = where(HF_TimeTicks_indices)[0].min()
                HF_end_index = where(HF_TimeTicks_indices)[0].max()+1
                HF_TimeTicks_window = HF_TimeTicks[HF_start_index:HF_end_index]
            
                num_HF_timestamps = HF_end_index - HF_start_index

                # Apply the time series window to the L1 data
                # Determine the indices of the L1 array that contains the HF window
                a = L1_TimeTicks>=HF_TimeTicks_window[0]
                b = L1_TimeTicks<=HF_TimeTicks_window[-1]
                L1_TimeTicks_indices = a*b
                L1_start_index = where(L1_TimeTicks_indices)[0].min()
                L1_end_index = where(L1_TimeTicks_indices)[0].max()+1
                L1_TimeTicks_window = L1_TimeTicks[L1_start_index:L1_end_index]

                L1_Real_window = L1_Real[L1_start_index:L1_end_index]
                L1_Imag_window = L1_Imag[L1_start_index:L1_end_index]
                L1_Amp_window = L1_Amp[L1_start_index:L1_end_index]
                L1_Pf_window = L1_Pf[L1_start_index:L1_end_index]

                # Apply the time series window to the L2 data
                # should be the same window as L1
                a = L2_TimeTicks>=HF_TimeTicks_window[0]
                b = L2_TimeTicks<=HF_TimeTicks_window[-1]
                L2_TimeTicks_indices = a*b
                L2_start_index = where(L2_TimeTicks_indices)[0].min()
                L2_end_index = where(L2_TimeTicks_indices)[0].max()+1
                L2_TimeTicks_window = L2_TimeTicks[L2_start_index:L2_end_index]

                L2_Real_window = L2_Real[L2_start_index:L2_end_index]
                L2_Imag_window = L2_Imag[L2_start_index:L2_end_index]
                L2_Amp_window = L2_Amp[L2_start_index:L2_end_index]
                L2_Pf_window = L2_Pf[L2_start_index:L2_end_index]

                # shift so that the 0th position is at time 0
                HF_TimeTicks_window_shifted = HF_TimeTicks_window - HF_TimeTicks_window[0]
                L1_TimeTicks_window_shifted = L1_TimeTicks_window - HF_TimeTicks_window[0]
                L2_TimeTicks_window_shifted = L2_TimeTicks_window - HF_TimeTicks_window[0]


                off_window_start_time = HF_TimeTicks[HF_start_index] - HF_TimeTicks_window[0]
                off_window_end_time = HF_TimeTicks[HF_start_index+int(round(num_HF_timestamps/4.))] - HF_TimeTicks_window[0]

                on_window_start_time = HF_TimeTicks[HF_end_index-int(round(num_HF_timestamps/4.))] - HF_TimeTicks_window[0]
                on_window_end_time = HF_TimeTicks[HF_end_index] - HF_TimeTicks_window[0]


                off_average_spectrum = HF[:,HF_start_index:HF_start_index+int(round(num_HF_timestamps/4.))].sum(axis=1) / float(HF[:,HF_start_index:HF_start_index+int(round(num_HF_timestamps/4.))].shape[1])
                on_average_spectrum = HF[:,HF_end_index-int(round(num_HF_timestamps/4.)):HF_end_index].sum(axis=1) / float(HF[:,HF_end_index-int(round(num_HF_timestamps/4.)):HF_end_index].shape[1])
                diff_spectrum = on_average_spectrum - off_average_spectrum
            
                df_spec = pd.DataFrame({
                "off":off_average_spectrum,
                "on":on_average_spectrum,
                "diff":on_average_spectrum-off_average_spectrum 
                })

                dfL1 = pd.DataFrame({
                "real":L1_Real_window,
                "imag":L1_Imag_window,
                "amp":L1_Amp_window,
                "pf0":L1_Pf_window[:,0],
                "pf1":L1_Pf_window[:,1],
                "pf2":L1_Pf_window[:,2],
                "pf3":L1_Pf_window[:,3],
                "pf4":L1_Pf_window[:,4],
                "pf5":L1_Pf_window[:,5]
                }, index = L1_TimeTicks_window_shifted
                )
            
                dfL2 = pd.DataFrame({
                "real":L2_Real_window,
                "imag":L2_Imag_window,
                "amp":L2_Amp_window,
                "pf0":L2_Pf_window[:,0],
                "pf1":L2_Pf_window[:,1],
                "pf2":L2_Pf_window[:,2],
                "pf3":L2_Pf_window[:,3],
                "pf4":L2_Pf_window[:,4],
                "pf5":L2_Pf_window[:,5]
                }, index = L2_TimeTicks_window_shifted
                )
            
                dfL2.real.plot()
                dfL1.real.loc[(dfL2.index < 1334358010) & (dfL2.index > 1334354690)].plot()

                L1_L2_plot_data = ((L1_TimeTicks_window_shifted, L1_Real_window),
                                   (L1_TimeTicks_window_shifted, L1_Imag_window),
                                   (L1_TimeTicks_window_shifted, L1_Amp_window),
                                   (L2_TimeTicks_window_shifted, L2_Real_window),
                                   (L2_TimeTicks_window_shifted, L2_Imag_window),
                                   (L2_TimeTicks_window_shifted, L2_Amp_window),
                                   (L1_TimeTicks_window_shifted, L1_Pf_window[:,0]),
                                   (L1_TimeTicks_window_shifted, L1_Pf_window[:,1]),
                                   (L1_TimeTicks_window_shifted, L1_Pf_window[:,2]),
                                   (L1_TimeTicks_window_shifted, L1_Pf_window[:,3]),
                                   (L1_TimeTicks_window_shifted, L1_Pf_window[:,4]),
                                   (L1_TimeTicks_window_shifted, L1_Pf_window[:,5]),
                                   (L2_TimeTicks_window_shifted, L2_Pf_window[:,0]),
                                   (L2_TimeTicks_window_shifted, L2_Pf_window[:,1]),
                                   (L2_TimeTicks_window_shifted, L2_Pf_window[:,2]),
                                   (L2_TimeTicks_window_shifted, L2_Pf_window[:,3]),
                                   (L2_TimeTicks_window_shifted, L2_Pf_window[:,4]),
                                   (L2_TimeTicks_window_shifted, L2_Pf_window[:,5]))

                L1_L2_plot_labels = ("L1_Real", "L1_Imag", "L1_Amp", "L2_Real", "L2_Imag", "L2_Amp",
                    "L1_Pf0", "L1_Pf1", "L1_Pf2", "L1_Pf3", "L1_Pf4", "L1_Pf5",
                    "L2_Pf0", "L2_Pf1", "L2_Pf2", "L2_Pf3", "L2_Pf4", "L2_Pf5")


    
    
def parse_tagging_info(tagging_info_buffer):
    '''
    This function parses the weirdly-formatted tagged info into a more friendly
    dictionary. "tagged info" are time ranges for when an identified appliance is
    turned off (it is one in between the times given in the OnOffSeq)
    
    '''
    tagging_info_dict = {}
    for n in range(len(tagging_info_buffer)):
        id = tagging_info_buffer[n][0][0][0].astype(int)
        name = tagging_info_buffer[n][1][0][0][0].astype(str)
        on_time = tagging_info_buffer[n][2][0][0].astype(int)
        off_time = tagging_info_buffer[n][3][0][0].astype(int)
        if id not in tagging_info_dict.keys():
            tagging_info_dict[id] = {}
            tagging_info_dict[id]["ApplianceName"] = name
            tagging_info_dict[id]["OnOffSeq"] = [(on_time, off_time)]
        else:
            tagging_info_dict[id]["OnOffSeq"].append( (on_time, off_time) )
    return tagging_info_dict    
    
    
  

def load(pklpath):
    '''Given an input object and an output path, load a pickle file.'''
    if os.path.exists(pklpath):
        storefile=open(pklpath)
        loadedpkl = pickle.load(storefile)
        storefile.close()
        print "Loaded pickle file for this object."
        return loadedpkl
    else:
        print "Pickle file %s does not exist" % pklpath
        return None

def save(input,outpath,clobber=False):
    "Save an object as a pickle file."
    path_existed = os.path.exists(outpath)
    if path_existed and not clobber:
        print '%s already exists and clobber == False; not overwriting.' % (outpath)
        return
    else:
        storefile = open(outpath,'w')
        pickle.dump(input,storefile)
        storefile.close
        if not path_existed:
            print "No Pickle file existed, so one was created:"
            print outpath
        else:
            print "Overwrote Pickle file:"
            print outpath



class untitledTests(unittest.TestCase):
    def setUp(self):
        pass

if __name__ == '__main__':
    unittest.main()