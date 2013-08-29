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
import glob
import gc

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
    # appliance_id = "H1A8"  
    gc.collect()
    
    if not os.path.exists(filename):
        raise IOError('Cannot find the specified filename')
    
    # In [22]: os.path.basename("data/H1/Tagged_Training_04_13_1334300401.mat").split('_')
    # Out[22]: ['Tagged', 'Training', '04', '13', '1334300401.mat']        
    pathlist = os.path.basename(filename).split('_')
    
    
    if str(pathlist[0]) == "Testing":
        teststr = "_testing"
        date_str = str(pathlist[1]) + '-' + str(pathlist[2])
    elif str(pathlist[0]) == "Tagged":
        teststr = ''
        date_str = str(pathlist[2]) + '-' + str(pathlist[3])
    else:
        raise IOError
        
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
    
    outkey = house_id + '_' + date_str
    

        
    outh5 = outdir + house_id + '_' + date_str + teststr + '.h5'
    # outh5l2 = outdir + house_id + '_' + date_str + 'l2.h5'
    # outh5hf = outdir + house_id + '_' + date_str + 'hf.h5'
    
    if not os.path.exists(outh5) or clobber == True:
        # if the buffer isn't already saved as a pickle file, or if you're 
        # reloading it, reload the buffer
        # Read in the matlab datafile
        if not os.path.exists(outh5):
            print "Saved data does not exist for {}: Converting from matlab binary".format(outkey)
        elif clobber:
            print "Saved data already exists for {} but clobber == True; overwriting".format(outkey)
            
        buf = io.loadmat(filename)['Buffer']

    else:

        print "Saved data for {} already exists.\n>Loading {}".format(filename,outh5)
        
        dfl1 = pd.read_hdf(outh5,'dfl1')
        dfl2 = pd.read_hdf(outh5,'dfl2')
        dfhf = pd.read_hdf(outh5,'dfhf')
        return dfl1, dfl2, dfhf


    
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

    # Calculate power (by convolution)
    L1_P = LF1V * LF1I.conjugate()              # Nx6, N is number of time steps
    L2_P = LF2V * LF2I.conjugate()              # Nx6, N is number of time steps
    #
    L1_ComplexPower = L1_P.sum(axis=1)          # length N, N is number of time steps
    L2_ComplexPower = L2_P.sum(axis=1)          # length N, N is number of time steps

    # Extract components
    L1_Real = L1_ComplexPower.real
    L1_Imag = L1_ComplexPower.imag
    L1_Amp  = abs(L1_ComplexPower)
    L2_Real = L2_ComplexPower.real
    L2_Imag = L2_ComplexPower.imag
    L2_Amp  = abs(L2_ComplexPower)
    # Power Factor 
    L1_Pf = np.cos(np.angle(L1_P))
    L2_Pf = np.cos(np.angle(L2_P))


    # the timeticks for L1 and L2 are not in fact equal.. 
    # assert len(L1_TimeTicks) == len(L2_TimeTicks)
    # assert L1_TimeTicks[0] == L2_TimeTicks[0]
    # assert L1_TimeTicks[-1] == L2_TimeTicks[-1]
    
    dfL1 = pd.DataFrame({
    "v0":LF1V[:,0],
    "v1":LF1V[:,1],
    "v2":LF1V[:,2],
    "v3":LF1V[:,3],      
    "v4":LF1V[:,4], 
    "v5":LF1V[:,5],   
    "i0":LF1I[:,0],
    "i1":LF1I[:,1],
    "i2":LF1I[:,2],
    "i3":LF1I[:,3],      
    "i4":LF1I[:,4], 
    "i5":LF1I[:,5],
    }, index = (L1_TimeTicks*1e6).astype('datetime64[ms]'))
    
    dfL2 = pd.DataFrame({
    "v0":LF2V[:,0],
    "v1":LF2V[:,1],
    "v2":LF2V[:,2],
    "v3":LF2V[:,3],      
    "v4":LF2V[:,4], 
    "v5":LF2V[:,5],   
    "i0":LF2I[:,0],
    "i1":LF2I[:,1],
    "i2":LF2I[:,2],
    "i3":LF2I[:,3],      
    "i4":LF2I[:,4], 
    "i5":LF2I[:,5],
    }, index = (L2_TimeTicks*1e6).astype('datetime64[ms]'))
    
    # dfL1 = pd.DataFrame({
    # "real":L1_Real,
    # "imag":L1_Imag,
    # "amp":L1_Amp,
    # "pf0":L1_Pf[:,0],
    # "pf1":L1_Pf[:,1],
    # "pf2":L1_Pf[:,2],
    # "pf3":L1_Pf[:,3],
    # "pf4":L1_Pf[:,4],
    # "pf5":L1_Pf[:,5]
    # }, index = (L1_TimeTicks*1e6).astype('datetime64[ms]')
    # )
    # 
    # dfL2 = pd.DataFrame({
    # "real":L2_Real,
    # "imag":L2_Imag,
    # "amp":L2_Amp,
    # "pf0":L2_Pf[:,0],
    # "pf1":L2_Pf[:,1],
    # "pf2":L2_Pf[:,2],
    # "pf3":L2_Pf[:,3],
    # "pf4":L2_Pf[:,4],
    # "pf5":L2_Pf[:,5]
    # }, index = (L2_TimeTicks*1e6).astype('datetime64[ms]')
    # )
    
    
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

    
    
    # 4096 columns, rows indexed by time
    dfHF=pd.DataFrame(HF.transpose(),index=(HF_TimeTicks*1e6).astype('datetime64[ms]'),columns=np.arange(4096))

    if not save_all and "TaggingInfo" not in buf:
        print "{} does not appear to be a training data set. Not truncating.".format(filename)
        save_all = True
        
    if not save_all:
        # Parse the tagged info for appliance identification
        taggingInfo_dict = parse_tagging_info(buf["TaggingInfo"][0][0])
    
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
        truncate_start = pd.Timestamp(datetime.fromtimestamp(truncate_start))
        truncate_stop = pd.Timestamp(datetime.fromtimestamp(truncate_stop))
    
        print "Truncating the data to the time windows of relevance: {}-{}".format(truncate_start,truncate_stop)
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

        
class Appliance:
    def __init__(self,applianceid):    
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