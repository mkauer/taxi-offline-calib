#!/usr/bin/env python

import sys, os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle

sys.path.append('./modules')
from udaq_decoder_subruns import *
from utils import *
from log_parser import *
from fitting_funcs import *


SAVE = 1
DEBUG = 0
write_json = 1
skip_existing = 0
load_pickle = 0

taxi = 'taxi01'
#taxi = 'taxi02'

#runs = list(range(2139, 2443))
runs = [2149]



data_dir = './data/'+taxi
json_dir = './json/'+taxi
fits_dir = './mipfits/'+taxi

if not os.path.exists(json_dir):
    os.makedirs(json_dir)
if not os.path.exists(fits_dir):
    os.makedirs(fits_dir)


# go through the list of runs and fit the MIP
for run in runs:
    
    runNum = str(run).zfill(7)
    path = data_dir+'/run_'+runNum
    if not os.path.exists(path):
        print('run dir not found -->', path)
        continue
    
    # skip reprocessing if json already exists
    if os.path.exists(json_dir+'/'+runNum+'.json') and skip_existing:
        continue
    
    print('Run {0}'.format(runNum))
    
    logfiles = sorted(glob.glob(os.path.join(path, '*info.txt')))
    info = None
    for logfile in logfiles:
        info = getSpoolInfo(logfile, info)

    channels = info['channels']
    YEAR = int(info['date'].split('-')[0])

    picklefile = os.path.join(path, 'run_{0}_alldata.pkl'.format(runNum))
    if os.path.exists(picklefile) and load_pickle:
        print('loading pickle file')
        with open(picklefile, 'rb') as pfile:
            alldata = pickle.load(pfile)
    else:
        alldata = {}
        for chan in channels:
            runfiles = sorted(glob.glob(os.path.join(path, 'run_{0}_chan-{1}.bin*'.format(runNum, chan))))
            if not runfiles:
                print('WARNING: no binary files not found for run {0} chan-{1}'.format(runNum, chan))
                continue
            print('DECODING: chan-{0}'.format(chan))
            print('-----------------------------------------------')
            data = payloadReader(runfiles, debug=DEBUG)
            alldata[str(chan)] = data
            print()
        # save decoded data to pickle
        if alldata:
            print('saving pickle file')
            with open(picklefile, 'wb') as pfile:
                pickle.dump(alldata, pfile)
    
    channels = sorted([int(x) for x in alldata.keys()])
    #channels = [6]
    
    ### build the histograms as lists
    hists = {}
    peds = {}
    for chan in channels:
        chan = str(chan)
        hists[chan] = np.zeros(4096)
        peds[chan] = np.zeros(4096)
        for hit in alldata[chan]:
            ### 0 for hits - 1 for pedestal
            if hit[4] == 0: hists[chan][int(hit[1])] += 1
            if hit[4] == 1: peds[chan][int(hit[1])] += 1
        # remove saturation bin
        hists[chan][-1] = 0
        peds[chan][-1] = 0

    
    ### fit the MIP and save results
    adcfit_infos = {}
    pefit_infos = {}
    sname_adc = ''
    sname_pe = ''
    for chan in channels:
        chan = str(chan)
        if SAVE:
            sname_adc = fits_dir+'/run{0}_chan{1}_adc.png'.format(runNum, chan)
            sname_pe = fits_dir+'/run{0}_chan{1}_pe.png'.format(runNum, chan)
        print('Chan-{0}: fitting adc ...'.format(chan))
        adcfit_infos[chan] = adcFitMIP(ped=peds[chan],
                                       hist=hists[chan],
                                       config=info[chan],
                                       plot=False,
                                       title='Chan-{0}'.format(chan),
                                       save=sname_adc,
                                       debug=DEBUG
                                       )
        print('Chan-{0}: fitting pe ...'.format(chan))
        pefit_infos[chan] = peFitMIP(ped=peds[chan],
                                     hist=hists[chan],
                                     config=info[chan],
                                     plot=False,
                                     title='Chan-{0}'.format(chan),
                                     save=sname_pe,
                                     debug=DEBUG
                                     )
            
    ### write info and gain to json
    if write_json:
        print('writing info to json')
        data = {}
        data['runNum'] = runNum
        for chan in channels:
            ch = info[str(chan)]
            uid = ch['uid']
            data[uid] = {}
            data[uid]['channel'] = str(chan)
            data[uid]['voltage'] = ch['voltage']
            data[uid]['threshold'] = ch['threshold']
            data[uid]['temperature'] = ch['temperature'][1]
            data[uid]['temp_K'] = data[uid]['temperature'] + 273.15
            data[uid].update(adcfit_infos[str(chan)])
            data[uid].update(pefit_infos[str(chan)])
            
        with open(json_dir+'/'+runNum+'.json', 'w') as jsfile:
            json.dump(data, jsfile, separators=(', ', ': '), indent=4)

    print()

