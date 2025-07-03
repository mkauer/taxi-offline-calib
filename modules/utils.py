import os
import json
from math import sqrt, exp

# 2022-08-20

#CONFIG = '/home/mkauer/SCINT_HW/hub-data/bbb06-psl/scintconfig/uids_beacon.json'
#CONFIG = '/home/mkauer/SCINT_HW/hub-data/bbb06-psl/scintconfig/uids_taxi.json'

here = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(here, 'uids_taxi.json')


### get the uid mapping
def get_panel(uid):
    uid = str(uid)
    if not os.path.exists(CONFIG):
        print('ERROR: config not found -> {0}'.format(CONFIG))
        return uid
    with open(CONFIG) as jf:
        config = json.load(jf, object_pairs_hook=raise_on_duplicates)
    if uid in config:
        return config[uid]['name']
    else:
        return uid


### for checking duplicates in get_panel() json load
def raise_on_duplicates(ordered_pairs):
    # reject duplicate keys in json
    # https://stackoverflow.com/a/14902564
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("Duplicate json key: \'{0}\'".format(k))
            #print('Duplicate json key: {0}'.format(k))
            #return False
        else:
            d[k] = v
    return d


### find Threshold to maintain Rate setpoint (in Hz) at temp (Kelvin)
def getThresh(*pars, setpoint, temp, temp_unit='kelvin'):
    # functional form of expected fit is:
    # Thresh = p[0]/(1.+exp((R/p[1])*(T/p[2]))) + p[3] + p[4]*R + p[5]*T
    if len(pars) != 6:
        print('ERROR: *pars requires 6 parameters')
        return 0
    if temp_unit.lower() not in ['k', 'kelvin', 'c', 'celsius']:
        print('ERROR: temp_unit must be one of {0}'
              .format(['k', 'kelvin', 'c', 'celsius']))
        return 0
    if temp_unit.lower() in ['k', 'kelvin']:
        if float(temp) <= 100:
            print('ERROR: temp does not seem to be in Kelvin')
            return 0
        T = float(temp)
    if temp_unit.lower() in ['c', 'celsius']:
        if float(temp) > 100:
            print('ERROR: temp does not seem to be in Celsius')
            return 0
        T = float(temp) + 273.15

    Rate = float(setpoint)
    if Rate < 400 or Rate > 3000:
        print('ERROR: Rate setpoint in Hz should be 400 < Rate < 3000')
        return 0
    p = pars
    thresh = int(round(p[0]/(1.+exp((Rate/p[1])*(T/p[2]))) + p[3] + p[4]*Rate + p[5]*T, 0))
    return thresh


### get fit pars for panel from config and find Threshold
def getPanelThresh(uid, setpoint, temp, temp_unit='kelvin'):
    if not os.path.exists(CONFIG):
        print('ERROR: config not found -> {0}'.format(CONFIG))
        return 0
    with open(CONFIG) as jf:
        config = json.load(jf, object_pairs_hook=raise_on_duplicates)
    if uid not in config:
        print('ERROR: uid [{0}] not in config'.format(uid))
        return 0
    if 'thresh_pars' not in config[uid]:
        print('ERROR: \"thresh_pars\" not found for [{0}]'.format(uid))
        return 0
    params = [float(x) for x in config[uid]['thresh_pars']]
    thresh = getThresh(*params, setpoint=setpoint, temp=temp, temp_unit=temp_unit)
    return thresh


### find HV to maintain MIP setpoint (in ADC) at temp (Kelvin)
def getHV(*pars, setpoint, temp, temp_unit='kelvin'):
    # functional form of expected fit is:
    # MIP = p[0] + p[1]*T + p[2]*HV + p[3]*T*HV + p[4]*(T**2) + p[5]*(HV**2)
    if len(pars) != 6:
        print('ERROR: *pars requires 6 parameters')
        return 0
    if temp_unit.lower() not in ['k', 'kelvin', 'c', 'celsius']:
        print('ERROR: temp_unit must be one of {0}'
              .format(['k', 'kelvin', 'c', 'celsius']))
        return 0
    if temp_unit.lower() in ['k', 'kelvin']:
        if float(temp) <= 100:
            print('ERROR: temp does not seem to be in Kelvin')
            return 0
        T = float(temp)
    if temp_unit.lower() in ['c', 'celsius']:
        if float(temp) > 100:
            print('ERROR: temp does not seem to be in Celsius')
            return 0
        T = float(temp) + 273.15

    MIP = float(setpoint)
    if MIP < 500 or MIP > 2000:
        print('ERROR: MIP setpoint in ADC should be 500 < MIP < 2000')
        return 0
    p = pars
    A = p[5]
    B = p[2] + p[3]*T
    C = p[1]*T + p[4]*(T**2) + p[0] - MIP
    HV = int(round((-B + sqrt(B**2 - 4*A*C)) / (2*A), 0))
    return HV


### get fit pars for panel from config and find HV
def getPanelHV(uid, setpoint, temp, temp_unit='kelvin'):
    if not os.path.exists(CONFIG):
        print('ERROR: config not found -> {0}'.format(CONFIG))
        return 0
    with open(CONFIG) as jf:
        config = json.load(jf, object_pairs_hook=raise_on_duplicates)
    if uid not in config:
        print('ERROR: uid [{0}] not in config'.format(uid))
        return 0
    if 'mip_pars' not in config[uid]:
        print('ERROR: \"mip_pars\" not found for [{0}]'.format(uid))
        return 0
    params = [float(x) for x in config[uid]['mip_pars']]
    hv = getHV(*params, setpoint=setpoint, temp=temp, temp_unit=temp_unit)
    return hv


### round in scientific format
### useful for comparing fit pars to bounds
def sciround(number):
    if not isinstance(number, (int, float)):
        print('ERROR: value {0} is not int or float'.format(number))
        return number
    num = '{:0.2e}'.format(number)
    return float(num)


### put run info into text for plots
def makeText(info):
    text = {}
    for chan in range(1,9):
        chan = str(chan)
        if chan in info:
            text[chan] = []
            #if 'firmware' in info:
                #text[chan].append('{0}'.format(info['firmware'][:-4]))
            #if 'uid' in info[chan]:
                #text[chan].append('{0}'.format(info[chan]['uid']))
            if 'date' in info and 'time' in info:
                text[chan].append('{0} {1} UTC'.format(info['date'], info['time']))
            if 'temperature' in info[chan]:
                text[chan].append('temp = {0} C'.format(round(info[chan]['temperature'][1],1)))
            if 'voltage' in info[chan]:
                text[chan].append('hv = {0}'.format(info[chan]['voltage']))
            if 'threshold' in info[chan]:
                text[chan].append('thresh = {0}'.format(info[chan]['threshold']))
            if 'good_duration' in info[chan]:
                text[chan].append('livetime = {0} sec'.format(int(round(info[chan]['good_duration'],1))))
            elif 'livetime' in info:
                text[chan].append('livetime = {0} sec'.format(int(round(info['livetime'],1))))
            if 'good_rate' in info[chan]:
                text[chan].append('trig rate = {0} Hz'.format(int(round(info[chan]['good_rate'],1))))
            elif 'trigrate' in info[chan]:
                text[chan].append('trig rate = {0} Hz'.format(int(round(info[chan]['trigrate'],1))))
            if 'comment' in info:
                if info['comment']:
                    text[chan].append('{0}'.format(info['comment']))
    return text


### try to get real run duration skipping gaps between subruns
### subrun gaps generally >1 sec
### not tested on LBM overflows yet
def livetime(hits):
    time_sum = 0
    for i, hit in enumerate(hits):
        dt = float(hit[0]) - float(hits[i-1][0])
        if dt > 1.0: continue
        elif dt < 0.0: continue
        else: time_sum += dt
    return time_sum


### find dt between coinc hits
def findCoinc(times1, times2, window, offset=0.0):

    debug = 0

    offset=float(offset)
    
    coinc1 = []
    coinc2 = []
    coinc_dts = []
    
    k=0
    for i, t0 in enumerate(times1):
        
        if t0-window+offset > times2[-1]: continue
        if t0+window+offset < times2[0]: continue
        
        if debug: print('\n--> find closest time to', t0)
        
        stop = 0
        m = 0
        for j, t1 in enumerate(times2[k:]):
            
            if stop: break
            
            if abs(t0-t1+offset) < window:
                mintime = abs(t0-t1+offset)
                if debug: print('1', mintime, k, j)
                for l, t2 in enumerate(times2[j+k:]):
                    if abs(t0-t2+offset) <= mintime:
                        mintime = abs(t0-t2+offset)
                        pt0 = t0
                        pt2 = t2
                        if debug: print('2', mintime)
                        continue
                    elif abs(t0-t2+offset) > mintime:
                        k = j+k+l-1
                        if debug: print('3', abs(t0-t2+offset))
                        #if pt0 < pt2:
                        #    coinc.append(pt0)
                        #else:
                        #    coinc.append(pt2)
                        coinc1.append(pt0)
                        coinc2.append(pt2)
                        coinc_dts.append(pt0-pt2+offset)
                        stop = 1
                        break
                    else:
                        print('does this happen?')
                        continue
            
            if t1 < t0-window+offset:
                m = j
            
            if t1 > t0+window+offset:
                k = k+m
                if debug: print('4', t1, '>?', t0+window+offset)
                stop = 1
                break
            
    return coinc1, coinc2, coinc_dts

