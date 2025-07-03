#!/usr/bin/python

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import moyal
import matplotlib.pyplot as plt


# 2023-07-07 - updated simpleMIPFit
# 2023-07-21 - fixed findGain() bug when all bins are 0
# 2023-09-27 - tweaked peMIPFit() to help with SET-3 chan-6 weirdness
#              seems KIT0140 might have low light yield causing the issues
# 2023-09-28 - more tweaks to findGain() and peMIPFit()
# 2023-11-15 - more tweaks to findGain() - handles low gains better
# 2023-11-16 - normalize peMIPfit hist so fitting bounds are handled better



### bunch of fitting functions
#-----------------------------------------------------------
def expon(x, *p):
    return (p[0]/(1+np.exp(x/p[1])))

def exponoff(x, *p):
    return (p[0]/(1+np.exp(x/p[1]))) + p[2]

def expon2off(x, *p):
    return (p[0]/(1+np.exp(x/p[1]))) * (p[2]/(1+np.exp(x/p[3]))) + p[4]

def gauss(x, *p):
    return (p[0]*np.exp(-0.5*((x-p[1])/(p[2]))**2))

def gaussoff(x, *p):
    return (p[0]*np.exp(-0.5*((x-p[1])/(p[2]))**2)) + p[3]

def expogauss(x, *p):
    return expon(x, p[:2]) + gauss(x, p[2:])

def expogaussoff(x, *p):
    return exponoff(x, p[:3]) + gauss(x, p[3:])

def natlog(x, *p):
    return p[0]*(1-np.log(x+p[1]))

def natlogoff(x, *p):
    return p[0]*(1-np.log(x+p[1])) + p[2]

def logexpo(x, *p):
    p1 = list(p[:3])
    p2 = list(p[3:5])
    return natlogoff(x, *p1) * expon(x, *p2) + p[5]

def landau(x, *p):
    return (p[0]*p[2]) * moyal.pdf(x, p[1], p[2])

def expolandau(x, *p):
    p1 = list(p[:2])
    p2 = list(p[2:])
    return expon(x, *p1) + landau(x, *p2)
#-----------------------------------------------------------


### some smoothing functions
#-----------------------------------------------------------
def smooth3(data):
    this=[0 for k in range(len(data))]
    for i in range(1, len(data)-1):
        this[i] = (data[i-1] + 2*data[i] + data[i+1]) / 4.
    return this

def smooth5(data):
    this=[0 for k in range(len(data))]
    for i in range(2, len(data)-2):
        this[i] = (data[i-2] + 2*data[i-1] + 3*data[i] 
                   + 2*data[i+1] + data[i+2]) / 9.
    return this

def smooth7(data):
    this=[0 for k in range(len(data))]
    for i in range(3, len(data)-3):
        this[i] = (data[i-3] + 2*data[i-2] + 3*data[i-1] + 4*data[i] 
                   + 3*data[i+1] + 2*data[i+2] + data[i+2]) / 16.
    return this

def peakext3(data):
    this=[0 for k in range(len(data))]
    for i in range(1, len(data)-1):
        this[i] = (data[i-1] + data[i]**2 + data[i+1])
    this = np.asarray(this) / max(this)
    return this

def peakext5(data):
    this=[0 for k in range(len(data))]
    for i in range(2, len(data)-2):
        this[i] = (data[i-2] + data[i-1]**2 + data[i]**3 
                   + data[i+1]**2 + data[i+2])
    this = np.asarray(this) / max(this)
    return this

def peakext7(data):
    this=[0 for k in range(len(data))]
    for i in range(3, len(data)-3):
        this[i] = (data[i-3] + data[i-2]**2 + data[i-1]**3 + data[i]**4 
                   + data[i+1]**3 + data[i+2]**2 + data[i+3])
    this = np.asarray(this) / max(this)
    return this
#-----------------------------------------------------------


### some other random functions
#-----------------------------------------------------------
# find the mean of a histogram
def weighted_mean(histo, xmin, xmax):
    x = []
    for i in range(len(histo[xmin:xmax])):
        x.append(i * ((xmax - xmin) / len(histo[xmin:xmax])))
    try:
        wm = float(np.average(x, weights=histo[xmin:xmax])) + xmin
    except Exception as e:
        print('ERROR: in weighted_mean()')
        print('      ', e)
        wm = 0
    return wm

# simple gauss fit to find mean and sigma
def prefit(hist, guess, fmin, fmax):
    fmin = int(fmin)
    fmax = int(fmax)
    hmax = max(hist[fmin:fmax])
    nmin = hmax / 1000
    nmax = hmax * 1000
    ###        N     mean   sig
    #seed   = [ 10,  guess,  200]
    #lo_bnd = [  1,   fmin,   50]
    #hi_bnd = [1e6,   fmax,  800]
    seed   = [hmax,  guess,  200] # 2023-11-16
    lo_bnd = [nmin,   fmin,   50] # 2023-11-16
    hi_bnd = [nmax,   fmax,  800] # 2023-11-16

    opt, cov = curve_fit(gauss,
                         range(fmin,fmax), 
                         hist[fmin:fmax],
                         p0=seed,
                         bounds=[lo_bnd, hi_bnd]
                         )
    return [opt[1], opt[2]]

# find chi2 of two arrays
def chisquare(x, y):
    if len(x) != len(y):
        print('ERROR: chi2 lengths {0} != {1}'.format(len(x), len(y)))
        return 0
    chi2 = 0
    for X, Y in zip(x, y):
        chi2 += ((X-Y)**2) / Y
    return chi2 / float(len(x))

# find the pedestal
def findPed(data):
    pedbin = np.argmax(data[:600])
    pedestal = weighted_mean(data, pedbin-10, pedbin+10)
    return pedestal

# for finding the valley in the FFT
def findValley(data, startbin=10, debug=False):
    if debug: print('Starting valley scan bin = {0}'.format(startbin))
    binsize = 0.5/float(len(data))
    # try a few scan widths starting with the larger
    for inc in [0.005, 0.002]:
    #for inc in [0.002, 0.004]: # 2023-11-15
        if debug: print('Trying valley scan width = {0}'.format(inc))
        fftbin = 0
        fftbins = []
        minvals = []
        for i, j in enumerate(range(int(0.1/inc))):
            minbin = int(j*(inc/binsize)) + startbin
            maxbin = int((j+1)*(inc/binsize)) + startbin
            fftbins.append(np.argmin(data[minbin:maxbin])+minbin)
            minvals.append(np.min(data[minbin:maxbin]))
            if i==0 or i==1: continue
            if minvals[i] < minvals[i-1]:
                fftbin = fftbins[i]
            if minvals[i] > minvals[i-1]:
                return fftbin
        
    return startbin

# round in scientific format
def sciround(number):
    if not isinstance(number, (int, float)):
        print('ERROR: value {0} is not int or float'.format(number))
        return number
    num = '{:0.2e}'.format(number)
    return float(num)

# find the first bin with usable data in it
def firstDataBin(data):
    #thresh = 0.8 * max(data[:-1])
    thresh = 0.5 * max(data[:-1])
    for i, val in enumerate(data):
        if val >= thresh:
            return i
    return 0

# find the last bin with usable data in it
def lastDataBin(data):
    # skip last saturation bin just in case
    data = data[:-1]
    thresh = 0.02 * max(data)
    for i in range(len(data)):
        if data[-i] >= thresh:
            return len(data) - i
    return 0

# info to add to fit plots
def reducedText(info):
    text = []
    if 'temperature' in info:
        text.append('{0} C temp'.format(round(info['temperature'][1],1)))
    if 'voltage' in info:
        text.append('{0} volts'.format(info['voltage']))
    if 'threshold' in info:
        text.append('{0} thresh'.format(info['threshold']))
    if 'good_rate' in info:
        text.append('{0} Hz rate'.format(int(round(info['good_rate'],0))))
    elif 'trigrate' in info:
        text.append('{0} Hz rate'.format(int(round(info['trigrate'],0))))
    
    return text
#-----------------------------------------------------------


# try to find the gain of a charge histogram
def findGain(hist, title='', save='', plot=False, debug=False):

    #if debug: print('\nRunning findGain()')

    info = {}
    info['gain'] = 0
    info['good_gain'] = True
    info['bad_gain_reasons'] = []
    
    ypoints = list(hist)
    #ypoints = smooth3(hist)
    
    #ypoints.extend(np.zeros(4096-len(ypoints)))
    ypoints.extend(np.zeros(8192-len(ypoints)))
    
    # 512 bin is noisy and is a gain ~8
    # should just skip everything > 500?
    # but going to 1000 helps the mean test fail
    lastbin = 1000

    real = np.fft.rfft(ypoints)
    mag = abs(real)

    this = mag

    freq = list(range(len(this)))
    binsize = 0.5/float(len(this))
    
    # normalize the fft 2023-11-15
    baseline = np.mean(smooth5(this)[lastbin-200:lastbin])
    norm = np.asarray(smooth5(this)) / baseline
    #mnorm = np.mean(norm[800:1000])
    #print('norm mean =', mnorm)
    
    func = logexpo
    ####       N1    t1   off1     N2     t2   off2
    #seed  = [  10,    1,    10,    10,    10,    10]
    #lobnd = [   1,    0,     1,     1,     1,     1]
    #hibnd = [ 1e3,  200,   1e4,   1e3,   100,   1e4]
    seed  = [  10,   10,    10,    10,    10,     1] # 2023-11-15
    lobnd = [   0,    0,     0,     1,     1,     0] # 2023-11-15
    hibnd = [ 1e3,  300,   1e4,   1e3,   100,    10] # 2023-11-15

    for vbin in [30, 20, 10]:
        
        # find the first valley
        firstvalley = findValley(smooth5(this), startbin=vbin, debug=debug) # 2023-09-28
        #firstvalley = findValley(smooth5(this)[:lastbin], startbin=vbin, debug=debug) # 2023-11-15

        #fmax = int(len(freq))
        #fmax = firstvalley # 2023-09-28
        fmax = int(firstvalley*1.3) # 2023-11-15

        passed1 = False
        #for k in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]: # 2023-09-28
        #for k in [0.6, 0.5, 0.4, 0.3, 0.2]: # 2023-11-15
        for k in [0.2, 0.3, 0.4, 0.5, 0.6]:
            if debug: print('Expo fit fraction = {0}'.format(k))
            
            startbin = int(firstvalley * k)
            fmin = startbin
            try:
                fftopt, fftcov = curve_fit(func,
                                           freq[fmin:fmax],
                                           #freq[fmin:fmax].extend([800, 900]), # 2023-11-15
                                           #this[fmin:fmax],
                                           #smooth5(this)[fmin:fmax], # 2023-11-15
                                           norm[fmin:fmax], # 2023-11-15
                                           #norm[fmin:fmax].extend([1, 1]), # 2023-11-15
                                           p0=seed,
                                           bounds=[lobnd, hibnd])
                passed1 = True
                #print(fftopt)
                break
            except Exception as e:
                if debug: print('Try next fraction...')
                continue

        if not passed1:
            """
            print('ERROR: expo fit to gain FFT failed')
            print('      ', e)
            info['good_gain'] = False
            info['bad_gain_reasons'].append('expo fit to gain FFT failed')
            info['gain'] = 0
            return info
            """
            if debug: print('Try next start bin...')
            continue
        
        #print(title, ['{:0.2e}'.format(x) for x in fftopt])


        sub = [val-fit for val,fit in zip(this, [func(x, *fftopt) for x in freq])]

        #skip_first = 50
        if startbin > firstvalley:
            skip_first = startbin
        else:
            skip_first = firstvalley


        sub = smooth5(sub)

        # zero negative bins and first 50 bins
        for i, val in enumerate(sub):
            if val < 0 or i < skip_first or i > lastbin:
                sub[i] = 0

        # if all bins are 0, no good
        passed2 = False
        if sum(sub) == 0:
            if debug: print('all zeros, try next sbin...')
            continue
        
        passed2 = True
        break
    
    if not passed1:
        print('ERROR: expo fit to gain FFT failed')
        #print('      ', e)
        info['good_gain'] = False
        info['bad_gain_reasons'].append('expo fit to gain FFT failed')
        info['gain'] = 0
        return info
    
    if not passed2:
        print('ERROR: all bins are 0')
        info['good_gain'] = False
        info['bad_gain_reasons'].append('all bins are 0')
        info['gain'] = 0
        return info
    
    # try exagerating the peak
    #sub = smooth5(sub)
    sub = peakext5(sub)
    #sub = peakext7(sub)
    
    # find the mean of the expo subtracted fft
    meanbin = int(round(weighted_mean(sub, 0, lastbin), 0))
    #print(meanbin)
    
    # find the max of the expo subtracted fft
    maxbin = np.argmax(sub[:lastbin])
    #maxamp = this[maxbin]
    maxamp = norm[maxbin]
    #print(maxbin)
    
    # find the max in the range of 1st valley + X bins
    #pwindow = 300
    #peakbin = np.argmax(smooth5(this)[firstvalley:firstvalley+pwindow])+firstvalley
    #if peakbin == firstvalley+pwindow-1:
    #    info['good_gain'] = False
    #    info['bad_gain_reasons'].append('peak found at edge {0}'.format(pwindow))
    # search to the end 2023-11-15
    peakbin = np.argmax(smooth5(this)[firstvalley:lastbin+1])+firstvalley
    if peakbin >= lastbin:
        info['good_gain'] = False
        info['bad_gain_reasons'].append('peak found at last bin {0}'.format(lastbin))

    if maxbin > meanbin: diff = round(((maxbin-meanbin)/maxbin), 2)
    else: diff = round(((meanbin-maxbin)/meanbin), 2)
    
    if maxbin > peakbin: diff2 = round(((maxbin-peakbin)/maxbin), 2)
    else: diff2 = round(((peakbin-maxbin)/peakbin), 2)
    
    
    #print('diff = {2}   maxbin = {0}  meanbin = {1}'.format(maxbin, meanbin, diff))
    
    # if the mean is too far from max, mark it bad
    #if diff < 0.4:
    if diff > 0.2:
        info['good_gain'] = False
        info['bad_gain_reasons'].append('large maxbin-meanbin = {0}'.format(round(diff, 2)))
    if diff2 > 0.05:
        info['good_gain'] = False
        info['bad_gain_reasons'].append('large maxbin-peakbin = {0}'.format(round(diff2, 2)))
    #print(diff, diff2)

    # try weighted mean of the peak so gain is more accurate
    #wmax = weighted_mean(this, maxbin-10, maxbin+10)
    wmax = weighted_mean(norm, maxbin-10, maxbin+10) # 2023-11-15

    try:
        #calib = len(ypoints) / float(maxbin) # 2023-09-27
        calib = len(ypoints) / float(wmax) # 2023-11-15
        info['gain'] = calib
    except Exception as e:
        print('ERROR: calibration division by zero')
        print('      ', e)
        info['good_gain'] = False
        info['bad_gain_reasons'].append('calibration division by zero')
        info['gain'] = 0
        return info
    
    # sometimes the 512 bin noise is flagged as good
    # need to mark this as bad, has gain ~8
    #min_gain = 9
    min_gain = 8 # 2023-11-15
    if info['gain'] < min_gain:
        info['good_gain'] = False
        info['bad_gain_reasons'].append('low gain = {0}'.format(round(calib, 1)))

    # plot the FFT and gain value
    if plot or save:
        fig, ax = plt.subplots(2,1,figsize=(18,6),facecolor='w',edgecolor='k')
        fig.suptitle(title, size=20)
        plt.sca(ax[0])

        # fft data
        plt.plot(freq,
                 #smooth5(this),
                 norm,
                 color='black', alpha=0.8, ls='-', lw=2, 
                 #label='FFT of charge',
                )

        # expon fit
        plt.plot(freq,  [func(x, *fftopt) for x in freq], 
                 color='red', alpha=0.8, ls='-', lw=1,
                 label='expo fit',
                )

        # line at fit start
        plt.plot([startbin, startbin],  [0, 2*maxamp],
                 color='green', alpha=0.8, ls='-', lw=1,
                 label='fit start = {0}'.format(int(startbin)),
                )

        # line at first valley
        plt.plot([firstvalley, firstvalley],  [0, 2*maxamp],
                 color='orange', alpha=0.8, ls='-', lw=1,
                 label='valley = {0}'.format(int(firstvalley)),
                )

        # line at max from 1st valley
        plt.plot([peakbin, peakbin],  [0, 2*maxamp],
                 color='purple', alpha=0.8, ls='-', lw=1,
                 label='peak find = {0}'.format(int(peakbin)),
                )
        """
        # line at peak
        plt.plot([maxbin, maxbin],  [0, 2*maxamp],
                 color='blue', alpha=0.8, ls='-', lw=1,
                 label='gain = {0}\nstatus = {1}'.format(round(info['gain'],1), info['good_gain']),
                )
        """
        plt.xlim(0, lastbin)
        plt.ylim(0, 2*maxamp)
        #plt.ylim(0, 5000)
        #plt.title(title, size=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        plt.legend(fontsize=16, loc='upper right')


        # plot the expon subtracted data
        #fig, ax = plt.subplots(1,1,figsize=(18,4),facecolor='w',edgecolor='k')
        plt.sca(ax[1])

        plt.plot(freq[:lastbin],  sub[:lastbin], 
                 color='black', alpha=0.8, ls='-', lw=2,
                 #label='expo subtracted fft',
                )

        plt.plot([maxbin, maxbin],  [0, 1.2*sub[maxbin]],
                 color='blue', alpha=0.8, ls='-', lw=1,
                 label='max bin = {0}'.format(int(maxbin)),
                 #label='gain = {0}\nstatus = {1}'.format(round(info['gain'],1), info['good_gain']),
                 )

        plt.plot([meanbin, meanbin],  [0, 1.2*sub[maxbin]],
                 color='green', alpha=0.8, ls='-', lw=1,
                 label='mean = {0}'.format(int(meanbin)),
                 )
        """
        plt.plot([peakbin, peakbin],  [0, 1.2*sub[maxbin]],
                 color='orange', alpha=0.8, ls='-', lw=1,
                 label='peak find',
                 )
        """

        plt.xlim(0, lastbin)
        plt.ylim(0, 1.2*sub[maxbin])
        #plt.title(title, size=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        plt.legend(fontsize=16, loc='upper right')

        if save: plt.savefig(save, bbox_inches='tight')
        if not plot: plt.close()
            
        
    return info


# do a more basic expo-landau fit without knowing the gain
# only results for adc/MIP
def adcFitMIP(hist, ped, config=False, title='', save='', plot=False, debug=False):

    if debug: print('\nRunning adcFitMIP()')

    if config: plotText = reducedText(config)

    info = {}
    info['good_adcfit'] = True
    info['adcfit_results'] = False
    info['bad_adcfit_reasons'] = []
    
    maxlook = 4000
    
    pedestal = findPed(ped)
    if debug: print('pedestal = {0}'.format(round(pedestal, 1)))
    info['pedestal'] = pedestal
    
    thresh = int(np.argmax(hist[:-1]))
    firstdata = firstDataBin(hist)
    info['thresh'] = firstdata - pedestal
    # sometimes the MIP is set as the threshold
    # so fail when this happens
    if thresh-firstdata > 50:
        info['good_adcfit'] = False
        info['bad_adcfit_reasons'].append('large maxbin-firstdata = {0}'.format(int(thresh-firstdata)))
        
    #print(thresh-firstdata)
    #thresh = findThresh(hist)
    maxval = max(hist[:-1])
    
    mean1 = int(weighted_mean(hist, thresh, maxlook))
    mean2 = int(weighted_mean(hist, mean1, maxlook))
    
    #valley = findValley((hist), startbin=thresh)
    #print(valley)
    
    fmin = thresh
    #fmin = mean1

    fmax = mean2*2
    fmax = maxlook
    if fmax > maxlook:
        fmax = maxlook

    hmax = max(hist)
    nmax = hmax * 100
    mmax = hmax * 10
    ###         N    tau     M     mean     sig
    #seed   = [ 1e4,  500,   500,  thresh,   200]
    seed   = [hmax,  500,  hmax,  thresh,   200]
    lo_bnd = [  10,   50,    10,  thresh,    50]
    hi_bnd = [nmax,  1e3,  mmax,   fmax,   1000]

    try:
        opt, cov = curve_fit(expolandau,
                         list(range(fmin, fmax)), 
                         hist[fmin:fmax],
                         p0=seed,
                         bounds=[lo_bnd, hi_bnd]
                        )
    except Exception as e:
        print('ERROR: adc mip pre-fit failed')
        print('      ', e)
        info['good_adcfit'] = False
        info['bad_adcfit_reasons'].append('adc mip pre-fit failed')
        return info
    
    
    # refit again?
    prevmean = 1
    prevdiff = 1
    l = 0
    maxl = 20
    #for i in range(2):
    while True:
        # continue re-fitting until mean stabilizes?
        fitdiff = abs((prevmean-opt[3])/prevmean)
        diffdiff = abs(prevdiff-fitdiff)
        #print('adc fit {0} diff = {1}'.format(l, fitdiff))
        if fitdiff <= 0.01 or (l > 10 and diffdiff <= 0.005):
            if debug: print('{0} adc fit iterations to {1}'.format(l, round(fitdiff,4)))
            break
        prevmean = opt[3]
        prevdiff = fitdiff
        l += 1
        # break after some iterations in case things don't converge
        if l > maxl:
            break
        #-------------------------------------------
        
        mipadc = opt[3] - pedestal

        # set new fmin and fmax
        fmin = int(opt[3] - 0.7*mipadc)
        if fmin < thresh:
            fmin = thresh
        fmax = int(opt[3] + 1.0*mipadc)
        if fmax > 4000:
            fmax = 4000

        # set new bounds
        ###        N       tau     M       mean    sig
        seed   = [opt[0], opt[1], opt[2], opt[3], opt[4]]
        lo_bnd = [  10,     50,     10,    fmin,     50]
        hi_bnd = [ nmax,   1e3,    mmax,   fmax,   1000]

        # re fit
        try:
            opt, cov = curve_fit(expolandau,
                             list(range(fmin, fmax)), 
                             hist[fmin:fmax],
                             p0=seed,
                             bounds=[lo_bnd, hi_bnd]
                            )
        except Exception as e:
            print('ERROR: adc mip fit failed')
            print('      ', e)
            info['good_adcfit'] = False
            info['bad_adcfit_reasons'].append('adc mip re-fit failed')
            return info
        
    if l > maxl:
        print('WARNING: exceeded {0} adc fit iterations'.format(maxl))
        info['good_adcfit'] = False
        info['bad_adcfit_reasons'].append('exceeded {0} fit iterations'.format(maxl))
    
    # sanity checks on bounds
    out_of_bounds = False
    for i in range(len(opt)):
        if sciround(opt[i]) <= sciround(lo_bnd[i]):
            if debug: print('WARNING: adc fit opt[{0}] at lower bound'.format(i))
            info['good_adcfit'] = False
            info['bad_adcfit_reasons'].append('opt[{0}] at lower bound'.format(i))
            out_of_bounds = True
        if sciround(opt[i]) >= sciround(hi_bnd[i]):
            if debug: print('WARNING: adc fit opt[{0}] at upper bound'.format(i))
            info['good_adcfit'] = False
            info['bad_adcfit_reasons'].append('opt[{0}] at upper bound'.format(i))
            out_of_bounds = True
        
    #print(title, ['{:0.2e}'.format(x) for x in opt])

    # find the chi2 of the fit
    lastdata = lastDataBin(smooth5(hist))
    if fmax > lastdata: lastdata = fmax
    chi2 = chisquare(smooth5(hist[fmin:lastdata]),
                     [expolandau(x, *opt) for x in range(fmin, lastdata)])
    if debug: print('adc fit chi2 = {0}  range = {1} - {2}'.format(round(chi2, 1), fmin, lastdata))
    if chi2 > 8:
        print('WARNING: large adc fit chi2 of {0}'.format(round(chi2, 1)))
        info['good_adcfit'] = False
        info['bad_adcfit_reasons'].append('large chi2 of {0}'.format((round(chi2, 1))))
        
    if debug: 
        print('ADC MIP Fit Results')
        print('   fit range = {0} - {1}'.format(fmin, fmax))
        print('   fit iterations = {0}'.format(l))
        print('   chi2 = {0}'.format(round(chi2, 1)))
        print('   N    = {0}  [{1}, {2}]'.format(int(opt[0]), int(lo_bnd[0]), int(hi_bnd[0])))
        print('   tau  = {0}  [{1}, {2}]'.format(int(opt[1]), int(lo_bnd[1]), int(hi_bnd[1])))
        print('   M    = {0}  [{1}, {2}]'.format(int(opt[2]), int(lo_bnd[2]), int(hi_bnd[2])))
        print('   mean = {0}  [{1}, {2}]'.format(int(opt[3]), int(lo_bnd[3]), int(hi_bnd[3])))
        print('   sig  = {0}  [{1}, {2}]'.format(int(opt[4]), int(lo_bnd[4]), int(hi_bnd[4])))

    info['adcfit_results'] = {}
    info['adcfit_results']['fit_iter'] = l
    info['adcfit_results']['chi2'] = chi2
    info['adcfit_results']['N'] = opt[0]
    info['adcfit_results']['tau'] = opt[1]
    info['adcfit_results']['M'] = opt[2]
    info['adcfit_results']['mean'] = opt[3]
    info['adcfit_results']['mean_err'] = np.sqrt(cov[3][3])
    info['adcfit_results']['sig'] = opt[4]
    info['adcfit_results']['out_of_bounds'] = out_of_bounds

    adcPerMIP = opt[3] - pedestal
    adcPerMIP_err = np.sqrt(cov[3][3])
    info['adcfit_results']['adcPerMIP'] = adcPerMIP
    info['adcfit_results']['adcPerMIP_err'] = adcPerMIP_err
    info['adcPerMIP'] = adcPerMIP
    
    if debug: print('MIP = {0} +/- {1} adc/MIP'.format(int(adcPerMIP), int(adcPerMIP_err)))
    
    if plot or save:
        
        pmin = fmin
        pmax = fmax

        fig, ax = plt.subplots(1,1,figsize=(18,6),facecolor='w',edgecolor='k')
        fig.suptitle(title, size=22)
        plt.sca(ax)
        
        plttext = [
            '{0} +/- {1}  adc/mip'.format(round(adcPerMIP,1), round(adcPerMIP_err,1)),
            'thresh = {0} mip'.format(round(info['thresh']/adcPerMIP,2)),
        ]
        
        Y = 0.8
        for text in plttext:
            plt.text(x=0.99, y=Y, s=text,
                     fontsize=18, horizontalalignment='right', transform=ax.transAxes)
            Y -= 0.07
        if plotText:
            Y -= 0.07 # add a space
            for text in plotText:
                plt.text(x=0.99, y=Y, s=text,
                     fontsize=18, horizontalalignment='right', transform=ax.transAxes)
                Y -= 0.07
                
        plt.plot(range(len(hist)), 
                 hist, 
                 color='black', alpha=0.7,
                 #label='data'
                 )

        plt.plot(range(len(ped)), 
                 ped, 
                 color='purple', alpha=0.7,
                 #label='pedestal'
                 )

        #plt.plot([valley, valley], 
        #         [0, maxval], 
        #         color='orange', alpha=0.7,
        #         label='find valley'
        #         )

        plt.plot(range(pmin,pmax),
                 [expon(x, opt[0], opt[1]) for x in range(pmin, pmax)], 
                 color='green', 
                 #label='expo darknoise'
                 )

        plt.plot(range(pmin,pmax),
                 [landau(x, opt[2], opt[3], opt[4]) for x in range(pmin, pmax)], 
                 color='blue', 
                 #label='landau MIP'
                 )

        if info['bad_adcfit_reasons']: reasons = info['bad_adcfit_reasons']
        else: reasons = ''
        plt.plot(range(fmin, fmax), 
                 [expolandau(x, *opt) for x in range(fmin, fmax)], 
                 color='red', alpha=0.9, linestyle='-', lw=2,
                 label='good fit = {0}  {1}'.format(info['good_adcfit'], reasons)
                 )

        #plot_title = ('adc/mip = {0} +/- {1}'
        #                  .format(round(adcPerMIP,1), round(adcPerMIP_err,1)))
        #if title: plot_title = title+'  '+plot_title
        #plt.title(plot_title, size=20)
        #plt.title(title, size=20)
        
        plt.ylabel('counts', size=20)
        plt.xlabel('charge (adc)',size=20)
        plt.tick_params(axis='both', which='major', labelsize=18)

        plt.legend(fontsize=16, loc='upper right')

        plt.xlim(0, 5000)
        plt.ylim(1)
        plt.yscale('log')
        
        if save: plt.savefig(save, bbox_inches='tight')
        if not plot: plt.close()
        
    return info


# more thorough fit using the gain to determine pe/MIP
def peFitMIP(hist, ped, config=False, title='', save='', plot=False, debug=False):

    if debug:
        print()
        print(title)
        print('Running peFitMIP()')
    
    if config: plotText = reducedText(config)
    
    # need a rough guess at pe/mip
    #mip_pe = 25  # PSL panels
    mip_pe = 50  # KIT panels

    info = {}
    info['good_pefit'] = True
    info['pefit_results'] = False
    info['bad_pefit_reasons'] = []
    
    # find the gain from the fft
    if debug: print('INFO: finding the gain')
    gaininfo = findGain(hist[1:-1], title=title, plot=False, debug=debug)
    info.update(gaininfo)
    gain = info['gain']
    if debug: print('gain = {0}'.format(round(gain, 1)))
    if not info['good_gain']:
        # 2023-09-27
        # Modified this part to try the MIP fit even
        # if the gain might be a little off
        if info['gain'] < 10:
            print('ERROR: undetermined gain')
            info['good_pefit'] = False
            info['bad_pefit_reasons'].extend(info['bad_gain_reasons'])
            return info
        else:
            print('WARNING: gain possibly wrong, trying MIP fit anyway')
            info['bad_pefit_reasons'].extend(['gain possibly wrong'])
    
    # find the pedestal
    pedestal = findPed(ped)
    if debug: print('pedestal = {0}'.format(round(pedestal, 1)))
    info['pedestal'] = pedestal
    
    # estimate the threshold
    #thresh = int(np.argmax(hist[:-1]))
    thresh = int(np.argmax(hist[:-10])) # avoid saturation bins 2023-11-16
    firstdata = firstDataBin(hist)
    info['thresh'] = firstdata - pedestal
    # sometimes the MIP is set as the threshold
    # so fail when this happens
    if thresh-firstdata > 50:
        info['good_pefit'] = False
        info['bad_pefit_reasons'].append('large maxbin-firstdata = {0}'.format(int(thresh-firstdata)))

    # test normalizing the histogram? 2023-11-16
    normalize = 1
    if normalize:
        norm = max(hist[:-10])
        hist = list(np.asarray(hist) / norm)
        ped = list(np.asarray(ped) / norm)

    # try weighted mean again
    # num PEs to start at is touchy
    wmin = int(pedestal+((0.5*mip_pe)*gain))
    if wmin >= 4000:
        #wmin = 3000
        wmin = int(pedestal)
    wmax = 4000
    
    if debug: print('weighted mean fit range {0} - {1}'.format(wmin, wmax))
    try:
        wmean = weighted_mean(hist, wmin, wmax)
    except:
        print('WARNING: trying simple weighted mean')
        wmean = weighted_mean(hist, int(pedestal), 4000)

    #if wmean < pedestal:
    #    print('WARNING: weighted mean < pedestal, trying again')
    #    wmean = weighted_mean(hist, int(pedestal), 4000)

    if debug: print('weighted mean = {0}'.format(round(wmean, 1)))
    
    # pre-fit with gauss to get better handle on mean
    try:
        pmean, psigma = prefit(hist, wmean, 0.7*wmean, 1.4*wmean)
        gmean = pmean
        lmbnd = pmean-(0.5*psigma)
        hmbnd = pmean+(0.5*psigma)
        gsigma = psigma
        lsbnd = 0.3*psigma
        hsbnd = 1.5*psigma
    except:
        print('WARNING: gaussian pre-fit failed, trying weighted mean')
        gmean = wmean
        lmbnd = int(pedestal)
        hmbnd = 1.5*wmean
        gsigma = 200
        lsbnd = 100
        hsbnd = wmean-int(pedestal)
        
    if debug: 
        print('mean guess =', round(gmean,1))
        print('sigma guess =', round(gsigma,1))
        
    # start at first data?
    #fmin = thresh
    # start at 10 pe?
    #fmin = int(pedestal+(10*gain)) # 2023-09-27
    # start at 1/4 mip?
    #fmin = int(pedestal+(0.25*(gmean-pedestal)))
    fmin = int(pedestal+(0.5*(gmean-pedestal))) # 2023-09-28
    
    # fit out to 2.5 mips if possible
    #fmax = int(pedestal+(2.5*gain*mip_pe))
    # use mean from pre-fit
    fmax = int(pedestal+(2.5*(gmean-pedestal)))
    if fmax > 4000:
        fmax = 4000

    if debug: print('pe mip pre-fit range = {0} - {1}'.format(fmin, fmax))
    
    hmax = max(hist)
    nmin = hmax / 100
    nmax = hmax * 100
    mmin = hmax / 10
    mmax = hmax * 10
    ###         N    tau     M    mean     sig
    #seed   = [ 1e4,  500,   500,  gmean,  gsigma]
    seed   = [hmax,  500,  hmax,  gmean,  gsigma]
    #lo_bnd = [  10,   50,    10,  lmbnd,   lsbnd]
    lo_bnd = [nmin,   50,  mmin,  lmbnd,   lsbnd] # 2023-11-16
    hi_bnd = [nmax,  1e3,  mmax,  hmbnd,   hsbnd]


    try:
        opt, cov = curve_fit(expolandau,
                         range(fmin,fmax), 
                         hist[fmin:fmax],
                         p0=seed,
                         bounds=[lo_bnd, hi_bnd])
    except Exception as e:
        print('ERROR: pe mip pre-fit failed')
        print('      ', e)
        info['good_pefit'] = False
        info['bad_pefit_reasons'].append('pe mip pre-fit failed')
        return info

    if debug: print('pe mip pre-fit mean = {0}'.format(round(opt[3], 1)))
    
    # refit again?
    #prevmean = 1
    prevmean = gmean # 2023-09-27
    prevdiff = 1
    l = 1
    maxl = 20
    
    while True:
        if l > maxl: break
        
        # continue re-fitting until mean stabilizes?
        """
        fitdiff = abs((prevmean-opt[3])/prevmean)
        diffdiff = abs(prevdiff-fitdiff)
        #print('pe fit {0} diff = {1}'.format(l+1, fitdiff))
        if fitdiff <= 0.01 or (l > 10 and diffdiff <= 0.005):
            if debug: print('{0} pe fit iterations to {1}'.format(l, round(fitdiff,4)))
            break
        """
        # changed to this 2023-09-27
        fitdiff = ((prevmean - opt[3]) / prevmean)
        diffdiff = (prevdiff - fitdiff)
        if debug: print('pe re-fit {0} diff = {1} %'.format(l, round(fitdiff, 4)))
        if (abs(fitdiff) <= 0.01) or (l > 10 and abs(fitdiff) <= abs(prevdiff)):
            if debug: print('{0} pe fit iterations to {1}'.format(l, round(fitdiff, 4)))
            break
        
        prevmean = opt[3]
        prevdiff = fitdiff
        mipadc = opt[3] - pedestal
        
        # set new fmin and fmax
        fmin = int(opt[3] - 0.7*mipadc)
        if fmin < thresh:
            fmin = int(thresh)
        fmax = int(opt[3] + 1.0*mipadc)
        if fmax > 4000:
            fmax = 4000

        # set new bounds
        nmin = opt[0] / 100
        nmax = opt[0] * 100
        mmin = opt[2] / 10
        mmax = opt[2] * 10
        
        # set new bounds
        ###        N       tau      M      mean     sig
        seed   = [opt[0], opt[1], opt[2], opt[3],  opt[4]]
        #lo_bnd = [ 10,     40,     10,     fmin,    50]
        #hi_bnd = [nmax,    1e3,    mmax,   fmax,   1000]
        #lo_bnd = [ 10,     40,     10,    int(opt[3]*0.5),  int(opt[4]*0.5)] # 2023-09-27
        lo_bnd = [nmin,     40,    mmin,  int(opt[3]*0.5),  int(opt[4]*0.5)] # 2023-11-16
        hi_bnd = [nmax,    1e3,    mmax,  int(opt[3]*1.5),  int(opt[4]*1.5)] # 2023-09-27

        if debug:
            print('lo =',[int(x) for x in lo_bnd])
            print('p0 =',[int(x) for x in seed])
            print('hi =',[int(x) for x in hi_bnd])
        
        # re fit
        try:
            opt, cov = curve_fit(expolandau,
                             list(range(fmin, fmax)), 
                             hist[fmin:fmax],
                             p0=seed,
                             bounds=[lo_bnd, hi_bnd]
                            )
        except Exception as e:
            print('ERROR: pe mip re-fit failed')
            print('      ', e)
            info['good_pefit'] = False
            info['bad_pefit_reasons'].append('pe mip re-fit failed')
            return info

        l += 1
        
    if l > maxl:
        print('WARNING: exceeded {0} pe fit iterations'.format(maxl))
        info['good_pefit'] = False
        info['bad_pefit_reasons'].append('exceeded {0} fit iterations'.format(maxl))
    
    # sanity checks on bounds
    out_of_bounds = False
    for i in range(len(opt)):
        if sciround(opt[i]) <= sciround(lo_bnd[i]):
            print('WARNING: pe fit opt[{0}] at lower bound'.format(i))
            info['good_pefit'] = False
            info['bad_pefit_reasons'].append('opt[{0}] at lower bound'.format(i))
            out_of_bounds = True
        if sciround(opt[i]) >= sciround(hi_bnd[i]):
            print('WARNING: pe fit opt[{0}] at upper bound'.format(i))
            info['good_pefit'] = False
            info['bad_pefit_reasons'].append('opt[{0}] at upper bound'.format(i))
            out_of_bounds = True
    
    # find the chi2 of the fit
    lastdata = lastDataBin(smooth5(hist))
    if fmax > lastdata: lastdata = fmax
    chi2 = chisquare(smooth5(hist[fmin:lastdata]),
                     [expolandau(x, *opt) for x in range(fmin, lastdata)])
    if debug: print('pe fit chi2 = {0}, fit range = {1} - {2}'
                    .format(round(chi2, 1), fmin, lastdata))
    if chi2 > 8:
        print('WARNING: large pe fit chi2 of {0}'.format(round(chi2, 1)))
        info['good_pefit'] = False
        info['bad_pefit_reasons'].append('large chi2 of {0}'.format((round(chi2, 1))))
        
    if debug: 
        print('PE MIP Fit Results')
        print('   fit range = {0} - {1}'.format(fmin, fmax))
        print('   fit iterations = {0}'.format(l))
        print('   chi2 = {0}'.format(round(chi2, 1)))
        print('   N    = {0}  [{1}, {2}]'.format(int(opt[0]), int(lo_bnd[0]), int(hi_bnd[0])))
        print('   tau  = {0}  [{1}, {2}]'.format(int(opt[1]), int(lo_bnd[1]), int(hi_bnd[1])))
        print('   M    = {0}  [{1}, {2}]'.format(int(opt[2]), int(lo_bnd[2]), int(hi_bnd[2])))
        print('   mean = {0}  [{1}, {2}]'.format(int(opt[3]), int(lo_bnd[3]), int(hi_bnd[3])))
        print('   sig  = {0}  [{1}, {2}]'.format(int(opt[4]), int(lo_bnd[4]), int(hi_bnd[4])))

    info['pefit_results'] = {}
    info['pefit_results']['fit_iter'] = l
    info['pefit_results']['chi2'] = chi2
    info['pefit_results']['N'] = opt[0]
    info['pefit_results']['tau'] = opt[1]
    info['pefit_results']['M'] = opt[2]
    info['pefit_results']['mean'] = opt[3]
    info['pefit_results']['mean_err'] = np.sqrt(cov[3][3])
    info['pefit_results']['sig'] = opt[4]
    info['pefit_results']['out_of_bounds'] = out_of_bounds

    adcPerMIP = opt[3] - pedestal
    adcPerMIP_err = np.sqrt(cov[3][3])
    info['pefit_results']['adcPerMIP'] = adcPerMIP
    info['pefit_results']['adcPerMIP_err'] = adcPerMIP_err
    info['adcPerMIP'] = adcPerMIP
    
    pePerMIP = (opt[3]-pedestal)/gain
    pePerMIP_err = np.sqrt(cov[3][3])/gain
    info['pefit_results']['pePerMIP'] = pePerMIP
    info['pefit_results']['pePerMIP_err'] = pePerMIP_err
    info['pePerMIP'] = pePerMIP
    
    if debug: print('MIP = {0} +/- {1} adc/MIP'.format(int(adcPerMIP), int(adcPerMIP_err)))
    if debug: print('MIP = {0} +/- {1} pe/MIP'.format(round(pePerMIP,1), round(pePerMIP_err,1)))
    
    
    if plot or save:
        
        #pmin = thresh
        pmin = fmin
        #pmax = 4000
        pmax = fmax

        fig, ax = plt.subplots(1,1,figsize=(18,6),facecolor='w',edgecolor='k')
        fig.suptitle(title, size=22)
        plt.sca(ax)

        plttext = [
            '{0}  adc/pe'.format(round(gain,1)),
            'thresh = {0} pe'.format(round(info['thresh']/gain,1)),
            '{0} +/- {1}  adc/mip'.format(round(adcPerMIP,1), round(adcPerMIP_err,1)),
            '{0} +/- {1}  pe/mip'.format(round(pePerMIP,1), round(pePerMIP_err,1)),
            'thresh = {0} mip'.format(round(info['thresh']/adcPerMIP,2)),
        ]
        
        Y = 0.8
        for text in plttext:
            plt.text(x=0.99, y=Y, s=text,
                     fontsize=18, horizontalalignment='right', transform=ax.transAxes)
            Y -= 0.07
        if plotText:
            Y -= 0.07 # add a space
            for text in plotText:
                plt.text(x=0.99, y=Y, s=text,
                     fontsize=18, horizontalalignment='right', transform=ax.transAxes)
                Y -= 0.07
                
        #plot_title = ('adc/pe = {0}  |  adc/mip = {1} +/- {2}  |  pe/mip = {3} +/- {4}'
        #              .format(round(gain,1),
        #                      round(adcPerMIP,1), round(adcPerMIP_err,1),
        #                      round(pePerMIP,1), round(pePerMIP_err,1)))
        #if title: plot_title = title+'  '+plot_title
        #plt.title(plot_title, size=20)

        plt.ylabel('counts', size=20)
        plt.xlabel('charge (adc)',size=20)
        plt.tick_params(axis='both', which='major', labelsize=18)

        plt.plot(range(len(hist)), 
                 hist, 
                 color='black', alpha=0.7,
                 #label='data'
                 )

        plt.plot(range(len(ped)), 
                 ped, 
                 color='purple', alpha=0.7,
                 #label='pedestal'
                 )

        plt.plot(range(pmin,pmax),
                 [expon(x, opt[0], opt[1]) for x in range(pmin, pmax)], 
                 color='green', 
                 #label='expo darknoise'
                 )

        plt.plot(range(pmin,pmax),
                 [landau(x, opt[2], opt[3], opt[4]) for x in range(pmin, pmax)], 
                 color='blue', 
                 #label='landau MIP'
                 )
        
        if info['bad_pefit_reasons']: reasons = info['bad_pefit_reasons']
        else: reasons = ''
        plt.plot(range(pmin, pmax), 
                 [expolandau(x, opt[0], opt[1], opt[2], opt[3], opt[4]) for x in range(pmin, pmax)], 
                 color='red', alpha=0.9, linestyle='-', lw=2,
                 #label='Fit Total\nR2 = {0}\nAt limit = {1}'.format(round(r2, 3), out_of_bounds)
                 label='good fit = {0}  {1}'.format(info['good_pefit'], reasons)
                 )

        plt.legend(fontsize=16, loc='upper right')
        plt.xlim(0, 5000)
        if normalize: plt.ylim(0.002)
        else: plt.ylim(1)
        plt.yscale('log')
        
        if save: plt.savefig(save, bbox_inches='tight')
        if not plot: plt.close()
        
    return info





def simpleFitMIP(hist, ped, config=False, title='', save='', plot=False, debug=False):
    
    if debug: print('\nRunning simpleFitMIP()')

    if config: plotText = reducedText(config)
    
    info = {}
    info['good_simplefit'] = True
    info['simplefit_results'] = False
    info['bad_simplefit_reasons'] = []
    
    # find the pedestal
    pedestal = findPed(ped)
    if debug: print('pedestal =', pedestal)
    info['pedestal'] = pedestal
    
    # estimate the peak position
    wmean = weighted_mean(hist[:-1], 0, len(hist[:-1]))
    
    try:
        pmean, psigma = prefit(hist, wmean, 0.7*wmean, 1.4*wmean)
        gmean = pmean
        lmbnd = pmean-(0.5*psigma)
        hmbnd = pmean+(0.5*psigma)
        gsigma = psigma
        lsbnd = 0.3*psigma
        hsbnd = 1.5*psigma
    except:
        print('WARNING: gaussian pre-fit failed, trying weighted mean')
        gmean = wmean
        lmbnd = int(pedestal)
        hmbnd = 1.5*wmean
        gsigma = 200
        lsbnd = 100
        hsbnd = wmean-int(pedestal)
        
    if debug: 
        print('mean guess =', gmean)
        print('sigma guess =', gsigma)
    
    
    # determine fit range
    fmin = int(pedestal+(0.4*(gmean-pedestal)))
    fmax = int(pedestal+(2.5*(gmean-pedestal)))
    if fmax > 4000:
        fmax = 4000

    M = max(hist)
    ###           M     mean     sig
    seed   = [    M,   gmean,  gsigma]
    lo_bnd = [    1,   lmbnd,   lsbnd]
    hi_bnd = [ M*10,   hmbnd,   hsbnd]
    
    try:
        opt, cov = curve_fit(landau,
                         range(fmin,fmax), 
                         hist[fmin:fmax],
                         p0=seed,
                         bounds=[lo_bnd, hi_bnd]
                         )
    except Exception as e:
        print('ERROR: simple mip pre-fit failed')
        print(e)
        info['good_simplefit'] = False
        info['bad_simplefit_reasons'].append('simple mip pre-fit failed')
        return info

    
    # refit again?
    for i in range(2):
        mipadc = opt[1] - pedestal

        # set new fmin and fmax
        fmin = int(opt[1] - 0.35*mipadc)
        if fmin < pedestal:
            fmin = pedestal
        fmax = int(opt[1] + 1.0*mipadc)
        if fmax > 4000:
            fmax = 4000

        # set new bounds
        # seems to want a slightly lower tau min bound
        ###         M          mean     sig
        seed   = [opt[0],     opt[1],  opt[2]]
        lo_bnd = [opt[0]/10,   fmin,      50]
        hi_bnd = [opt[0]*10,   fmax,    1000]

        # re fit
        try:
            opt, cov = curve_fit(landau,
                             list(range(fmin, fmax)), 
                             hist[fmin:fmax],
                             p0=seed,
                             bounds=[lo_bnd, hi_bnd]
                            )
        except:
            print('ERROR: simple mip re-fit failed')
            info['good_simplefit'] = False
            info['bad_simplefit_reasons'].append('simple mip re-fit failed')
            return info

        
    # sanity checks on bounds
    out_of_bounds = False
    for i in range(len(opt)):
        if sciround(opt[i]) <= sciround(lo_bnd[i]):
            print('WARNING: opt[{0}] at lower bound'.format(i))
            info['good_simplefit'] = False
            info['bad_simplefit_reasons'].append('opt[{0}] at lower bound'.format(i))
            out_of_bounds = True
        if sciround(opt[i]) >= sciround(hi_bnd[i]):
            print('WARNING: opt[{0}] at upper bound'.format(i))
            info['good_simplefit'] = False
            info['bad_simplefit_reasons'].append('opt[{0}] at upper bound'.format(i))
            out_of_bounds = True
    

    if debug: 
        print('Simple MIP Fit Results')
        print('fit range = {0} - {1}'.format(fmin, fmax))
        print('M    = {0}  [{1}, {2}]'.format(opt[0], int(lo_bnd[0]), int(hi_bnd[0])))
        print('mean = {0}  [{1}, {2}]'.format(opt[1], int(lo_bnd[1]), int(hi_bnd[1])))
        print('sig  = {0}  [{1}, {2}]'.format(opt[2], int(lo_bnd[2]), int(hi_bnd[2])))


    info['simplefit_results'] = {}
    info['simplefit_results']['N'] = 0
    info['simplefit_results']['tau'] = 0
    info['simplefit_results']['M'] = opt[0]
    info['simplefit_results']['mean'] = opt[1]
    info['simplefit_results']['mean_err'] = np.sqrt(cov[1][1])
    info['simplefit_results']['sig'] = opt[2]
    info['simplefit_results']['out_of_bounds'] = out_of_bounds

    adcPerMIP = opt[1] - pedestal
    adcPerMIP_err = np.sqrt(cov[1][1])
    info['simplefit_results']['adcPerMIP'] = adcPerMIP
    info['simplefit_results']['adcPerMIP_err'] = adcPerMIP_err
    info['adcPerMIP'] = adcPerMIP
    
    
    if plot or save:

        #pmin = thresh
        pmin = fmin
        #pmax = 4000
        pmax = fmax
        
        fig, ax = plt.subplots(1,1,figsize=(18,6),facecolor='w',edgecolor='k')
        fig.suptitle(title, size=22)
        plt.sca(ax)
        
        plttext = [
            '{0} +/- {1}  adc/mip'.format(round(adcPerMIP,1), round(adcPerMIP_err,1)),
        ]
        
        Y = 0.8
        for text in plttext:
            plt.text(x=0.99, y=Y, s=text,
                     fontsize=18, horizontalalignment='right', transform=ax.transAxes)
            Y -= 0.07
        if plotText:
            Y -= 0.07 # add a space
            for text in plotText:
                plt.text(x=0.99, y=Y, s=text,
                     fontsize=18, horizontalalignment='right', transform=ax.transAxes)
                Y -= 0.07
                
        #ptitle = ('adc/mip = {0} +/- {1}'
        #          .format(round(adcPerMIP,1), round(adcPerMIP_err,1)))
        #if title: ptitle = title+'  '+ptitle
        #plt.title(ptitle, size=20)
        
        plt.ylabel('counts', size=20)
        plt.xlabel('charge (adc)',size=20)
        plt.tick_params(axis='both', which='major', labelsize=18)
    
        plt.plot(range(len(hist)), 
                 hist, 
                 color='black', alpha=0.7,
                 #label='data'
                 )
        
        plt.plot(range(len(ped)), 
                 ped, 
                 color='purple', alpha=0.7,
                 #label='pedestal'
                 )
        
        if info['bad_simplefit_reasons']: reasons = info['bad_simplefit_reasons']
        else: reasons = ''
        plt.plot(range(pmin, pmax), 
                 [landau(x, opt[0], opt[1], opt[2]) for x in range(pmin, pmax)], 
                 color='red', alpha=0.9, linestyle='-', lw=2,
                 label='good fit = {0}  {1}'.format(info['good_simplefit'], reasons)
                 )
        
        plt.legend(fontsize=16, loc='upper right')
        plt.xlim(0, 5000)
        plt.ylim(1)
        plt.yscale('log')

        if save: plt.savefig(save, bbox_inches='tight')
        if not plot: plt.close()
        
    return info

