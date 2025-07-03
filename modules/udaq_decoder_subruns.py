#!/usr/bin/env python

from struct import unpack
from cobs import cobs
import numpy as np
import argparse


# 2022-12-05 - fixed printing of cuptrig info
# 2022-12-05 - add printing of adc channel number
# 2023-07-17 - tweak the "OK" check
# 2023-07-21 - add start/end of stream not found warnings
# 2025-01-03 - revamping for new Dec_26_2025 format for
#              subruns and ascii moni data are the big changes


def main():
    aparse = argparse.ArgumentParser()
    aparse.add_argument(dest='infiles', type=str, nargs="+", help='file name(s)')
    aparse.add_argument('--skip', dest='skip', type=int, default=0, help='skip X bytes at beginning')
    aparse.add_argument('--nopps', dest='nopps', action='store_true', help='if no PPS to udaq')
    aparse.add_argument('--cobs', dest='cobs', action='store_true', help='cobs decode the data first')
    aparse.add_argument('--hv', dest='hv', action='store_true', help='if using the HV auto correction')
    aparse.add_argument('--debug', dest='debug', type=int, default=0, help='debug output level [0-3]')
    
    args = aparse.parse_args()

    data = payloadReader(args.infiles,
                                skip_bytes=args.skip,
                                no_pps=args.nopps,
                                cobs_decode=args.cobs,
                                hv_corr=args.hv,
                                debug=args.debug)

    events = len(data)
    duration = round(data[-1][0] - data[0][0], 1)
    if events > 0 and duration > 0:
        rate = round(events / duration, 1)
        print('{0} events in {1} seconds = {2} Hz'.format(events, duration, rate))
    else:
        print('{0} events in {1} seconds'.format(events, duration))
        
    return


def payloadReader(infiles, skip_bytes=0, no_pps=0, cobs_decode=0, hv_corr=0, debug=0):
    data = []
    if isinstance(infiles, str):
        infiles = [infiles]
    if isinstance(infiles, list):
        for infile in infiles:
            if len(infiles)>1: print('DECODE: '+infile.split('/')[-1])
            ud = udaqDecode(infile,
                            skip_bytes=skip_bytes,
                            no_pps=no_pps,
                            cobs_decode=cobs_decode,
                            hv_corr=hv_corr,
                            debug=debug)
            data.extend(ud.data)
    else:
        print('ERROR: Unsupported file type {0}, must be str or list'
              .format(type(infiles)))
    
    return data


class udaqDecode:

    OBJECT_CODE_PPS_SECOND   = 0xe0
    OBJECT_CODE_PPS_YEAR     = 0xe4
    OBJECT_CODE_TRIG_CONFIG  = 0xe5
    OBJECT_CODE_DATA_FORMAT  = 0xe6
    OBJECT_CODE_PAGE_END     = 0xe7
    OBJECT_CODE_HEADER       = 0xe8
    OBJECT_CODE_GENERIC      = 0xf0
    
    OBJECT_MASK_PPS_SECOND   = 0xfc
    OBJECT_MASK_GENERIC      = 0xf0
    
    STATUS_CPUTRIG_ACTIVE    = np.uint32(1<<5)
    
    DATA_FORMAT_TIMESTAMP               = 1
    DETAIL_TIMESTAMP_FINE               = (1<<0)
    DATA_FORMAT_TIMESTAMP_TOT_ADCS      = 2
    DATA_FORMAT_TIMESTAMP_TOT_ALL_CCRS  = 3
    
    N_ADCS = 3  # number of adcs to expect
    
    def __init__(self, infile, skip_bytes=0, no_pps=0, cobs_decode=0, hv_corr=0, debug=0):

        self.debug = debug
        # additional debugging
        self.print_pps = 0  # print all pps
        self.print_year = 0  # print all year

        self.no_pps = no_pps

        self.hv_corr = hv_corr
        
        with open(infile, 'rb') as binFile:
            binout = binFile.read()

        if cobs_decode:
            binout = cobsDecode(binout, debug=1)
        
        # remove "OK" bytes if they still exist
        #if 'OK' in binout[-4:].decode(errors='ignore'):
        if 'OK' == binout[-4:].decode(errors='ignore'):
            print('INFO: cutting out \"OK\" bytes --> {0}'.format(binout[-4:]))
            binout = binout[:-4]
        
        if skip_bytes:
            print('INFO: skipping first {0} bytes'.format(skip_bytes))
            binout = binout[skip_bytes:]

        if debug > 1:
            print('header e8xxxxxx - year e4xxxxxx - pps e0xxxxxx - cpu e5xxxxxx')
            print('INFO: word 1: {0}'.format(binout[:4][::-1].hex()))
            print('INFO: word 2: {0}'.format(binout[4:8][::-1].hex()))
            print('INFO: word 3: {0}'.format(binout[8:12][::-1].hex()))

        nwords = len(binout)/4.
        if nwords != int(nwords):
            extra_bytes = int((nwords-int(nwords))*4)
            print('WARNING: Number of bytes not a factor of 4. {0} extra bytes: {1}'
                  .format(extra_bytes, binout[-extra_bytes:][::1].hex()))

        self.nwords = int(nwords)
        self.binout = binout
        self.data = []
        
        self.hv_data = {}
        self.thresh_data = {}
        self.corr_keys = ['time', 'temp', 'value']
        for key in self.corr_keys:
            self.hv_data[key] = []
            self.thresh_data[key] = []
            
        self.decodeData()


    def getWord(self, i):
        return np.uint32(unpack('<I', self.binout[i*4:i*4+4])[0])

    
    def phex(self, word):
        # pretty print hex
        hword = hex(word)[2:]
        hword = hword.zfill(8)
        hword = ('{0} {1} {2} {3}'
                 .format(hword[0:2].lower(),
                         hword[2:4].lower(),
                         hword[4:6].lower(),
                         hword[6:8].lower()
                         ))
        return hword


    def pindx(self, index):
        # zero pad index number
        n_char = len(str(self.nwords))
        return str(index).rjust(n_char)


    def hvString2list(self, string, time):
        try:
            temp = float(string.split('temp:')[-1].split(',')[0].strip())
        except:
            temp = 0
        try:
            value = int(string.split('HV:')[-1].split('}')[0].strip())
        except:
            value = 0
        self.hv_data['time'].append(time)
        self.hv_data['temp'].append(temp)
        self.hv_data['value'].append(value)


    def threshString2list(self, string, time):
        try:
            temp = float(string.split('temp:')[-1].split(',')[0].strip())
        except:
            temp = 0
        try:
            value = int(string.split('THRESH:')[-1].split('}')[0].strip())
        except:
            value = 0
        self.thresh_data['time'].append(time)
        self.thresh_data['temp'].append(temp)
        self.thresh_data['value'].append(value)

        
    def decodeData(self):

        year                = np.uint16(0)
        prev_year           = np.uint16(0)
        pps                 = np.uint64(0)
        prev_pps            = np.uint64(0)
        format_subtype      = np.uint16(0)
        format_detail_bits  = np.uint16(0)
        cputrig             = np.uint16(0)

        t  = np.uint64(0)
        t0 = np.float64(0)
        t1 = np.float64(0)

        time_errors = 0
        type_errors = 0
        nadc_errors = 0
        miss_errors = 0
        vals_errors = 0
        lbmo_errors = 0
        asci_errors = 0

        hv_corrected = 0
        thresh_corrected = 0

        start_of_stream = 0
        end_of_stream = 0
        
        # skip first TIMESTAMP_TOT_ADCS because
        # it always has wrongs values, probably
        # a bug in the firmware initialization
        skip_first_hit = 0
        
        
        # loop over all the words
        index = 0
        while index < self.nwords:

            word = self.getWord(index)

            type_bits = np.uint8(word >> 24)
            
            if type_bits < 0xe0:

                t = ((np.uint64(word)* 25.) / 72.) / 1.e9
                t1 = t + pps
                # check for linear time
                if t0 > t1:
                    if self.debug > 1:
                        print('{0} [{1}] WARNING: non-linear time t1-t0 = {2}'
                              .format(self.phex(word), self.pindx(index), t1-t0))
                    time_errors += 1
                t0 = t1
                
                if format_subtype == self.DATA_FORMAT_TIMESTAMP:
                    #if self.debug > 2:
                    print('{0} [{1}] TIMESTAMP - should never see these'
                          .format(self.phex(word), index))

                elif format_subtype == self.DATA_FORMAT_TIMESTAMP_TOT_ADCS:
                    # init these to -1 because of the 0-charge issue! 2022-01-28
                    tot = -1
                    adc1 = -1
                    adc2 = -1
                    adc3 = -1
                    n_adcs = 0
                    
                    if self.debug > 2:
                        # show time as udaq + PPS
                        #print('{0} [{1}] TIMESTAMP_TOT_ADCS : time={2:.9f}'
                        #      .format(self.phex(word), self.pindx(index), t1))
                        # show time as recorded by udaq
                        print('{0} [{1}] TIMESTAMP_TOT_ADCS : offset={2} (time={3:.9f})'
                              .format(self.phex(word), self.pindx(index), t, t1))
                        
                    # grab next word
                    if index+1 < self.nwords:
                        index += 1
                        word = self.getWord(index)

                        n_adcs = (word >> 28) & 0xf
                        # n_adcs should always be == 3 for what we typically do
                        # remember each word contains 2 ADC values
                        if n_adcs != self.N_ADCS:
                            nadc_errors += 1
                            n_words = int(np.ceil((n_adcs-3)/2.))
                            if self.debug > 1:
                                print('{0} [{1}] WARNING: n_adcs = {2} ( != {3})'
                                      .format(self.phex(word), self.pindx(index), n_adcs, self.N_ADCS))
                            # sometimes gets out of sync so skip ahead? 2022-12-05
                            #index += 2
                            #continue
                        
                        # grab the Time-over-Threshold
                        tot = (word >> 16) & 0xfff

                        # grab the first adc
                        adc1 = (word & 0xfff)
                        # grab adc channel
                        ch1 = (word >> 12) & 0xf
                        if self.debug > 2:
                            print('{0} [{1}]     nadcs={2}  tot={3}  adc[{4}]={5}'
                                  .format(self.phex(word), self.pindx(index), n_adcs, tot, ch1, adc1))
                    else:
                        miss_errors += 1
                    
                    # grab the next word if more adcs
                    if n_adcs > 1 and index+1 < self.nwords:

                        index += 1
                        word = self.getWord(index)

                        adc2 = (word >> 16) & 0xfff
                        ch2 = (word >> 28) & 0xf
                        if n_adcs > 2:
                            adc3 = (word & 0xfff)
                            ch3 = (word >> 12) & 0xf
                        if self.debug > 2:
                            print('{0} [{1}]     adc[{2}]={3}  adc[{4}]={5}'
                              .format(self.phex(word), self.pindx(index), ch2, adc2, ch3, adc3))
                    else:
                        miss_errors += 1
                    
                    
                    
                    # check if adcs changed from -1
                    if n_adcs > 0 and adc1 == -1:
                        if self.debug > 1:
                            print('[{0}] WARNING: ADC-1 still at -1'.format(index))
                        vals_errors += 1
                    if n_adcs > 1 and adc2 == -1:
                        if self.debug > 1:
                            print('[{0}] WARNING: ADC-2 still at -1'.format(index))
                        vals_errors += 1
                    if n_adcs > 2 and adc3 == -1:
                        if self.debug > 1:
                            print('[{0}] WARNING: ADC-3 still at -1'.format(index))
                        vals_errors += 1
                    
                    # append the list of chargestamps
                    #----------------------------------------------------------
                    if n_adcs == 0:
                        index += 1
                        continue
                    if skip_first_hit:
                        skip_first_hit = 0
                        index += 1
                        continue
                    if self.hv_corr and hv_corrected < 1:
                        index += 1
                        continue
                    #print(t1, 'writing chargestamp')
                    self.data.append([t1, adc1, adc2, adc3, int(cputrig), tot])
                    #----------------------------------------------------------
                
                
                elif format_subtype == self.DATA_FORMAT_TIMESTAMP_TOT_ALL_CCRS:
                    # we should never see this in scint data
                    #if self.debug > 0:
                    print('{0} [{1}] TIMESTAMP_TOT_ALL_CCRS - should never see these'
                          .format(self.phex(word), index))
                    index += 17 
                    continue
                
                else:
                    #if self.debug > 0:
                    print('{0} [{1}] unknown format_subtype {2}'
                          .format(self.phex(word), self.pindx(index), hex(format_subtype)))
                    type_errors += 1


            elif type_bits == self.OBJECT_CODE_PPS_YEAR:
                # PPS year word
                year = np.uint16(word)
                if abs(year - prev_year) > 1 and prev_year:
                    print('[{0}] WARNING: year changed by {1} years'.format(self.pindx(index), year-prev_year))
                prev_year = year
                if self.debug > 0 or self.print_year:
                    print('{0} [{1}] PPS_YEAR : {2}'
                          .format(self.phex(word), self.pindx(index), year))

            elif (type_bits & self.OBJECT_MASK_PPS_SECOND) == self.OBJECT_CODE_PPS_SECOND:
                # PPS seconds word
                if self.no_pps:
                    # if using udaq without PPS, just increment by 1 to
                    # preserve linear time
                    pps += np.uint32(1)
                else:
                    pps = np.uint32(word & 0x03ffffff)
                if abs(pps - prev_pps) > 1 and prev_pps:
                #if abs(pps - prev_pps) > 1:
                    print('[{0}] WARNING: pps changed by {1} seconds'
                          .format(self.pindx(index), np.int64(pps) - np.int64(prev_pps)))
                #if prev_pps == 0:
                #   print('INFO: First PPS = {0}'.format(pps)) 
                prev_pps = pps
                if self.debug > 0 or self.print_pps:
                    print('{0} [{1}] PPS_SECOND : {2}'
                          .format(self.phex(word), self.pindx(index), pps))

            elif type_bits == self.OBJECT_CODE_DATA_FORMAT:
                # hit data format selector
                format_subtype = np.uint16((word & 0x00ff0000) >> 16)
                format_detail_bits = np.uint16(word & 0xffff)
                if self.debug > 0:
                    print('{0} [{1}] DATA_FORMAT : subtype={2}  detail={3}'
                          .format(self.phex(word), self.pindx(index), hex(format_subtype),
                                  hex(format_detail_bits)))

            elif type_bits == self.OBJECT_CODE_TRIG_CONFIG:
                # config bits, that tells if live (not cpu trigger mode)
                cputrig = ((word & self.STATUS_CPUTRIG_ACTIVE) != 0)
                if self.debug > 0:
                    print('{0} [{1}] TRIG_CONFIG : cpu_trig={2}'
                          .format(self.phex(word), self.pindx(index), cputrig))
                # Next word has the time
                index += 1
                word = self.getWord(index)
                timestamp = (((np.uint64(word)* 25.) / 72.) / 1.e9) + pps
                if self.debug > 0:
                    print('{0} [{1}]     time={2:.9f}'
                          .format(self.phex(word), self.pindx(index), timestamp))
                
            elif type_bits == self.OBJECT_CODE_HEADER:
                # header info is 1 word 
                
                # number of words in page
                num_words = ((word >> 10) & 0x3fff)
                
                # stream status
                # 0 = Start of Stream
                # 1 = Continuous running
                # 2 = buffer overflow
                # 3 = End of Stream
                status = ((word >> 8) & 0x3)
                
                # source id (given by field hub)
                sid = ((word) & 0xff)
                
                if self.debug > 0:
                    print('{0} [{1}] HEADER : nwords={2}  status={3}  sid={4}'
                          .format(self.phex(word), self.pindx(index), num_words, status, sid))
                    
                if status == 0:
                    start_of_stream = 1
                    #if self.debug > 1:
                    print('INFO: Start of Stream flagged')
                        
                if status == 1:
                    if self.debug > 2:
                        print('INFO: Continued running')
                        
                if status == 2:
                    lbmo_errors += 1
                    if self.debug > 1:
                        print('[{0}] WARNING: buffer overflow'.format(index))
                    
                if status == 3:
                    end_of_stream = 1
                    #if self.debug > 1:
                    print('INFO: End of Stream flagged')
                
            elif (type_bits & self.OBJECT_MASK_GENERIC) == self.OBJECT_CODE_GENERIC:
                num_words = np.uint8(word & 0xffffff)
                string = ''
                i1 = index
                for i in range(num_words-1):
                    index += 1
                    try:
                        string += self.binout[index*4:index*4+4].decode('ascii')
                    except:
                        string += self.binout[index*4:index*4+4].decode('ascii', errors='ignore')
                        word = self.getWord(index)
                        asci_errors += 1
                        if self.debug > 1:
                            print('{0} [{1}] WARNING: cannot decode to ascii'
                                  .format(self.phex(word), index))
                            
                if string.startswith('HV_'):
                    self.hvString2list(string, t1)
                    hv_corrected += 1
                    if hv_corrected == 1:
                        print('INFO: First HV correction at {0}'.format(t1))
                if string.startswith('THRESH_'):
                    self.threshString2list(string, t1)
                    thresh_corrected += 1
                    if thresh_corrected == 1:
                        print('INFO: First Thresh correction at {0}'.format(t1))

                if self.debug > 0:
                    print('{0} [{1}-{2}] GENERIC : time={3}'.format(self.phex(word), i1, index, t1))
                    print('     {0}'.format(string))

            else:
                #if self.debug > 0:
                print('{0} [{1}] unknown object type - skipping'
                      .format(self.phex(word), index))
                type_errors += 1
                
            index += 1


        ### show some info counts
        if hv_corrected: print('INFO: number of HV corrections = {0}'.format(hv_corrected))
        if thresh_corrected: print('INFO: number of Threshold corrections = {0}'.format(thresh_corrected))
        
        ### show total errors
        if time_errors: print('WARNING: non-linear time errors = {0}'.format(time_errors))
        if nadc_errors: print('WARNING: n adcs != {1} errors = {0}'.format(nadc_errors, self.N_ADCS))
        if miss_errors: print('WARNING: missing adc errors = {0}'.format(miss_errors))
        if type_errors: print('WARNING: object type errors = {0}'.format(type_errors))
        if vals_errors: print('WARNING: -1 charge errors = {0}'.format(vals_errors))
        if lbmo_errors: print('WARNING: buffer overflows = {0}'.format(lbmo_errors))
        if asci_errors: print('WARNING: decode ascii errors = {0}'.format(asci_errors))

        ### start/stop flags not found?
        #if not start_of_stream: print('WARNING: start of stream flag not found')
        #if not end_of_stream: print('WARNING: end of stream flag not found')
        
        return

        
def cobsDecode(binary, debug=0):
                
    # find all the frame markers
    markers = []
    for i, val in enumerate(binary):
        bval = val.to_bytes(length=1, byteorder='little')
        if bval == b'\x00':
            markers.append(i)
    if debug: print('COBS: found', len(markers)/2., 'frames')
    
    alldata = bytearray()
    for i in range(0, len(markers), 2):
        
        # grab the frame markers
        fstart = markers[i]
        fstop = markers[i+1]
        
        # select the frame
        cdata = binary[fstart+1:fstop]
        
        # cobs decode the frame
        data = cobs.decode(cdata)
        
        # grab the checksum for maybe checking later - trailing 2 bytes
        #cs = data[-2:]
        #if debug: print(cs)

        # skip "number of messages" frame
        if len(data) == 5:
            if debug: print('COBS: skipping message frame --> {0}'.format(data))
            continue
        
        # skip "OK" frame
        if 'OK' in data.decode(errors="ignore"):
            if debug: print('COBS: skipping \"OK\" frame --> {0}'.format(data))
            continue
        
        # strip off the BusID - first 1 byte
        # strip off the checksum - trailing 2 bytes
        alldata.extend(bytearray(data[1:-2]))
        
    return alldata



if __name__ == "__main__":
    main()
