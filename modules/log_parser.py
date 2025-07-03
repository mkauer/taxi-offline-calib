import os
import datetime


# 2023-05-09 - add hv and thresh ctrl to spool info parsing
# 2023-07-27 - add getInfoByUID()


# parse the logs created from Tims field_hub package
# keyed by channel number
def getSpoolInfo(logfile, info=None):
    if not info:
        info = {}
    if 'channels' not in info:
        info['channels'] = []

    runNum = os.path.basename(logfile).split('_')[1]
    
    with open(logfile) as lf:
        lines = lf.readlines()

    for i, line in enumerate(lines):
        if line.startswith('channel['):
            chan = int(line[8])
            if chan not in info['channels']:
                info['channels'].append(chan)
                info[str(chan)] = {}
            info[str(chan)]['runNum'] = runNum
            
            if ']: uid:' in line:
                uid = lines[i+1].strip().split()
                uid = ''.join(uid).zfill(24)
                info[str(chan)]['uid'] = uid
                
            if '] config:' in line:
                thresh = int(lines[i+1].strip().split(':')[-1])
                volt = int(lines[i+2].strip().split(':')[-1])
                info[str(chan)]['voltage'] = volt
                info[str(chan)]['threshold'] = thresh
                
            if '] monitor:' in line:
                t0 = float(lines[i+1].strip().split(':')[-1]) - 273.15
                t1 = float(lines[i+2].strip().split(':')[-1]) - 273.15
                t2 = float(lines[i+3].strip().split(':')[-1]) - 273.15
                info[str(chan)]['temperature'] = [t0, t1, t2]
                
            if '] run start:' in line:
                year = int(lines[i+1].strip().split(':')[-1])
                secs = int(lines[i+2].strip().split(':')[-1])
                usec = int(lines[i+3].strip().split(':')[-1])
                info[str(chan)]['daq_time'] = [year, secs, usec]
                dt = datetime.datetime(year=year, month=1, day=1) + datetime.timedelta(seconds=secs)
                info['date'] = str(dt.date())
                info['time'] = str(dt.time())
                
            if '] run stats:' in line:
                counts = 0
                overflows = 0
                duration_ms = 0
                rate = 0
                for j in range(1, 5):
                    if i+j >= len(lines): continue
                    if 'counts' in lines[i+j]:
                        counts = int(lines[i+j].strip().split(':')[-1])
                    if 'overflow' in lines[i+j]:
                        overflows = int(lines[i+j].strip().split(':')[-1])
                    if 'duration' in lines[i+j]:
                        duration_ms = int(lines[i+j].strip().split(':')[-1])
                    if 'rate' in lines[i+j]:
                        rate = round(float(lines[i+j].strip().split(':')[-1]), 1)
                info[str(chan)]['counts'] = counts
                info[str(chan)]['overflows'] = overflows
                info[str(chan)]['duration'] = round(duration_ms/1000., 1)
                info[str(chan)]['trigrate'] = rate

            if '] hv ctrl config:' in line:
                hv_ctrl_enabled = None
                mip_setpoint = None
                for j in range(1, 10):
                    if i+j >= len(lines): continue
                    if 'enabled' in lines[i+j]:
                        hv_ctrl_enabled = int(lines[i+j].strip().split(':')[-1])
                    if 'mip_setpoint' in lines[i+j]:
                        mip_setpoint = int(lines[i+j].strip().split(':')[-1])
                info[str(chan)]['hv_ctrl_enabled'] = hv_ctrl_enabled
                info[str(chan)]['mip_setpoint'] = mip_setpoint

            if '] hv ctrl status:' in line:
                hv_ctrl_temp = None
                hv_ctrl_hv = None
                for j in range(1, 13):
                    if i+j >= len(lines): continue
                    if 'temp' in lines[i+j]:
                        hv_ctrl_temp = float(lines[i+j].strip().split(':')[-1])
                    if 'hv_applied' in lines[i+j]:
                        hv_ctrl_hv = int(lines[i+j].strip().split(':')[-1])
                info[str(chan)]['hv_ctrl_temp'] = hv_ctrl_temp
                info[str(chan)]['hv_ctrl_hv'] = hv_ctrl_hv
                if hv_ctrl_hv is not None:
                    info[str(chan)]['voltage'] = hv_ctrl_hv
                    
            if '] thresh ctrl config:' in line:
                thresh_ctrl_enabled = None
                thresh_setpoint = None
                for j in range(1, 10):
                    if i+j >= len(lines): continue
                    if 'enabled' in lines[i+j]:
                        thresh_ctrl_enabled = int(lines[i+j].strip().split(':')[-1])
                    if 'thresh_setpoint' in lines[i+j]:
                        thresh_setpoint = int(lines[i+j].strip().split(':')[-1])
                info[str(chan)]['thresh_ctrl_enabled'] = thresh_ctrl_enabled
                info[str(chan)]['thresh_setpoint'] = thresh_setpoint

            if '] thresh ctrl status:' in line:
                thresh_ctrl_temp = None
                thresh_ctrl_thresh = None
                for j in range(1, 13):
                    if i+j >= len(lines): continue
                    if 'temp' in lines[i+j]:
                        thresh_ctrl_temp = float(lines[i+j].strip().split(':')[-1])
                    if 'thresh_applied' in lines[i+j]:
                        thresh_ctrl_thresh = int(lines[i+j].strip().split(':')[-1])
                info[str(chan)]['thresh_ctrl_temp'] = thresh_ctrl_temp
                info[str(chan)]['thresh_ctrl_thresh'] = thresh_ctrl_thresh
                if thresh_ctrl_thresh is not None:
                    info[str(chan)]['threshold'] = thresh_ctrl_thresh

            
    return info


# keyed by UID
def getInfoByUID(logfile, info=None):
    if not info:
        info = {}
    
    runNum = os.path.basename(logfile).split('_')[1]
    
    with open(logfile) as lf:
        lines = lf.readlines()

    for i, line in enumerate(lines):
        if line.startswith('channel['):

            chan = int(line[8])
            
            if ']: uid:' in line:
                uid = lines[i+1].strip().split()
                uid = ''.join(uid).zfill(24)
                if uid not in info:
                    info[uid] = {}
                    info[uid]['channel'] = chan
                    info[uid]['runNum'] = runNum
            
            if '] config:' in line:
                thresh = int(lines[i+1].strip().split(':')[-1])
                volt = int(lines[i+2].strip().split(':')[-1])
                info[uid]['voltage'] = volt
                info[uid]['threshold'] = thresh
                
            if '] monitor:' in line:
                t0 = float(lines[i+1].strip().split(':')[-1]) - 273.15
                t1 = float(lines[i+2].strip().split(':')[-1]) - 273.15
                t2 = float(lines[i+3].strip().split(':')[-1]) - 273.15
                info[uid]['temperature'] = [t0, t1, t2]
                
            if '] run start:' in line:
                year = int(lines[i+1].strip().split(':')[-1])
                secs = int(lines[i+2].strip().split(':')[-1])
                usec = int(lines[i+3].strip().split(':')[-1])
                info[uid]['daq_time'] = [year, secs, usec]
                dt = datetime.datetime(year=year, month=1, day=1) + datetime.timedelta(seconds=secs)
                info[uid]['date'] = str(dt.date())
                info[uid]['time'] = str(dt.time())
                
            if '] run stats:' in line:
                counts = 0
                overflows = 0
                duration_ms = 0
                rate = 0
                for j in range(1, 5):
                    if i+j >= len(lines): continue
                    if 'counts' in lines[i+j]:
                        counts = int(lines[i+j].strip().split(':')[-1])
                    if 'overflow' in lines[i+j]:
                        overflows = int(lines[i+j].strip().split(':')[-1])
                    if 'duration' in lines[i+j]:
                        duration_ms = int(lines[i+j].strip().split(':')[-1])
                    if 'rate' in lines[i+j]:
                        rate = round(float(lines[i+j].strip().split(':')[-1]), 1)
                info[uid]['counts'] = counts
                info[uid]['overflows'] = overflows
                info[uid]['duration'] = round(duration_ms/1000., 1)
                info[uid]['trigrate'] = rate

            if '] hv ctrl config:' in line:
                hv_ctrl_enabled = None
                mip_setpoint = None
                for j in range(1, 10):
                    if i+j >= len(lines): continue
                    if 'enabled' in lines[i+j]:
                        hv_ctrl_enabled = int(lines[i+j].strip().split(':')[-1])
                    if 'mip_setpoint' in lines[i+j]:
                        mip_setpoint = int(lines[i+j].strip().split(':')[-1])
                info[uid]['hv_ctrl_enabled'] = hv_ctrl_enabled
                info[uid]['mip_setpoint'] = mip_setpoint

            if '] hv ctrl status:' in line:
                hv_ctrl_temp = None
                hv_ctrl_hv = None
                for j in range(1, 13):
                    if i+j >= len(lines): continue
                    if 'temp' in lines[i+j]:
                        hv_ctrl_temp = float(lines[i+j].strip().split(':')[-1])
                    if 'hv_applied' in lines[i+j]:
                        hv_ctrl_hv = int(lines[i+j].strip().split(':')[-1])
                info[uid]['hv_ctrl_temp'] = hv_ctrl_temp
                info[uid]['hv_ctrl_hv'] = hv_ctrl_hv
                if hv_ctrl_hv is not None:
                    info[uid]['voltage'] = hv_ctrl_hv
                    
            if '] thresh ctrl config:' in line:
                thresh_ctrl_enabled = None
                thresh_setpoint = None
                for j in range(1, 10):
                    if i+j >= len(lines): continue
                    if 'enabled' in lines[i+j]:
                        thresh_ctrl_enabled = int(lines[i+j].strip().split(':')[-1])
                    if 'thresh_setpoint' in lines[i+j]:
                        thresh_setpoint = int(lines[i+j].strip().split(':')[-1])
                info[uid]['thresh_ctrl_enabled'] = thresh_ctrl_enabled
                info[uid]['thresh_setpoint'] = thresh_setpoint

            if '] thresh ctrl status:' in line:
                thresh_ctrl_temp = None
                thresh_ctrl_thresh = None
                for j in range(1, 13):
                    if i+j >= len(lines): continue
                    if 'temp' in lines[i+j]:
                        thresh_ctrl_temp = float(lines[i+j].strip().split(':')[-1])
                    if 'thresh_applied' in lines[i+j]:
                        thresh_ctrl_thresh = int(lines[i+j].strip().split(':')[-1])
                info[uid]['thresh_ctrl_temp'] = thresh_ctrl_temp
                info[uid]['thresh_ctrl_thresh'] = thresh_ctrl_thresh
                if thresh_ctrl_thresh is not None:
                    info[uid]['threshold'] = thresh_ctrl_thresh

            
    return info



### helper class for parsing info out of the logfile
# this is for matts script "run_tdaq.sh" for collecting
# data via the udaq_terminal and writing info to logfile
class LogParse:
    def __init__(self, logfile):
        self.logfile = logfile
        self.info = {}
        self.depth = 10
        passed = True
        if not self.readLog():
            return
        self.getRun()
        self.getFirmware()
        self.getVersion()
        self.getChan()
        self.getDate()
        self.getTime()
        self.getVolts()
        self.getThresh()
        self.getRuntime()
        self.getUID()
        self.getMon()
        self.getDaqTime()
        self.getDaqPPS()
        self.getRunStats()

        
    def readLog(self):
        if not os.path.exists(self.logfile):
            print('ERROR: file not found --> {0}'.format(self.logfile))
            return False
        with open(self.logfile) as lf:
            self.lines = lf.readlines()
        return True

    
    def checkChanObj(self):
        if str(self.chan) not in self.info:
            self.info[str(self.chan)] = {}

            
    def getRun(self):
        for line in self.lines:
            if line.startswith('rundir'):
                run = line.strip().split('/')[-1]
                self.info['run'] = run
                return True
        print('WARN: run not found')
        self.info['run'] = None
        return False

    
    def getVersion(self):
        for line in self.lines:
            if line.startswith('script version'):
                ver = line.strip().split('=')[-1]
                self.info['version'] = ver
                return True
        print('WARN: version not found')
        self.info['version'] = None
        return False

    
    def getDate(self):
        for line in self.lines:
            if line.startswith('date'):
                date = line.strip().split('=')[-1]
                self.info['date'] = date
                return True
        print('WARN: date not found')
        self.info['date'] = None
        return False

    
    def getTime(self):
        for line in self.lines:
            if line.startswith('time'):
                time = line.strip().split('=')[-1]
                self.info['time'] = time
                return True
        print('WARN: time not found')
        self.info['time'] = None
        return False

    
    def getChan(self):
        for line in self.lines:
            if line.startswith('channel'):
                if 'channels' not in self.info:
                    self.info['channels'] = []
                self.chan = int(line.strip().split('=')[-1])
                self.info['channels'].append(self.chan)
                return True
        print('WARN: channels not found')
        return False

    
    def getVolts(self):
        for line in self.lines:
            if line.startswith('voltage'):
                voltage = int(line.strip().split('=')[-1])
                self.checkChanObj()
                self.info[str(self.chan)]['voltage'] = voltage
                return True
        print('WARN: voltage not found')
        self.info[str(self.chan)]['voltage'] = None
        return False

    
    def getThresh(self):
        for line in self.lines:
            if line.startswith('threshold'):
                threshold = int(line.strip().split('=')[-1])
                self.checkChanObj()
                self.info[str(self.chan)]['threshold'] = threshold
                return True
        print('WARN: threshold not found')
        self.info[str(self.chan)]['threshold'] = None
        return False

    
    def getRuntime(self):
        for line in self.lines:
            if line.startswith('runtime'):
                runtime = int(line.strip().split('=')[-1])
                self.info['runtime'] = runtime
                return True
        print('WARN: runtime not found')
        self.info['runtime'] = None
        return False

    
    def getFirmware(self):
        for line in self.lines:
            if line.startswith('firmware'):
                firmware = (line.strip().split('=')[-1])
                self.info['firmware'] = firmware
                return True
        print('WARN: firmware not found')
        self.info['firmware'] = None
        return False

    
    def getUID(self):
        for i, line in enumerate(self.lines):
            if line.startswith('GET_UID'):
                for n in range(i, i+self.depth):
                    if self.lines[n].startswith('>') and 'Goodbye' not in self.lines[n]:
                        uid = self.lines[n].strip()[1:].split()
                        uid = ''.join(uid)
                        uid = uid.zfill(24)
                        self.checkChanObj()
                        self.info[str(self.chan)]['uid'] = uid
                        return True
        print('WARN: uid not found')
        self.info[str(self.chan)]['uid'] = None
        return False

    
    def getMon(self):
        for i, line in enumerate(self.lines):
            if line.startswith('GETMON'):
                for n in range(i, i+self.depth):
                    if self.lines[n].startswith('>') and 'Goodbye' not in self.lines[n]:
                        mon = self.lines[n].strip()[1:].split()
                        temps = [round(float(mon[0])-273.15, 2), 
                                 round(float(mon[1])-273.15, 2),
                                 round(float(mon[2])-273.15, 2)]
                        self.checkChanObj()
                        self.info[str(self.chan)]['temperature'] = temps
                        return True
        print('WARN: temps not found')
        self.info[str(self.chan)]['temperature'] = [None, None, None]
        return False

    
    def getDaqTime(self):
        for i, line in enumerate(self.lines):
            if line.startswith('PRINT_TIME'):
                for n in range(i, i+self.depth):
                    if self.lines[n].startswith('>') and 'Goodbye' not in self.lines[n]:
                        dtime = self.lines[n].strip()[1:].split()
                        self.checkChanObj()
                        self.info[str(self.chan)]['daq_time'] = dtime
                        return True
        print('WARN: daq time not found')
        self.info[str(self.chan)]['daq_time'] = None
        return False

    
    def getDaqPPS(self):
        for i, line in enumerate(self.lines):
            if line.startswith('PRINT_PPS'):
                for n in range(i, i+self.depth):
                    if self.lines[n].startswith('>') and 'Goodbye' not in self.lines[n]:
                        pps = self.lines[n].strip()[1:].split('\t')
                        self.checkChanObj()
                        self.info[str(self.chan)]['daq_pps'] = pps
                        return True
        print('WARN: daq pps not found')
        self.info[str(self.chan)]['daq_pps'] = None
        return False

    
    def getRunStats(self):
        for i, line in enumerate(self.lines):
            if line.startswith('GET_RUN_STATISTICS'):
                for n in range(i, i+self.depth):
                    if self.lines[n].startswith('>') and 'Goodbye' not in self.lines[n]:
                        triggers = float(self.lines[n].strip()[1:].split()[-1])
                        time = float(self.lines[n+1].strip().split()[-1])
                        trigrate = round(triggers/time, 2)
                        self.checkChanObj()
                        self.info[str(self.chan)]['trigrate'] = trigrate
                        return True
        print('WARN: run stats not found')
        self.info[str(self.chan)]['trigrate'] = None
        return False

