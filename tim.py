'''tim.py'''
import time

class Tim(object):
    def __init__(self):
        timstamp = Timstamp("start")
        self.timstamps = [timstamp]
    
    def print_since_last(self, msg):
        timstamp =Timstamp(msg)
        diff = timstamp.start - self.timstamps[-1].start
        self.timstamps.append(timstamp)
        print(msg, diff)
        return diff
    
    def print_since_start(self, msg):
        timstamp =Timstamp(msg)
        diff = timstamp.start - self.timstamps[0].start 
        self.timstamps.append(timstamp)
        print(msg, diff)
        return diff
    
    def print_since(self, msg, i):
        timstamp =Timstamp(msg)
        diff = timstamp.start - self.timstamps[i].start
        self.timstamps.append(timstamp)
        print(msg, diff)
        return diff
    
    def add_timstamp(self, msg):
        timstamp =Timstamp(msg)
        self.timstamps.append(timstamp)
    
    def __getitem__(self, i):
        return self.timstamps[i].start

class Timstamp(object):
    def __init__(self,msg):
        self.msg = msg
        self.start = time.time()