'''tim.py'''
import time

class Timer(object):
    def __init__(self):
        self.last = time.time()
        self.start = self.last
    
    def print_since_last(self, msg):
        now = time.time()
        diff = now - self.last
        print(msg, diff)
        self.last = now
        return diff
    
    def print_since_start(self, msg):
        now = time.time()
        diff = now - self.start
        print(msg, diff)
        self.last = now
        return diff
    
    def since_last(self):
        now = time.time()
        diff = now - self.last
        self.last = now
        return diff

    def since_start(self):
        now = time.time()
        diff = now - self.start
        self.last = now
        return diff
    
    def reset(self):
        self.last = time.time()
        self.start = self.last

    def set_last(self):
        self.last = time.time()
    
