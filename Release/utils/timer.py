import time
import logging
logger = logging.getLogger('timer.py')



class Timer(object):
    def __init__(self, name):
        self.name = name
        self.elapsed = 0.0
        self.counter = 0.0
        self.state = None

    def tic(self):
        self.reference_time = time.time()

    def toc(self):
        self.elapsed += time.time() - self.reference_time
        self.counter += 1

    def summary(self):
        if self.counter > 0:
            ms_per_frame = self.elapsed/self.counter
            print(f'{self.name} : {1000*ms_per_frame} ms/frame')
            logger.info(f'{self.name} : {1000*ms_per_frame} ms/frame')