from threading import Thread
from queue import Queue
import time
import cv2
import logging
logger = logging.getLogger('multithread.py')



class ThreadedImageWriter(object):
    """
    Class that uses multithreading to buffer and save video frames
    """
    def __init__(self):
        self.stopped = False
        self.Q = Queue()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        logger.info(f"Threaded cv2.imwrite() started.")
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.Q.empty():
                front =  self.Q.get()
                cv2.imwrite(front['name'], front['frame'])
                logger.info(f"{front['name']} saved.")

            else:
                time.sleep(0.1)

    def save(self, name, frame):
        self.Q.put({'name': name, 'frame': frame})

    def running(self):
        return not self.stopped

    def stop(self):
        while not self.Q.empty():
            front =  self.Q.get()
            cv2.imwrite(front['name'], front['frame'])
            logger.info(f"{front['name']} saved.")

        self.stopped = True
        logger.info(f"Threaded cv2.imwrite() stopped.")
        self.thread.join()