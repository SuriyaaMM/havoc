import logging
import sys
import json
import os

class loggingSubsystem(object):
    logDir = None
    baseLogger = None
    consoleLogger = None
    fileLogger = None
    formatter = None

    def __init__(self):
        cfg = None
        with open("cfg.json") as f:
            cfg = json.load(f)

        self.logDir = cfg["LOG_DIR"]
        self.baseLogger = logging.getLogger()
        self.baseLogger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def setConsoleLogger(self, stream = sys.stdout):
        self.consoleLogger = logging.StreamHandler(stream)
        self.consoleLogger.setFormatter(self.formatter)
        self.baseLogger.addHandler(self.consoleLogger)
        pass
    
    def setFileLogger(self, fileName: str):
        self.fileLogger = logging.FileHandler(os.path.join(self.logDir, fileName))
        self.fileLogger.setFormatter(self.formatter)
        self.baseLogger.addHandler(self.fileLogger)
        pass

    def getBaseLogger(self) -> logging.Logger:  
        return self.baseLogger

LogSubsystem = loggingSubsystem()