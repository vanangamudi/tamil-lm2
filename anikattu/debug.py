import os
import psutil
process = psutil.Process(os.getpid())

def memory_consumed():
    return process.memory_info().rss
