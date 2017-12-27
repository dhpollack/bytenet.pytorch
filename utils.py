import os
import errno
import time
#import requests
#import bs4
#import chardet


def print_running_time(start):
    elapsed = time.time()-start
    print("time (s): {:0.2f}s".format(elapsed))
    return(elapsed)

def _make_dir_iff(d):
    try:
        os.makedirs(os.path.join(d))
        print("{} created".format(d))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def _download_extract(root, url, type="zip"):
    raise NotImplementedError
