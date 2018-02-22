"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import errno
import logging
import os


class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        self.data_root = '/home/siyuan/data/CAD120/'
        self.project_root = '/home/siyuan/projects/papers/iccv2017/'
        self.metadata_root = '/home/siyuan/projects/papers/iccv2017/src/metadata/'
        self.tmp_root = '/home/siyuan/projects/papers/iccv2017/tmp/'

    def path_huang(self):   # change to huang's path
        self.data_root = '/home/siyuan/Documents/Dataset/CAD120/'
        self.project_root = '/home/siyuan/Dropbox/Code/iccv2017'
        self.metadata_root = '/home/siyuan/Documents/iccv2017/'


def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
