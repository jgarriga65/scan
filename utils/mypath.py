"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200', '2014', '2015'}
        assert(database in db_names)

        if database == 'cifar-10':
            return './vdisk/scan/datasets/cifar10/'
        
        elif database == 'cifar-20':
            return './vdisk/scan/datasets/cifar20/'

        elif database == 'stl-10':
            return '/path/to/stl-10/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/path/to/imagenet/'

        elif database == '2014':
            return './vdisk/data/2014/'

        elif database == '2015':
            return './vdisk/data/2015/'
        
        else:
            raise NotImplementedError
