# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os

def my_paths(server, db):

    if server == 'location':
        output_dir = '/scratch/ssd002/home/pritam/projects/OUTPUTS'
        data_dir = fetch_db(db)
    else:
        NotImplementedError
        
    return output_dir, data_dir

def fetch_db(db):
    
    if db == 'ucf101':
        return '/scratch/ssd004/datasets/UCF101'
    elif db == 'hmdb51':
        return '/scratch/ssd004/datasets/HMDB51'
    elif db == 'esc50':
        return '/scratch/ssd004/datasets/ESC-50'
    elif db == 'dcase':
        return '/scratch/ssd004/datasets/DCASE'
    else:
        raise NotImplementedError



