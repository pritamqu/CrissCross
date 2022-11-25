import os

def my_paths(server, db):
    
    
    if server == 'vector':
        output_dir = '/scratch/ssd002/home/pritam/projects/OUTPUTS'
        data_dir = fetch_vector_db(db)
    elif server == 'local':
        output_dir = '/mnt/PS6T/OUTPUTS'
        data_dir = fetch_local_db(db)
        
    return output_dir, data_dir

def fetch_vector_db(db):
    
    # pretraining datasets
    if db == 'audioset':
        return '/scratch/ssd004/datasets/AudioSet'
    elif db == 'kinetics400': 
        return '/scratch/ssd004/datasets/Kinetics_All/kinetics400'
    elif db == 'kinetics_sound': 
        return '/scratch/ssd004/datasets/Kinetics_All/kinetics400'
    else:
        raise NotImplementedError
    

def fetch_local_db(db):
    
    if db == 'kinetics400':
        return '/mnt/PS6T/datasets/Video/kinetics/kinetics400'
    else:
        raise NotImplementedError
        
