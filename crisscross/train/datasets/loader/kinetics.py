import os
import glob
import numpy as np
import random
import ffmpeg
import json
from joblib import Parallel, delayed
from datasets.loader.backend_pretext.video_db import VideoDataset

# DATA_PATH = '/data/datasets/kinetics/'

def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream and video_stream and float(video_stream['duration']) > 1.1 and float(audio_stream['duration']) > 1.1:
            return True
        else:
            return False
    except:
        return False

def filter_videos(vid_paths):
    all_indices = Parallel(n_jobs=-1)(delayed(valid_video)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_files = ['/'.join(vid_paths[i].replace('\\', '/').split('/')[-2:]) for i, val in enumerate(all_indices) if val]
    return valid_files

class Kinetics(VideoDataset):
    def __init__(self,
                 DATA_PATH,
                 subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=100,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 max_offsync_augm=0,
                 mode='clip',
                 submode=None,
                 clips_per_video=1,
                 ):

        ROOT = os.path.join(f"{DATA_PATH}", f"{subset}")
        classes = sorted(os.listdir(ROOT))
        
        CACHE_FILE = os.path.join(os.getcwd(), 'datasets', 'cache', 'kinetics400', f"{subset}.txt")
        # CACHE_FILE--> labels/videoname.avi,
        
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                files=json.loads(f.read())
        else:
            all_files = [fn for fn in glob.glob(os.path.join(f"{DATA_PATH}",f"{subset}", "*","*.avi"))]
            files = filter_videos(all_files) # load files that has both audios and videos
            with open(CACHE_FILE, 'w') as f:
                f.write(json.dumps(files))   
        
        filenames = files
        labels = [classes.index(fn.replace('\\', '/').split('/')[-2]) for fn in filenames]
  
        super(Kinetics, self).__init__(
            return_video=return_video,
            video_root=ROOT,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=ROOT,
            audio_fns=filenames,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            submode=submode,
            clips_per_video=clips_per_video,
            max_offsync_augm=max_offsync_augm,
        )

        self.name = 'Kinetics dataset'
        self.root = ROOT
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
