import os
import numpy as np
import pandas as pd
from datasets.loader.backend_downstream.audio_db import AudioDataset # VideoDataset

class ESC(AudioDataset):
    def __init__(self,
                 DATA_PATH,
                 ANNO_PATH,
                 subset,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=94,
                 audio_transform=None,
                 return_labels=True,
                 return_index=True,
                 mode='clip',
                 clips_per_audio=10,
                 ):


        # subset= 'train-1' or 'val-1'
        meta = pd.read_csv(os.path.join(f"{ANNO_PATH}", 'esc50.csv'))
        split, fold = subset.split('-')
        if split == 'train':
            filenames = meta[meta.fold.values!=int(fold)].filename.values
        elif split == 'test':
            filenames = meta[meta.fold.values==int(fold)].filename.values
        class_names = list(np.unique(meta.category.values).astype(str))
        labels = [int(fn[:-4].split('-')[-1]) for fn in filenames]
 
        super(ESC, self).__init__(
            return_audio=True,
            audio_root=DATA_PATH,
            audio_fns=filenames,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_audio,
        )

        self.name = 'ESC-50'
        self.root = DATA_PATH
        self.classes = class_names
        self.num_audios = len(filenames)
        self.num_classes = len(class_names)

        self.sample_id = os.path.join(DATA_PATH, filenames[0])
