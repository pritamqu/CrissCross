import os
from datasets.loader.backend_downstream.video_db import VideoDataset

class UCF(VideoDataset):
    def __init__(self, 
                 DATA_PATH, 
                 ANNO_PATH,
                 subset,
                 video_clip_duration=0.5,
                 return_video=True,
                 video_fps=16.,
                 video_transform=None,
                 return_audio=False,
                 return_labels=True,
                 return_index=True,
                 mode='clip',
                 clips_per_video=20,
                 ):

        assert return_audio is False
        self.name = 'UCF-101'
        self.root = DATA_PATH
        self.subset = subset

        classes_fn = os.path.join(ANNO_PATH, 'classInd.txt')
        self.classes = [l.strip().split()[1] for l in open(classes_fn)]

        # filenames = [ln.strip().split()[0] for ln in open(os.path.join(ANNO_PATH, subset+'.txt'))]
        # labels = [fn.split('/')[0] for fn in filenames]
        # or
                
        filenames = [ln.strip().split()[0].split('/') for ln in open(os.path.join(f'{ANNO_PATH}', f'{subset}.txt'))]
        filenames = [os.path.join(p[0], p[1]) for p in filenames] # somebullshit stuff to support windows and linux both paths
        labels = [fn.replace('\\', '/').split('/')[0] for fn in filenames]        
        
        labels = [self.classes.index(cls) for cls in labels]

        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(UCF, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_labels=return_labels,
            return_index=return_index,
            labels=labels,
            mode=mode,
            clips_per_video=clips_per_video,
        )
