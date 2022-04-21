import os
import glob
from datasets.loader.backend_downstream.audio_db import AudioDataset # VideoDataset

# DATA_PATH = "D:\\datasets\\Audio\\DCASE\\"


class DCASE(AudioDataset):
    def __init__(self,
                 DATA_PATH,
                 subset='train',
                 audio_clip_duration=2.,
                 audio_fps=None,
                 audio_fps_out=100,
                 audio_transform=None,
                 return_labels=True,
                 return_index=True,
                 mode='clip',
                 clips_per_audio=60,
                 ):


        # subset= 'train' or 'val'
        DATA_PATH = os.path.join(DATA_PATH, 'set1') # we only need the acoustic scene classification dataset
        if subset == 'train':
            DATA_PATH = os.path.join(DATA_PATH, 'scenes_stereo', 'scenes_stereo')
        elif subset == 'val':
            DATA_PATH = os.path.join(DATA_PATH, 'scenes_stereo_testset', 'scenes_stereo_testset')
        else:
            ValueError(f'subset {subset} is not correct.')
            
        filenames = [fn.replace('\\', '/').split('/')[-1] for fn in glob.glob(f"{DATA_PATH}/*.wav")]
        # filenames = [fn for fn in glob.glob(f"{DATA_PATH}/*.wav")]
        class_names = sorted(set([fn[:-6] for fn in filenames]))
        labels = [fn[:-6] for fn in filenames]
        labels = [class_names.index(cls) for cls in labels]

        super(DCASE, self).__init__(
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

        self.name = 'DCASE'
        self.root = DATA_PATH
        self.classes = class_names
        self.num_audios = len(filenames)
        self.num_classes = len(class_names)

        self.sample_id = os.path.join(DATA_PATH, filenames[0])
