import os
import random
import torch
import numpy as np
import torch.utils.data as data
from datasets.loader.backend_downstream import av_wrappers
from collections import defaultdict

def chararray(fn_list):
    charr = np.chararray(len(fn_list), itemsize=max([len(fn) for fn in fn_list]))
    for i in range(len(fn_list)):
        charr[i] = fn_list[i]
    return charr


class VideoDataset(data.Dataset):
    def __init__(self,
                 return_video=True,
                 video_root=None,
                 video_fns=None,
                 video_clip_duration=0.5,
                 video_fps=16,
                 video_transform=None,
                 return_labels=False,
                 labels=None,
                 return_index=False,
                 mode='clip', # video
                 clips_per_video=1,
                 ):
        super(VideoDataset, self).__init__()
        self.num_samples = 0
        self.return_video = return_video
        self.video_root = video_root
        if return_video:
            self.video_fns = chararray(video_fns)
            self.num_samples = self.video_fns.shape[0]
        self.video_fps = video_fps
        self.video_transform = video_transform

        self.return_labels = return_labels
        if return_labels:
            self.labels = np.array(labels)
            self.labels = self.labels.astype(np.int64)
        self.return_index = return_index

        self.video_clip_duration = video_clip_duration
        self.clips_per_video = clips_per_video
        self.mode = mode               

    def _load_sample(self, sample_idx):
        """ it loads a sample video to a container"""
        video_ctr = None
        if self.return_video:
            video_fn = os.path.join(self.video_root, self.video_fns[sample_idx].decode())
            video_ctr = av_wrappers.av_open(video_fn)

        return video_ctr

    def __getitem__(self, index):
        
        ########### just one clip for regular use
        #########################################
        
        if self.mode == 'clip':
            try:
                sample_idx = index % self.num_samples
                video_ctr = self._load_sample(sample_idx)
                v_ss, v_dur = self._sample_snippet(video_ctr)   
                sample = self._get_clip(sample_idx, video_ctr, v_ss, video_clip_duration=v_dur)
                if sample is None:
                    return self[(index+1) % len(self)]

                return sample
            except Exception:
                return self[(index+1) % len(self)]
            
        ########### return clips_per_video number of clips from whole video
        ###################################################################

        elif self.mode == 'video':
            video_ctr = self._load_sample(index)

            # Load entire video
            vs, vf = self._get_time_lims(video_ctr)
            if self.return_video:
                start_time = vs
                final_time = vf
                if final_time <= start_time:
                    final_time = start_time + self.video_clip_duration
                    
            video_dur = final_time - start_time
            sample = self._get_clip(index, video_ctr, start_time, video_clip_duration=video_dur)

            # Split video into overlapping chunks
            chunks = defaultdict(list)

            if self.return_video:
                
                nf = sample['frames'].shape[1]
                chunk_size = int(self.video_clip_duration * self.video_fps)
                if chunk_size >= nf:
                    chunks['frames'] = torch.stack([sample['frames'] for _ in range(self.clips_per_video)])
                else:
                    timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                    chunks['frames'] = torch.stack([sample['frames'][:, ss:ss+chunk_size] for ss in timestamps])
                    
            if self.return_labels:
                chunks['label'] = sample['label']

            if self.return_index:
                chunks['index'] = sample['index']
                
            return chunks
        

    def __len__(self):
        if self.mode == 'clip' or self.mode == 'two_clips':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Subset: {}\n - Num videos: {}\n - Num samples: {}\n".format(
            self.name, self.root, self.subset, self.num_videos, self.num_videos * self.clips_per_video)
        if self.return_video:
            desc += " - Example video: {}/{}\n".format(self.video_root, self.video_fns[0].decode())
        return desc

    def _get_time_lims(self, video_ctr):
        video_st, video_ft = None, None
        if video_ctr is not None:
            video_stream = video_ctr.streams.video[0]
            tbase = video_stream.time_base
            video_st = video_stream.start_time * tbase
            video_dur = video_stream.duration * tbase
            video_ft = video_st + video_dur

        return video_st, video_ft

    def _sample_snippet(self, video_ctr):
        video_st, video_ft = self._get_time_lims(video_ctr)
        video_duration = video_ft - video_st
        if self.video_clip_duration > video_duration:
            return 0., video_duration
        else:
            min_d, max_d = self.video_clip_duration, min(self.video_clip_duration, video_duration)
            duration = random.uniform(min_d, max_d)
            sample_ss_v = random.uniform(video_st, video_ft - duration)
            return sample_ss_v, duration

    def _get_clip(self, clip_idx, video_ctr, video_start_time, video_clip_duration=None):
        if video_clip_duration is None:
            video_clip_duration = self.video_clip_duration

        sample = {}
        if self.return_video:
            frames, fps, start_time = av_wrappers.av_load_video(
                video_ctr,
                video_fps=self.video_fps,
                start_time=video_start_time,
                duration=video_clip_duration,
            )
            if self.video_transform is not None:
                frames = self.video_transform(frames)

            sample['frames'] = frames

        if self.return_labels:
            lbl = self.labels[clip_idx]
            if isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl)
            else:
                sample['label'] = lbl

        if self.return_index:
            sample['index'] = clip_idx

        return sample
