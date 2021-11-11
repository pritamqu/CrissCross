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


class AudioDataset(data.Dataset):
    def __init__(self,
                 return_audio=True,
                 audio_root=None,
                 audio_fns=None,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=None,
                 audio_transform=None,
                 return_labels=False,
                 labels=None,
                 return_index=False,
                 mode='clip',
                 clips_per_video=1,
                 ):
        super(AudioDataset, self).__init__()

        self.num_samples = 0

        self.return_audio = return_audio
        self.audio_root = audio_root
        if return_audio:
            self.audio_fns = chararray(audio_fns)
            self.num_samples = self.audio_fns.shape[0]
        self.audio_fps = audio_fps
        self.audio_fps_out = audio_fps_out
        self.audio_transform = audio_transform

        self.return_labels = return_labels
        if return_labels:
            self.labels = np.array(labels)
            self.labels = self.labels.astype(np.int64)
        self.return_index = return_index

        self.audio_clip_duration = audio_clip_duration
        self.clips_per_video = clips_per_video
        self.mode = mode
  
    def _load_sample(self, sample_idx):
        """ it loads a sample audio to a container"""
        audio_ctr = None
        if self.return_audio:
            audio_fn = os.path.join(self.audio_root, self.audio_fns[sample_idx].decode())
            audio_ctr = av_wrappers.av_open(audio_fn)

        return audio_ctr

    def __getitem__(self, index):
        
        ########### just one clip for regular use
        #########################################
        
        if self.mode == 'clip':
            try:
                sample_idx = index % self.num_samples
                audio_ctr = self._load_sample(sample_idx)
                a_ss, a_dur = self._sample_snippet(audio_ctr)   
                sample = self._get_clip(sample_idx, audio_ctr, a_ss, audio_clip_duration=a_dur)
                if sample is None:
                    return self[(index+1) % len(self)]

                return sample
            except Exception:
                return self[(index+1) % len(self)]
            
        ########### return clips_per_video number of clips from whole video
        ###################################################################

        elif self.mode == 'video':
            audio_ctr = self._load_sample(index)

            # Load entire video
            ss, sf = self._get_time_lims(audio_ctr)
            if self.return_audio:
                start_time = ss
                final_time = sf
                if final_time <= start_time:
                    final_time = start_time + self.audio_clip_duration
                    
            audio_dur = final_time - start_time
            sample = self._get_clip(index, audio_ctr, start_time, audio_clip_duration=audio_dur)

            # Split video into overlapping chunks
            chunks = defaultdict(list)

            if self.return_audio:
                
                nf = sample['audio'].shape[-1] # time dim [1, freq, time]
                chunk_size = int(self.audio_clip_duration * self.audio_fps_out)
                if chunk_size >= nf:
                    chunks['audio'] = torch.stack([sample['audio'] for _ in range(self.clips_per_video)])
                else:
                    timestamps = np.linspace(0, max(nf - chunk_size, 1), self.clips_per_video).astype(int)
                    chunks['audio'] = torch.stack([sample['audio'][:, :, int(ss):int(ss+chunk_size)] for ss in timestamps])
                    
            if self.return_labels:
                chunks['label'] = sample['label']

            if self.return_index:
                # ts = torch.from_numpy(np.linspace(start_time, final_time-self.audio_clip_duration, self.clips_per_video))
                # chunks['index'] = torch.stack([torch.tensor(sample['index']).repeat(self.clips_per_video), ts.float()], dim=1)
                chunks['index'] = sample['index']
                
            return chunks
        

    def __len__(self):
        if self.mode == 'clip' or self.mode == 'two_clips':
            return self.num_samples * self.clips_per_video
        else:
            return self.num_samples

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Subset: {}\n - Num videos: {}\n - Num samples: {}\n".format(
            self.name, self.root, self.subset, self.num_audios, self.num_audios * self.clips_per_video)
        if self.return_audio:
            desc += " - Example audio: {}/{}\n".format(self.audio_root, self.audio_fns[0].decode())
        return desc

    def _get_time_lims(self, audio_ctr):
        audio_st, audio_ft = None, None
        if audio_ctr is not None:
            audio_stream = audio_ctr.streams.audio[0]
            tbase = audio_stream.time_base
            ## TODO: recheck this
            # audio_stream is av.audio.codeccontext.AudioCodecContext object, does not have start time
            # audio_st = audio_stream.start_time * tbase
            audio_st = 0
            audio_dur = audio_stream.duration * tbase
            audio_ft = audio_st + audio_dur

        return audio_st, audio_ft

    def _sample_snippet(self, audio_ctr):
        audio_st, audio_ft = self._get_time_lims(audio_ctr)
        audio_duration = audio_ft - audio_st
        if self.audio_clip_duration > audio_duration:
            return 0., audio_duration
        else:
            min_d, max_d = self.audio_clip_duration, min(self.audio_clip_duration, audio_duration)
            duration = random.uniform(min_d, max_d)
            sample_ss_a = random.uniform(audio_st, audio_ft - duration)
            return sample_ss_a, duration

    def _get_clip(self, clip_idx, audio_ctr, audio_start_time, audio_clip_duration=None):
        if audio_clip_duration is None:
            audio_clip_duration = self.audio_clip_duration

        sample = {}
        if self.return_audio:
            samples, rate = av_wrappers.av_laod_audio(
                audio_ctr,
                audio_fps=self.audio_fps,
                start_time=audio_start_time,
                duration=audio_clip_duration,
            )
            if self.audio_transform is not None:
                samples = self.audio_transform(samples)
            sample['audio'] = samples

        if self.return_labels:
            lbl = self.labels[clip_idx]
            if isinstance(lbl, np.ndarray):
                sample['label'] = torch.from_numpy(lbl)
            else:
                sample['label'] = lbl

        if self.return_index:
            sample['index'] = clip_idx

        return sample
