import av
import numpy as np
import math
from fractions import Fraction
from scipy.interpolate import interp1d

av.logging.set_level(0)

def av_open(inpt):
    try:
        container = av.open(inpt)
    except:
        container = av.open(inpt, metadata_errors="ignore")
    return container

def adjust_framerate(frames, target_sz):
    
    
    def _resample(frames, current_sz, target_sz, mode='nearest'):
        # do not use linear, use nearest
        a = np.arange(0, len(frames), 1)
        xp = np.arange(0, len(a), current_sz/target_sz)
        if mode == 'nearest':
            nearest = interp1d(np.arange(len(a)), a, kind='nearest', fill_value="extrapolate")
            idx = nearest(xp).astype(int)
        elif mode == 'linear':
            lin = interp1d(np.arange(len(a)), a, kind='linear', fill_value="extrapolate")
            idx = lin(xp).astype(int)
        else:
            raise ValueError(f'{mode} is not a valid mode.')
            
        return [frames[i] for i in idx]        
    
    current_sz = len(frames)
    return _resample(frames, current_sz, target_sz)

def secs_to_pts(time_in_seconds: float, time_base: float, start_pts: float) -> float:
    """
    Converts a time (in seconds) to the given time base and start_pts offset
    presentation time.
    Returns:
        pts (float): The time in the given time base.
    """
    if time_in_seconds == math.inf:
        return math.inf

    time_base = float(time_base)
    return int(time_in_seconds / time_base) + start_pts

def pts_to_secs(time_in_pts: float, time_base: float, start_pts: float) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.
    Returns:
        time_in_seconds (float): The corresponding time in seconds.
    """
    if time_in_pts == math.inf:
        return math.inf

    return (time_in_pts - start_pts) * float(time_base)

        
def pyav_decode_stream(container, video_fps=None, start_secs=0, duration=None, target_sz=8):
    stream = container.streams.video[0]
    _video_time_base = stream.time_base
    
    _video_start_pts = stream.start_time # whole video
    _video_duration = stream.duration # whole video
    _video_end_pts = _video_start_pts+_video_duration
    _fps = stream.average_rate
    
    start_pts = secs_to_pts(start_secs, _video_time_base, _video_start_pts) # the small clip of our interest
    end_secs = start_secs+duration
    end_pts = secs_to_pts(end_secs, _video_time_base, _video_start_pts) # the small clip of our interest
        
    if video_fps is None:
        video_fps = _fps

    if duration is None:
        duration = pts_to_secs(_video_duration, _video_time_base, _video_start_pts)

    # Seeking in the stream is imprecise. Thus, seek to an earlier pts by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)
    container.seek(int(seek_offset), any_frame=False, backward=True, stream=stream)
    holder = {}
    max_pts = 0
    for frame in container.decode(video=0):
        max_pts = max(max_pts, frame.pts)
        if frame.pts >= start_pts and frame.pts <= end_pts:
            holder[frame.pts] = frame.to_image()
        elif frame.pts > end_pts:
            break

    frames = [holder[pts] for pts in sorted(holder)]
    start_time = pts_to_secs(list(holder.keys())[0], _video_time_base, _video_start_pts)
    
    # those frames are based on original video frame rate (30) of duration (0.5 second)
    # either interpolate or drop based on _fps and video_fps [like resize operation] 1./video_fps

    frames = adjust_framerate(frames, target_sz=target_sz) # it takes care of lowe frame rate issue, higher frame rate, lower duration video

    return frames, video_fps, start_time

def av_load_video(container, video_fps=None, start_time=0, duration=None):
    video_stream = container.streams.video[0]
    _ss = video_stream.start_time * video_stream.time_base
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur
    _fps = video_stream.average_rate

    if video_fps is None:
        video_fps = _fps

    if duration is None:
        duration = _ff - start_time

    # Figure out which frames to decode
    outp_times = [t for t in np.arange(start_time, min(start_time + duration - 0.5/_fps, _ff), 1./video_fps)][:int(duration*video_fps)]
    outp_vframes = [int((t - _ss) * _fps) for t in outp_times]
    start_time = outp_vframes[0] / float(_fps)

    # Fast forward
    container.seek(int(start_time * av.time_base))

    # Decode snippet
    frames = []
    for frame in container.decode(video=0):
        if len(frames) == len(outp_vframes):
            break   # All frames have been decoded
        frame_no = frame.pts * frame.time_base * _fps
        if frame_no < outp_vframes[len(frames)]:
            continue    # Not the frame we want

        # Decode
        pil_img = frame.to_image()
        while frame_no >= outp_vframes[len(frames)]:
            frames += [pil_img]
            if len(frames) == len(outp_vframes):
                break   # All frames have been decoded

    return frames, video_fps, start_time

def av_laod_audio(container, audio_fps=None, start_time=0, duration=None):
    audio_stream = container.streams.audio[0]
    _ss = audio_stream.start_time * audio_stream.time_base if audio_stream.start_time is not None else 0.
    _dur = audio_stream.duration * audio_stream.time_base
    _ff = _ss + _dur
    _fps = audio_stream.rate

    if audio_fps is None:
        resample = False
        audio_fps = _fps
    else:
        resample = True
        audio_resampler = av.audio.resampler.AudioResampler(format="s16p", layout="mono", rate=audio_fps)

    if duration is None:
        duration = _ff - start_time
    duration = min(duration, _ff - start_time)
    end_time = start_time + duration

    # Fast forward
    container.seek(int(start_time * av.time_base))

    # Decode snippet
    data, timestamps = [], []
    for frame in container.decode(audio=0):
        frame_pts = frame.pts * frame.time_base
        frame_end_pts = frame_pts + Fraction(frame.samples, frame.rate)
        if frame_end_pts < start_time:   # Skip until start time
            continue
        if frame_pts > end_time:       # Exit if clip has been extracted
            break

        try:
            frame.pts = None
            if resample:
                np_snd = audio_resampler.resample(frame).to_ndarray()
            else:
                np_snd = frame.to_ndarray()
            data += [np_snd]
            timestamps += [frame_pts]
        except AttributeError:
            break
    data = np.concatenate(data, 1)

    # Trim audio
    start_decoded_time = timestamps[0]
    ss = int((start_time - start_decoded_time) * audio_fps)
    t = int(duration * audio_fps)
    if ss < 0:
        data = np.pad(data, ((0, 0), (-ss, 0)), 'constant', constant_values=0)
        ss = 0
    if t > data.shape[1]:
        data = np.pad(data, ((0, 0), (0, t-data.shape[1])), 'constant', constant_values=0)
    data = data[:, ss: ss+t]
    data = data / np.iinfo(data.dtype).max

    return data, audio_fps


