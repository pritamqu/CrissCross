<p align="center"> 
<img src="./docs/assets/images/crisscross_legend.png" width=60% title="CrissCross" alt="CrissCross" /> 
</p>

<h1 style="text-align:center"> 
      Self-Supervised Audio-Visual Representation Learning with Relaxed Cross-Modal Temporal Synchronicity
      <br>
      by 
      <br>
      <a href="https://www.pritamsarkar.com">Pritam Sarkar</a>  and <a href="https://www.alietemad.com">Ali Etemad</a>
</h1>

---
<h3 align="center"> 
<a href="https://arxiv.org/pdf/">Paper</a> - <a href="https://github.com/pritamqu/CrissCross">Repository</a> - <a href="https://pritamqu.github.io/CrissCross/">Project Page</a> - <a href="https://www.pritamsarkar.com">My Home Page</a>
</h3>

---


### Items available
- [x] Paper
- [x] Model weights
- [x] Evaluation codes
- [ ] Training codes (will be released upon acceptance)

### Result
We present the top-1 accuracy averaged over all the splits of each dataset. Please note the results mentioned below are obtained by full-finetuning on UCF101 and HMDB51, and linear classififer (one-vs-all SVM) on ESC50.

| Pretraining Dataset | Pretraining Size | UCF101 | HMDB51 | ESC50 | Model |
| --------  |  --------  |-------------- | ---------- | ----- | -------|  
| Kinetics-Sound | 22K | 88.3% | 60.5% | 82.8% | [visual](../weights/vid_crisscross_kinetics_sound.pth.tar); [audio](../weights/aud_crisscross_kinetics_sound.pth.tar)
| Kinetics400 | 240K | 91.5% | 64.7% | 86.8% | [visual](../weights/vid_crisscross_kinetics400.pth.tar); [audio](../weights/aud_crisscross_kinetics400.pth.tar)
| AudioSet | 1.8M | 92.4% | 66.8% | 90.5% | [visual](../weights/vid_crisscross_audioset.pth.tar); [audio](../weights/aud_crisscross_audioset.pth.tar)


### Environment Setup
List of dependencies can be found [here](./docs/assets/files/requirements.txt). You can create an environment as `conda create --name crisscross --file requirements.txt`

### Datasets
Please make sure to keep the datasets in their respective directory, and change the path in `/tools/paths`. The sources of all the public datasets used in this study are mentioned here.
- AudioSet: Please check this [repository](https://github.com/speedyseal/audiosetdl) to download AudioSet.
- Kinetics400: You can either use a crawler (similar to one available for AudioSet) to download the Kinetics400, or simply download from amazon aws, prepared by [CVD Foundation](https://github.com/cvdfoundation/kinetics-dataset).
- UCF101: [Website to download.](https://www.crcv.ucf.edu/data/UCF101.php)
- HMDB51: [Website to download.](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- ESC50: [Website to download.](https://github.com/karolpiczak/ESC-50).

### Self-supervised Training

Here are few examples, how to train CrissCross in diffierent GPU setup. 
A batch size of 2048 can be trained on 8 RTX6000 or 8 V100 or similar GPUs.
To know more about Pytorch distributed training, please see [Pytorch official documentation](https://pytorch.org/tutorials/beginner/dist_overview.html).

#### Single GPU

```python
cd train
python main_pretext_audiovisual.py \
            --world-size 1 --rank 0 \
            --quiet --sub_dir 'pretext' \
            --config-file 'audvid_crisscross' \
            --db 'kinetics400'
```

#### Single Node Multiple GPU

```python
# MASTER="127.0.0.1" or HOSTNAME
# MPORT="8888" OR ANY FREE PORT
cd train
python main_pretext_audiovisual.py \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size 1 --rank 0 \
            --quiet --sub_dir 'pretext' \
            --config-file 'audvid_crisscross' \
            --db 'kinetics400'
```

#### Multiple Node Multiple GPU

```python
# MASTER="127.0.0.1" or HOSTNAME
# MPORT="8888" OR ANY FREE PORT

cd train
# Node 0:
python main_pretext_audiovisual.py \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size 2 --rank 0 \
            --quiet --sub_dir 'pretext' \
            --config-file 'audvid_crisscross' \
            --db 'kinetics400'
# Node 1:
python main_pretext_audiovisual.py \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size 2 --rank 1 \
            --quiet --sub_dir 'pretext' \
            --config-file 'audvid_crisscross' \
            --db 'kinetics400'
```

### Downstream Evaluation
You can directly use the given weights to evaluate on the following benchmarks, using the commands given below. Please make sure to save the model weights to the location at `/path/to/model`. Downstream evaluation is performed on a single GPU, Nvidia RTX 6000.

**UCF101**
```python
# 8 frame evaluation
cd evaluate
python evaluate/eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'ucf101' --config-file full_ft_8f_fold1 --pretext_model /path/to/model
# 32 frame evaluation
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'ucf101' --config-file full_ft_32f_fold1 --pretext_model /path/to/model
```
**HMDB51**
```python
# 8 frame evaluation
cd evaluate
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'hmdb51' --config-file full_ft_8f_fold1 --pretext_model /path/to/model
# 32 frame evaluation
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'hmdb51' --config-file full_ft_32f_fold1 --pretext_model /path/to/model
```
**ESC50**
```python
# linear evaluation using SVM
cd evaluate
python eval_audio.py --world-size 1 --rank 0 --gpu 0 --db 'esc50' --config-file config_fold1_2s --pretext_model /path/to/model
```

### Citation
Please cite our paper using the given bibtex entry.
```
@misc{sarkar2021crisscross,
      title={Self-Supervised Audio-Visual Representation Learning with Relaxed Cross-Modal Temporal Synchronicity}, 
      author={Pritam Sarkar and Ali Etemad},
      year={2021},
      eprint={2111.00000},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Acknowledgments
We are grateful to **Bank of Montreal** and **Mitacs** for funding this research. We are also thankful to **[Vector Institute](https://vectorinstitute.ai/)** and **SciNet HPC Consortium** for helping with the computation resources.

### Question
- You may directly contact me at <pritam.sarkar@queensu.ca> or connect with me on [LinkedIN](https://www.linkedin.com/in/sarkarpritam/).
