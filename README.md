<!-- <p align="center">  -->
<!-- <img src="./docs/assets/images/crisscross_legend_white_bg.jpg" width=50% title="CrissCross" alt="CrissCross" />  -->
<!-- </p> -->

<h1 align="center"> 
Self-Supervised Audio-Visual Representation Learning with Relaxed Cross-Modal Synchronicity
</h1>

<h3 align="center">
AAAI 2023
</h3>
<h3 align="center">
<a href="https://www.pritamsarkar.com">Pritam Sarkar</a>
&nbsp;
Ali Etemad
</h3>
<h3 align="center"> 
<a href="https://arxiv.org/pdf/2111.05329.pdf">[Paper]</a> <!-- change with aaai link -->
<a href="./assets/files/crisscross_supp.pdf">[Appendix]</a>  <a href="https://arxiv.org/pdf/2111.05329.pdf"> [ArXiv]</a>  <a href="https://github.com/pritamqu/CrissCross/"> [Code]</a> <a href="https://pritamqu.github.io/CrissCross/"> [Website]</a>
</h3>

We present **CrissCross**, a self-supervised framework for learning audio-visual representations. A novel notion is introduced in our framework whereby in addition to learning the intra-modal and standard *synchronous* cross-modal relations, CrissCross also learns *asynchronous* cross-modal relationships. We perform in-depth studies showing that by relaxing the temporal synchronicity between the audio and visual modalities, the network learns strong generalized representations useful for a variety of downstream tasks. To pretrain our proposed solution, we use 3 different datasets with varying sizes, Kinetics-Sound, Kinetics400, and AudioSet. The learned representations are evaluated on a number of downstream tasks namely action recognition, sound classification, and action retrieval. Our experiments show that CrissCross either outperforms or achieves performances on par with the current state-of-the-art self-supervised methods on action recognition and action retrieval with UCF101 and HMDB51, as well as sound classification with ESC50 and DCASE. Moreover, CrissCross outperforms fully-supervised pretraining while pretrained on Kinetics-Sound. 


### Updates
- [x] Paper
- [x] Pretrained model weights <!-- [Pretrained model weights](https://github.com/pritamqu/CrissCross/releases/tag/model_weights) -->
- [x] Evaluation codes
- [x] Training codes

### Result
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-audio-visual-representation/audio-classification-on-dcase)](https://paperswithcode.com/sota/audio-classification-on-dcase?p=self-supervised-audio-visual-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-audio-visual-representation/self-supervised-audio-classification-on-esc)](https://paperswithcode.com/sota/self-supervised-audio-classification-on-esc?p=self-supervised-audio-visual-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-audio-visual-representation/self-supervised-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-hmdb51?p=self-supervised-audio-visual-representation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-audio-visual-representation/self-supervised-action-recognition-on-ucf101)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-ucf101?p=self-supervised-audio-visual-representation) -->

We present the top-1 accuracy averaged over all the splits of each dataset. Please note that the results mentioned below are obtained by full-finetuning on UCF101 and HMDB51, and linear classifier on ESC50 and DCASE. 

| Pretraining Dataset | Pretraining Size | UCF101 | HMDB51 | ESC50 | DCASE | Model |
| --------  |  --------  |-------------- | ---------- | ----- | -------| -------| 
| Kinetics-Sound | 22K | 88.3% | 60.5% | 82.8% | 93.0% | [visual](https://github.com/pritamqu/CrissCross/releases/download/model_weights/vid_crisscross_kinetics_sound.pth.tar.zip); [audio](https://github.com/pritamqu/CrissCross/releases/download/model_weights/aud_crisscross_kinetics_sound.pth.tar.zip)
| Kinetics400 | 240K | 91.5% | 64.7% | 86.8% | 96.0% | [visual](https://github.com/pritamqu/CrissCross/releases/download/model_weights/vid_crisscross_kinetics_400.pth.tar.zip); [audio](https://github.com/pritamqu/CrissCross/releases/download/model_weights/aud_crisscross_kinetics_400.pth.tar.zip)
| AudioSet | 1.8M | 92.4% | 67.4% | 90.5% | 97.0% | [visual](https://github.com/pritamqu/CrissCross/releases/download/model_weights/vid_crisscross_audioset.pth.tar.zip); [audio](https://github.com/pritamqu/CrissCross/releases/download/model_weights/aud_crisscross_audioset.pth.tar.zip)


### Environment Setup
List of dependencies can be found [here](./docs/assets/files/requirements.txt). You can create an environment as `conda create --name crisscross --file requirements.txt`

### Datasets
Please make sure to keep the datasets in their respective directories, and change the path in `/tools/paths` accordingly. The sources of all the public datasets used in this study are mentioned here.
- AudioSet: Please check this [repository](https://github.com/speedyseal/audiosetdl) to download AudioSet.
- Kinetics400: You can either use a crawler (similar to the one available for AudioSet) to download the Kinetics400, or simply download from the Amazon AWS, prepared by [CVD Foundation](https://github.com/cvdfoundation/kinetics-dataset).
- UCF101: [Website to download.](https://www.crcv.ucf.edu/data/UCF101.php)
- HMDB51: [Website to download.](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- ESC50: [Website to download.](https://github.com/karolpiczak/ESC-50)
- DCASE: [Website to download.](https://dcase.community/challenge2013/download#audio-dataset)

### Self-supervised Training

Here are a few examples on how to train CrissCross in diffierent GPU setups. 
A batch size of 2048 can be used to train on 8X RTX6000 or 8X V100 or similar GPUs. 
To know more about PyTorch distributed training, please see [Pytorch official documentation](https://pytorch.org/tutorials/beginner/dist_overview.html).

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
You can directly use the given weights to evaluate the model on the following benchmarks, using the commands given below. Please make sure to save the model weights to the following location: `/path/to/model`. Downstream evaluation is performed on a single Nvidia RTX 6000 GPU. Note, codes are tested on a linux machine.

**UCF101**
```python
# full-finetuning
cd evaluate
# 8 frame evaluation
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'ucf101' --config-file kinetics400/full_ft_8f_fold1 --pretext_model /path/to/model
# 32 frame evaluation
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'ucf101' --config-file kinetics400/full_ft_32f_fold1 --pretext_model /path/to/model
```
**HMDB51**
```python
# full-finetuning
cd evaluate
# 8 frame evaluation
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'hmdb51' --config-file kinetics400/full_ft_8f_fold1 --pretext_model /path/to/model
# 32 frame evaluation
python eval_video.py --world-size 1 --rank 0 --gpu 0 --db 'hmdb51' --config-file kinetics400/full_ft_32f_fold1 --pretext_model /path/to/model
```
**ESC50**
```python
# linear evaluation using SVM
cd evaluate
# 2-second evaluation
python eval_audio.py --world-size 1 --rank 0 --gpu 0 --db 'esc50' --config-file config_fold1_2s --pretext_model /path/to/model
# 5-second evaluation
python eval_audio.py --world-size 1 --rank 0 --gpu 0 --db 'esc50' --config-file config_fold1_5s --pretext_model /path/to/model
```
**DCASE**
```python
# linear evaluation using fc tuning
cd evaluate
# 2-second evaluation
python eval_audio.py --world-size 1 --rank 0 --gpu 0 --db 'dcase' --config-file config_2s --pretext_model /path/to/model
```

### Citation
If you find this repository useful, please consider giving a star :star: and citation using the given BibTeX entry:

```
@misc{sarkar2021crisscross,
      title={Self-Supervised Audio-Visual Representation Learning with Relaxed Cross-Modal Synchronicity}, 
      author={Pritam Sarkar and Ali Etemad},
      year={2021},
      eprint={2111.05329},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgments
We are grateful to **Bank of Montreal** and **Mitacs** for funding this research. We are also thankful to **SciNet HPC Consortium** for helping with the computation resources.

### Question
You may directly contact me at <pritam.sarkar@queensu.ca> or connect with me on [LinkedIn](https://www.linkedin.com/in/sarkarpritam/).
