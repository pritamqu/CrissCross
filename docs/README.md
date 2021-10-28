<p align="center"> 
<img src="./assets/images/crisscross_legend.png" width=60% title="CrissCross" alt="CrissCross" /> 
</p>

<h3 align="center"> 
<a href="https://arxiv.org/pdf/">Paper</a> - <a href="https://github.com/pritamqu/crisscross">Repository</a> - <a href="https://pritamqu.github.io/crisscross/">Project Page</a> - <a href="https://www.pritamsarkar.com">My Home Page</a>
</h3>

<h1 style="text-align:center"> 
      Self-Supervised Audio-Visual Representation Learning with Relaxed Temporal Synchronicity 
      <br>
      by 
      <br>
      <a href="https://www.pritamsarkar.com">Pritam Sarkar</a>  and <a href="https://www.alietemad.com">Ali Etemad</a>
</h1>

<!-- # Self-Supervised Audio-Visual Representation Learning with Relaxed Temporal Synchronicity - by Pritam Sarkar and Ali Etemad -->
<!-- ### by Pritam Sarkar and Ali Etemad -->
<!-- ## by [Pritam Sarkar](https://www.pritamsarkar.com) and [Ali Etemad](https://www.alietemad.com) -->

This is the official page of our project **CrissCross**. Please note all the implementations are done using PyTorch. 

### Items available
- [x] Paper
- [x] Model weights
- [x] Evaluation codes
- [ ] Training codes (will be released upon acceptance)
<!-- - [ ] Additional findings -->

### Result
| Dataset   | Pretraining DB | Top-1 Acc. | Model | Config |
| --------  | -------------- | ---------- | ----- | -------|  
| UCF101    | Kinetics-Sound | 88.3% | [url](../weights/vid_crisscross_kinetics_sound.pth.tar) | [config path](../crisscross/evaluate/configs/ucf101/)
| HMDB51    | Kinetics-Sound | 60.5% | [url](../weights/vid_crisscross_kinetics_sound.pth.tar) | [config path](../crisscross/evaluate/configs/hmdb51/)
| ESC50     | Kinetics-Sound | 82.8% | [url](../weights/aud_crisscross_kinetics_sound.pth.tar) | [config path](../crisscross/evaluate/configs/esc50/)
| UCF101    | Kinetics400 | 91.5% | [url](../weights/vid_crisscross_kinetics400.pth.tar) | [config path](../crisscross/evaluate/configs/ucf101/)
| HMDB51    | Kinetics400 | 64.7% | [url](../weights/vid_crisscross_kinetics400.pth.tar) | [config path](../crisscross/evaluate/configs/hmdb51/)
| ESC50     | Kinetics400 | 86.8% | [url](../weights/aud_crisscross_kinetics400.pth.tar) | [config path](../crisscross/evaluate/configs/esc50/)
| UCF101    | AudioSet | 99.9% | [url](../weights/vid_crisscross_audioset.pth.tar) | [config path](crisscross/configs/config.yaml)
| HMDB51    | AudioSet | 99.9% | [url](../weights/vid_crisscross_audioset.pth.tar) | [config path](crisscross/configs/config.yaml)
| ESC50     | AudioSet | 99.9% | [url](../weights/aud_crisscross_audioset.pth.tar) | [config path](crisscross/configs/config.yaml)

### Environment Setup
List of dependencies can be found [here](./assets/files/requirements.txt). You can create an environment as `conda create --name crisscross --file requirements.txt`

### Datasets
Please make sure to keep the datasets in the correct directory, and also change the path in `/tools/paths`. I briefly mentioned the sources of all the public datasets used in this study.
- AudioSet: Please check this [repository](https://github.com/speedyseal/audiosetdl) to download AudioSet.
- Kinetics400: You can either use a crawler (similar to one available for AudioSet) to download Kinetics400, or simply download from amazon aws, prepared by [CVD Foundation](https://github.com/cvdfoundation/kinetics-dataset).
- UCF101: [Website to download.](https://www.crcv.ucf.edu/data/UCF101.php)
- HMDB51: [Website to download.](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- ESC50: [Website to download.](https://github.com/karolpiczak/ESC-50).

<!-- ### Self-supervised Training

Here is an example, how to train CrissCross in multiple nodes. To know more about Pytorch distributed training, please see [Pytorch official documentation](https://pytorch.org/tutorials/beginner/dist_overview.html).

```python
# MASTER="127.0.0.1" or HOSTNAME
# MPORT="8888" OR ANY FREE PORT

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
``` -->

### Downstream Evaluation
You can directly use the given weights to evaluate on the following benchmarks, using the commands given below. Please make sure to save the model weights to the location at `/path/to/model`. Downstream evaluation was performed on a single GPU, Nvidia RTX 6000.

**UCF101**
```python
# 8 frame evaluation
python main_finetune.py --world-size 1 --rank 0 --gpu 0 --db 'ucf101' --config-file full_ft_8f_fold1 --pretext_model /path/to/model
# 32 frame evaluation
python main_finetune.py --world-size 1 --rank 0 --gpu 0 --db 'ucf101' --config-file full_ft_32f_fold1 --pretext_model /path/to/model
```
**HMDB51**
```python
# 8 frame evaluation
python main_finetune.py --world-size 1 --rank 0 --gpu 0 --db 'hmdb51' --config-file full_ft_8f_fold1 --pretext_model /path/to/model
# 32 frame evaluation
python main_finetune.py --world-size 1 --rank 0 --gpu 0 --db 'hmdb51' --config-file full_ft_32f_fold1 --pretext_model /path/to/model
```
**ESC50**
```python
# linear evaluation using SVM
python main_svm.py --world-size 1 --rank 0 --gpu 0 --db 'esc50' --config-file config_fold1_2s --pretext_model /path/to/model
```

### Citation
Please cite our paper using the given bibtex entry.
```
@misc{sarkar2021crisscross,
      title={paper title}, 
      author={Pritam Sarkar and Ali Etemad},
      year={2021},
      eprint={2010.00104},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Acknowledgments
We are grateful to **Bank of Montreal** and **Mitacs** for funding this research. We are also thankful to **[Vector Institute](https://vectorinstitute.ai/)** and **SciNet HPC Consortium** for helping with the computation resources.

### Question
- You may directly contact me at <pritam.sarkar@queensu.ca> or connect with me on [LinkedIN](https://www.linkedin.com/in/sarkarpritam/).
