# An implementation of CVPR 2020 paper "SCATTER: Selective Context Attentional Scene Text Recognizer"

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Litman_SCATTER_Selective_Context_Attentional_Scene_Text_Recognizer_CVPR_2020_paper.pdf)

## Introduction
This is an unofficial implementation of paper "SCATTER: Selective Context Attentional Scene Text Recognizer" published at CVPR 2020. 

## Getting Started

### Dependency
- This work was tested with PyTorch 1.6.0, CUDA 10.2, python 3.6.10 and Ubuntu 18.04.
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
- requirements : lmdb, pillow, nltk, natsort
```
pip3 install lmdb pillow nltk natsort
```

### Dataset
- training dataset: [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1], [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2] and 
[SynthAdd (SA)](https://drive.google.com/drive/folders/1agZ9ufDNYfzdQe1fGWH3dSk6L5BQU0o0?usp=sharing) [3]
- validation datasets : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2)[4], [IC15](http://rrc.cvc.uab.es/?ch=4)[5], [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[6], and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[7].\
evaluation datasets : benchmark evaluation datasets, consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[7], [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[8], [IC13](http://rrc.cvc.uab.es/?ch=2)[4], [IC15](http://rrc.cvc.uab.es/?ch=4)[5], [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)[9], and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)[10].

###Pretrained Model

Two pretrained models are provided (Will be updated when better models are trained):
1. non-senstive: includes ten digits (0-9) and 26 characters (a-z).
2. sensitive: includes all readable digits.
   
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1niuPM6otpSQFSai8Ft2bO0lhdqEjE96Z?usp=sharing)

###Run demo
- With non-sensitve model
```
python demo.py --saved_model scatter-case-non-sensitive.pth --sensitive --image_folder <path_to_image_folder>
```

- With sensitve model
```
python demo.py --saved_model scatter-case-sensitive.pth --image_folder <path_to_image_folder>
```

### Training and evaluation

Training 
```
python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST-SA --batch_ratio 0.4-0.4-0.2 --sensitive 
```

Testing

```
python3 test.py --eval_data data_lmdb_release/evaluation --saved_model scatter-case-sensitive.pth --sensitive --data_filtering_off
```

###Comparison

## Acknowledgements
This code is built upon [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). 

## Reference
[1] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman. Synthetic data and artificial neural networks for natural scenetext  recognition. In Workshop on Deep Learning, NIPS, 2014. <br>
[2] A. Gupta, A. Vedaldi, and A. Zisserman. Synthetic data fortext localisation in natural images. In CVPR, 2016. <br>
[3] Hui Li, Peng Wang, Chunhua Shen, Guyu Zhang. Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition. In AAAI, 2019 <br>
[4] D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i Big-orda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazan, andL. P. De Las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013. <br>
[5] D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R.Chandrasekhar, S. Lu, et al. ICDAR 2015 competition on ro-bust reading. In ICDAR, pages 1156–1160, 2015. <br>
[6] A. Mishra, K. Alahari, and C. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012. <br>
[7] K. Wang, B. Babenko, and S. Belongie. End-to-end scenetext recognition. In ICCV, pages 1457–1464, 2011. <br>
[8] S. M. Lucas, A. Panaretos, L. Sosa, A. Tang, S. Wong, andR. Young. ICDAR 2003 robust reading competitions. In ICDAR, pages 682–687, 2003. <br>
[9] T. Q. Phan, P. Shivakumara, S. Tian, and C. L. Tan. Recognizing text with perspective distortion in natural scenes. In ICCV, pages 569–576, 2013. <br>
[10] A. Risnumawan, P. Shivakumara, C. S. Chan, and C. L. Tan. A robust arbitrary text detection system for natural scene images. In ESWA, volume 41, pages 8027–8048, 2014. <br>
[11] B. Shi, X. Bai, and C. Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. In TPAMI, volume 39, pages2298–2304. 2017. 

## Citation
Please consider citing this work in your publications if it helps your research.
```
@inproceedings{litman2020scatter,
  title={SCATTER: selective context attentional scene text recognizer},
  author={Litman, Ron and Anschel, Oron and Tsiper, Shahar and Litman, Roee and Mazor, Shai and Manmatha, R},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11962--11972},
  year={2020}
}
```
