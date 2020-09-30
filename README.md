# An implementation of CVPR 2020 paper "SCATTER: Selective Context Attentional Scene Text Recognizer"

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Litman_SCATTER_Selective_Context_Attentional_Scene_Text_Recognizer_CVPR_2020_paper.pdf)

Two pretrained models are provided:
1. non-senstive: includes ten digits (0-9) and 26 characters (a-z).
2. sensitive: includes all readable digits.
   
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1niuPM6otpSQFSai8Ft2bO0lhdqEjE96Z?usp=sharing)

Run demo: python demo.py --saved_model <path_to_saved_model> --sensitive --image_folder <path_to_image_folder>

This code is built upon [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). 
All requirements for running code, please consider the installation from [this](https://github.com/clovaai/deep-text-recognition-benchmark) framework.

(Will be updated..)
