# RAMER
<h2 align="center">
Leveraging Retrieval Augment Approach for Multimodal Emotion Recognition Under Missing Modalities
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
</p>


### Overview
<div style="text-align: center">
<img src="resources/framework.pdf" width = "100%"/>
</div>



### Key Implementations

- Don't add Multimodal interaction in the Pretrain Model ;
- First pretrain the Unimodal Basemodel in `models/pretrain_model.py`.
- Inferencing the whole dataset and saving the hidden state in `models/pretrain_model.py line 86/99/111  cls_output_A/V/L`.
- FAISS index creating `models/retrieval_augmentor.py line 192`;
- Retrieval approach `models/retrieval_augmentor.py line 141 get_most_similar_vectors`;
- Utilizing retrieval approach `models/retrieval_model.py line 105, 215`.



### Installations and Usage

Create a conda environment with Pytorch

```
conda create --name contrastive python=3.9
conda activate contrastive

pip install torch torchvision torchaudio numpy pandas sklearn scipy tqdm pickle omegaconf
```
Then you need to 
```
git clone https://github.com/zeroQiaoba/MERTools.git
```
Finally, put our model into ``MER2024/toolkit/models``.


This repository is constructed and gives the main modules used in our work, which are based on the codebase from [MER2024](https://github.com/zeroQiaoba/MERTools/tree/master/MER2024). 

You can get more information about the training framework or the competition from the link above.

Other requirements can also refer to the MER2024 GitHub repository.



### Datasets Preparation
#### MER2024 Dataset

Please download the End User License Agreement, fill it out, and send it to merchallenge.contact@gmail.com to access the data. The EULA file can be found at [MER2024](https://github.com/zeroQiaoba/MERTools/tree/master/MER2024). 

MER2024 Baseline also provided the code for feature extracting, including utterance-level and the frame-level.


### Acknowledgment
Thanks to the MER Challenge 2024 Committee for their support and resources.

