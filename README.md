# Replicating of paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) with Pytorch

This repository contains the code and necessary files to replicate the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".](https://arxiv.org/pdf/2010.11929.pdf)

The paper proposes a new architecture for image recognition, called Vision Transformer (ViT), which utilizes transformers, a type of neural network architecture originally introduced for natural language processing.

![image](https://user-images.githubusercontent.com/91833187/223550950-20eca9ea-526a-4bb0-a9b9-758cd05df4a8.png "Vision Transformor (ViT)")

## Getting Started
1. Create virtual environment and activate it, for convenience you can use conda:
```
conda create --name <name_of_enb> python=3.10.9
conda activate <name_of_env>
```
2. Clone the repository using the following command:
```
git clone https://github.com/seblful/vision-transformer.git
```
3. Install the required packages using pip:
```
pip install -r requirements.txt
```

## Download data
To download data you need to run data_download.py.

Example of usage:
```
python data_download.py --kaggle_username <your_username> --kaggle_api <your api>
```

## Train model
To train a model you need to run train.py.

Example of usage:
```
python train.py --epochs 10 --batch_size 32 --lr 0.001 --train_size 0.85 --test_size 0.15 --data_folder data
```

## References
* Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
