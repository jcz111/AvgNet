# AvgNet: Adaptive Visibility Graph Neural Network and It’s Application in Modulation Classification
Qi Xuan, Senior Member, IEEE, Jinchao Zhou, Kunfeng Qiu, Zhuangzhi Chen, Dongwei Xu, Shilian Zheng, Member, IEEE, Xiaoniu Yang
Official implement of the paper, AvgNet: Adaptive Visibility Graph Neural Network and It’s Application in Modulation Classification(https://ieeexplore.ieee.org/abstract/document/9695244/)

# Introduction
Our digital world is full of time series and graphs which capture the various aspects of many complex systems. Traditionally, there are respective methods in processing these two different types of data, e.g., Recurrent Neural Network (RNN) and Graph Neural Network (GNN), while in recent years, time series could be mapped to graphs by using the techniques such as Visibility Graph (VG), which simultaneously captures relevant aspects of both local and global dynamics in an easy way, so that researchers can use graph algorithms to mine the knowledge in time series and gain special latent graph representation features. Such mapping methods establish a bridge between time series and graphs, and have high potential to facilitate the analysis of various real-world time series. However, the VG method and its variants are just based on fixed rules and thus lack of flexibility, largely limiting their application in reality. In this paper, we propose an Adaptive Visibility Graph (AVG) algorithm that can adaptively map time series into graphs, based on which we further establish an end-to-end classification framework AvgNet, by utilizing GNN model DiffPool as the classifier. We then adopt AvgNet for radio signal modulation classification which is an important task in the field of wireless communication. The simulations validate that AvgNet outperforms a series of advanced deep learning methods, achieving the state-of-the-art performance in this task.

# Citation
If this work is useful for your research, please consider citing:
```
@ARTICLE{9695244,  
author={Xuan, Qi and Zhou, Jinchao and Qiu, Kunfeng and Chen, Zhuangzhi and Xu, Dongwei and Zheng, Shilian and Yang, Xiaoniu}, 
journal={IEEE Transactions on Network Science and Engineering},   title={Adaptive Visibility Graph Neural Network and Its Application in Modulation Classification},   year={2022},
volume={},
number={},
pages={1-1},
doi={10.1109/TNSE.2022.3146836}}
```

# Datasets
The dataset can be downloaded from the deepsig official website(https://www.deepsig.ai/datasets)

# Requirements
- Python 3.7.9
- torch 1.6.0
- torch-geometric 1.6.1
# Training
For the RadioML2016.10a dataset:
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 10a --lr 0.001 --batch-size 128 --epochs 20 --num_workers 4 --num_filter 11
```

For the RadioML2016.10b dataset:
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 10b --lr 0.001 --batch-size 128 --epochs 20 --num_workers 4 --num_filter 11
```
