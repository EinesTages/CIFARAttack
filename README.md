# CIFarAttack

## Competition URL:

https://challenge.aisafety.org.cn/#/competitionDetail?id=15

## About Model Download:

**Model in TorchVision**

https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view

**Model in RobustBench**

Automatic Download By RobustBench Using gdown

## Run Code:

~~~shell
python main.py --eps=22.0 (replace to yours)
python generate.py
python to_pic.py
~~~

## Further Improvement:

比较明显的上分点：

1. 显然攻击难度在这种情况下主要取决于样本自身的特征，有的样本其实并不需要很大的扰动就可以攻到了，可以有梯度的生成样本，增大SSIM，大概算了一下可能能提到47分左右

2. 显然可以用生成网络，其实集成了**AdvGAN**，但是训练成本比较高，就没再继续训下去，用生成模型做对抗样本的很多，你可以直接在损失函数里优化SSIM，同时加一些正则项之类，感觉结合上面再和之前结果投票，上限在50分左右

3. **Rethinking Model Ensemble in Transfer-based Adversarial  Attacks (ICLR2024)**     https://arxiv.org/abs/2303.09105

4. 加入对抗训练后的**transformer-based model**还可以上分，但我觉得意义不大