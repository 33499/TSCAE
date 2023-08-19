## **用于自监督表示学习的教师-学生互补掩码自动编码器**

<img src=".\imgs_model\11.png" alt="图11" style="zoom: 25%;" />

&emsp;&emsp;This is a PyTorch/GPU re-implementation of the paper Teacher-student Complementary Mask Autoencoder for Self-Supervised Representation Learning

### TSCAE模型简介
---

​&emsp;&emsp;针对自监督表示学习中掩码图像建模（MIM）方法存在的上下游不匹配问题，提出了一种新的基于教师-学生网络的互补掩码预训练模型（TSCAE模型）。该模型由教师模块和学生模块组成，学生模块是一个编码器结构，教师模块由编码器、解码器和掩码预测模块构成，编码器用于图像的表征学习，教师模块中的解码器负责从可见图像块的表征预测掩码图像块的表征。为了从大量无标签数据中学习更丰富的表示，设计了两种前置任务：教师模块中解码器预测得到的掩码图像块表征经过掩码预测模块预测真实的图像像素；引入对比损失，教师模块中解码器预测得到的表征与学生网络中编码器学习到的表征进行表征空间上的对比学习。此外，本文提出了互补掩码机制，即对教师和学生网络均输入一张完整的图片，对教师网络而言，输入图片随机掩蔽掉一部分，列如75%，学生网络掩蔽输入图片剩余的部分，即25%。TSCAE模型在COCO等中小规模数据集上做预训练，在三个经典的数据集和两个私有数据集上进行微调，实验表明TSCAE能够取得优异的性能，教师-学生互补掩码方法能够减少上下游不匹配的间隙。最后通过实验我们发现，在图片质量较差的中小型数据集Tiny-ImageNet上进行预训练，TSCAE在下游任务中表现比MAE优秀很多，预训练的图片质量对编码器的图片表征学习能力影响更小。

### 可视化解释模型
---

#### 像素复原可视化指令

```shell
python3 tscae_visualize.py
```

#### 图片复原的可视化
MLP模块恢复图像效果如下：

<img src=".\imgs_model\47.png" alt="图11" style="zoom: 25%;" />

#### 注意力图可视化
使用分类标记作为最后一层中不同头的查询时的注意力图如下：

<img src=".\imgs_model\2.png" alt="图11" style="zoom: 25%;" />

### 预训练指令

```shell
python3 main_pretrain.py 
```

### 微调指令

```shell
python3 main_finetune.py
```

**相关模型在Tiny-ImageNet和医学肝脏数据集上的微调准确率比较**
---

<img src=".\imgs_model\57.png" alt="图57" style="zoom: 80%;" />
