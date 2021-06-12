## 基于增强实体表示的序列到序列模型

### 预处理数据集
相比于Interspeech的代码，数据处理部分没有不同。

### 训练
同EER。

### 测试
同EER。

### 模型改进
相比于Interspeech的EER，SEER模型层面主要改进一下两点：<br/>
（1）Encoder增加了线性转化。<br/>
（2）Decoder初始化Encoder的激活函数从ReLU换成了Tanh。<br/>
Have Fun！:)

