## 基于流到图的序列到序列模型

### 预处理数据集
首先需要说明的是相比于ICASSP的代码，一个不同的地方是适配了[GLMP](https://github.com/jasonwu0731/GLMP)和[DDMN](https://github.com/siat-nlp/DDMN)的数据格式，分别对应文件夹data_glmp和data_ddmn。
因此我们需指定一个数据文件夹，默认为data，例如，这里用DDMN(论文基于此数据)，那么:
```console
❱❱❱ mv data_ddmn data
```

### 训练
InCar:
```console
❱❱❱ python myTrain.py -lr=0.001 -hdd=128 -dr=0.2 -bsz=8 -ds=kvr -B=15 -ss=10.0
```
CamRest:
```console
❱❱❱ python myTrain.py -lr=0.001 -hdd=128 -dr=0.3 -bsz=8 -ds=cam -B=5 -ss=10.0
```
Multi-WOZ 2.1:
```console
❱❱❱ python myTrain.py -lr=0.001 -hdd=128 -dr=0.4 -bsz=8 -ds=woz -B=5 -ss=10.0
```

### 测试
```console
❱❱❱ python myTest.py -path=<path_to_saved_model> 
```

### 模型改进
相比于ICASSP的FG2Seq，FG2Seq_v2模型层面主要改进一下两点：<br/>
（1）将KB Integration部分换成全局的KB实体。<br/>
（2）增加先生成后填充策略。<br/>
还有一点就是，模型的Flow-to-Graph部分代码重写了。Have Fun！:)
