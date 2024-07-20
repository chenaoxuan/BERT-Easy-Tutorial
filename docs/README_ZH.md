# BERT-Easy-Tutorial

<a href="../README.md">English</a> | 简体中文</a>

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

这是一个关于2018年Google AI Language提出的**BERT**模型的极简入门教程，用于带领初学者以最快的速度了解BERT。

## 🚩特点

- 超精简的数据：仅由两行文本构成数据集。
- 超详细的注释：每行核心代码都有解释说明。
- 超全面的文档：中英文文档详细介绍数据流水线。
- 无冗余的代码：不需要显卡训练、配置加载、模型保存等操作。
- 易配置的环境：只需要Python、Pytorch、Numpy即可运行。

## 💻环境

环境要求：Python 3.x、Pytorch>0.4、Numpy   

本项目开发使用的环境：
```shell
# Python 3.10.0
pip install torch==1.12.0 numpy==1.26.3
```

## 🚀快速开始

默认配置运行```prepare_vocab.py```，将```data/corpus.txt```转为```data/vocab```词表（非必须，vocab已经提供）。

默认配置运行```train.py```，启动训练！

数据与代码详细说明请查看<a href="Tutorial_zh.md">教程</a>。

## 参考

本项目的一些模块涉及以下项目：

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

