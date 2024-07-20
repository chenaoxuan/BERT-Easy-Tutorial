# BERT-Easy-Tutorial

English</a> | <a href="docs/README_ZH.md">简体中文</a>

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

This is a minimalist introductory tutorial on the **BERT** model proposed by Google AI Language in 2018, designed to guide beginners to understand BERT at the fastest speed possible.

## 🚩Features

- Ultra simplified data: A dataset consisting of only two lines of text.

- Ultra detailed comments: Each line of core code has an explanation.

- Comprehensive tutorial documents: Detailed introduction to data pipeline in both Chinese and English.

- No redundant code: No need for graphics card training, configuration loading, model saving, and other operations.

- Easily configure environment: Only Python, Pytorch, Numpy are needed to run.

## 💻Environment
Environmental requirements: Python 3.x、Pytorch>0.4、Numpy  
The environment used for the development of this project is:
```shell
Python 3.10.0
pip install torch==1.12.0 numpy==1.26.3
```

## 🚀Quickstart

Run ```prepare_vocab.py``` with default configuration to convert data/capus.txt to a data/vocab (Optional as vocab is already provided).

Run ```train.by``` with default configuration to start training!

For detailed explanations of data and code, please refer to the <a ref="Tutorial. md">Tutorial</a>.

## Reference

Some modules of this project refer to the following repos:

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)