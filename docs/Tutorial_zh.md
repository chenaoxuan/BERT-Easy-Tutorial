<!-- TOC -->
* [教程](#教程)
  * [数据集说明](#数据集说明)
  * [建立词表](#建立词表)
  * [任务一：随机mask并预测重建](#任务一随机mask并预测重建)
    * [举例说明](#举例说明)
      * [构建输入](#构建输入)
      * [构建目标](#构建目标)
      * [segment_label](#segment_label)
    * [随机mask的具体实现](#随机mask的具体实现)
  * [任务二：预测两个子句是否来自同一个句子](#任务二预测两个子句是否来自同一个句子)
    * [举例说明](#举例说明-1)
      * [构建输入](#构建输入-1)
      * [构建目标](#构建目标-1)
    * [随机选择子句的实现方式](#随机选择子句的实现方式)
  * [BERT模型](#bert模型)
    * [输入](#输入)
    * [生成mask](#生成mask)
    * [三种编码](#三种编码)
      * [Token编码](#token编码)
      * [Position编码](#position编码)
        * [初始化](#初始化)
        * [forward](#forward)
      * [Segment编码](#segment编码)
      * [求和](#求和)
    * [Encoder](#encoder)
      * [多头自注意力](#多头自注意力)
      * [前馈网络](#前馈网络)
  * [BERTLM模型](#bertlm模型)
    * [MaskedLanguageModel](#maskedlanguagemodel)
    * [NextSentencePrediction](#nextsentenceprediction)
<!-- TOC -->

# 教程
## 数据集说明
项目使用的数据集为```data/corpus.txt```，这是一个仅由两行句子构成的极为简单的数据集。在每行句子中，又通过\t分为两个子句，表示句子的上下文（也可认为前半句为question，后半句为answer）。

即：  
Welcome to the [\t] the jungle  
I can stay [\t] here all night

## 建立词表

为```data/corpus.txt```建立对应的词表vocab（需要额外引入一些特殊的token），词表的本质就是一个字典，实现由英文单词到数字的映射：  
{'< pad >': 0, '< unk >': 1, '< eos >': 2, '< sos >': 3, '< mask >': 4, 'the': 5, 'I': 6, 'Welcome': 7, 'all': 8, 'can': 9, 'here': 10, 'jungle': 11, 'night': 12, 'stay': 13, 'to': 14}  
其中< pad >表示填充，< unk >表示未知，< sos >表示句子的开头，< eos >表示句子的结尾，< mask >表示被遮挡。

该步骤由```prepare_vocab.py```完成，最终将词表保存为```data/vocab```。

## 任务一：随机mask并预测重建
在Bert论文中，提出了两种训练任务。

第一个任务是随机将句子中的一些词给遮挡（mask）住，然后使Bert预测这些被遮挡的词原来是什么。

### 举例说明
#### 构建输入
1. 将Welcome to the	the jungle转为词表的index序列：7 14 5 和 5 11  
2. 现在随机mask掉一部分，序列变为：4 14 5 和 5 10 （即将7变为mask token，将11变为随机token）    
3. 合并序列，并添加起止token，序列变为：3 4 14 5 2 5 10 2（需要注意的是，在前半句前后添加sos 3和eos 2，但是后半句只在最后添加eos 2）
4. 填充到预定义长度：3 4 14 5 2 5 10 2 0 0 0 0 0 0 0 0 0 0 0 0（此处预设长度为20）

至此，构建完了Bert任务一的输入序列，即```bert/dataset/dataset.py```中的```bert_input```（注意，只是展示了前后两端来自同一个句子的情况，也可能任务二中使用Welcome to the和here all night进行组合，但是这并不影响整个流程）

#### 构建目标
在构建输入中可知，将7变为mask token 4，将11变为随机token 10，这两个变化需要Bert将其重新预测为原来的值，因此构建目标target为0 7 0 0 0 0 11 0 0 0 0 0 0 0 0 0 0 0 0 0（注意，sos和eos的位置用padding的0代替）

至此，生成了Bert任务一的目标序列，即```bert/dataset/dataset.py```中的```bert_label```。

#### segment_label
表明当前的单词来自哪里，例如：1 1 1 1 1 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0  
若为1，则表示token来自前半段（前半个子句）；若为2，则表示token来自后半段（后半个子句）；若为0，则表示token来自padding。这一项在Bert模型的编码中需要使用。

### 随机mask的具体实现
以15%的概率随机mask，具体逻辑为：
```python
import random
sentence='Welcome to the'
toekns=[] # 转为index
output_label=[] # Bert需要预测的target

for i,word in enumerate(sentence):
    prob = random.random() # 随机一个0~1的数
    if prob < 0.15:
        prob /= 0.15
        if prob < 0.8: # 80%概率使用mask填充
            tokens[i] = vocab.mask_index
        elif prob < 0.9: # 10%概率使用完全的随机填充
            tokens[i] = random.randrange(len(vocab))
        else: # 10%概率不变，即取消mask，注意虽然这里没有mask，但是output_label仍需要做出预测
            toekns[i]=去vocal中查找word对应的index
        output_label[i]=去vocal中查找word对应的index
    else: # 不mask
        toekns[i]=去vocal中查找word对应的index
        output_label[i]=0 # 表示当前没有mask
```
通过这种方式构造出tokens，即英文单词在vocab字典中的位置索引序列；和output_label，即Bert的任务一需要预测的target。

## 任务二：预测两个子句是否来自同一个句子
通过\t可以将数据集的某一行句子拆分为两个子句。第二个任务有50%的概率选择不是来自同一个句子的子句，让Bert进行预测其来源。

### 举例说明
#### 构建输入
1. 根据传入的行号，读取数据集中的某一整行，并通过\t分为两个子句，如t1=Welcome to the，t2=the jungle
2. 有50%的概率将t2换为来自其他行的子句，如可以将t2更换为here all night

即任务一构建输入中的bert_input，具体代码可查看```bert/dataset/dataset.py```中的__getitem__方法

#### 构建目标
若bert_input由来自同一个句子的t1和t2构成，则目标为1；否则为0。

至此，构建为了```bert/dataset/dataset.py```中的is_next_label

### 随机选择子句的实现方式
```python
t1, t2 = get_corpus_line(index) # 根据index（行号）读取一个完整的句子，通过\t分为t1和t2
if random.random() > 0.5:
    return t1, t2, 1 # 50%的概率返回来自同一个句子的两个子句，并标记为1
else:
    return t1, get_random_line(), 0 # 50%的概率返回来自不同句子的两个子句，并标记为0
```

## BERT模型
```bert/model/bert.py/BERT```

从主要结构上来看，Bert对输入首先进行三种编码：
1. token编码，将输入的字典索引序列编码为稠密特征嵌入
2. position编码，生成序列的位置编码，区分token的位置
3. segment编码，对前文提到的segment_label进行编码，区分句子来源

然后进行多层的Transformer-Encoder结构，用于提取特征

下面我们只关注Bert模型的forward部分
### 输入
- x：shape为[batch_size, seq_len]的序列，seq_len默认设置为了20，表示随机mask和padding后的index序列
- segment_info：shape为[batch_size, seq_len]的序列，seq_len默认设置为了20，值为1表示当前token来自前半句，值为2表示当前token来自后半句，值为0表示当前token为padding的

### 生成mask
1. 由于padding用0填充，因此(x > 0)表示生成一个shape同样为[batch_size, seq_len]的bool类型的序列，>0的为True，否则填充的位置为False，表示不可见
2. unsqueeze(1)在维度1扩展，生成[batch_size, 1, seq_len]的序列
3. repeat(1, x.size(1), 1)，在扩充出来的维度重复seq_len次，生成[batch_size, seq_len, seq_len]的序列
```python
mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
```
### 三种编码
#### Token编码
```bert/model/embedding/token.py```   
TokenEmbedding
即单层的nn.Embedding(num_embeddings=len(vocab), embedding_dim=hidden, padding_idx=0)

将x [batch_size, seq_len]]的整数序列编码为转换为密集向量表示[batch_size, seq_len, hidden]

注意，padding_idx=0表示指定整数序列值0是padding填充，对应的向量编码（无论随着训练参数更新了多少次）也全部为0。

#### Position编码
```bert/model/embedding/position.py```   
Bert使用表示绝对位置的正余弦编码，可以在init方法中提前预处理出一个shape为[max_len, hidden(即d_model)]的位置编码map，然后在forward中只截取前seq_len个返回即可。因此，传参初始化时，必须传递hidden，可以不传递max_len

##### 初始化
核心公式

$$ PE(pos,2i)=sin(pos/10000^{2i/dim}) $$

$$ PE(pos,2i+1)=cos(pos/10000^{2i/dim}) $$

其中pos表示单词在token序列中的位置，取值范围0 ~ seq_len；i表示维度的位置，取值0 ~ dim；dim表示维度长度

1. 首先创建[max_len, d_model]的全0 tensor，后续在这上面做修改
```python
pe = torch.zeros(max_len, d_model).float()
```
2. 生成0 ~ max_len-1的序列，然后扩充后的维度，变成[max_len, 1]，即0~max_len-1中每个数字一行，这表示单词在token序列中的位置，unsqueeze是为了后续的广播
```python
position = torch.arange(0, max_len).float().unsqueeze(1)
```
3. 借助log和exp计算分数部分，div_term的shape为[d_model//2]
```python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
```
公式推理：

$$ e^{(2i) \cdot -(log(10000)/ dim)}=\frac{1}{e^{\frac{2i \cdot log(10000)}{dim}}} $$

分母单独拿出来

$$ e^{\frac{2i \cdot log(10000)}{dim}}=e^{log(10000^{\frac{2i}{dim}})}=10000^{\frac{2i}{dim}} $$

4. 计算正余弦。pe的为[max_len, d_model]的全0tensor；position为[max_len, 1]的序列位置；div_term为[d_model//2]。首先position * div_term后，shape变为[max_len, d_model//2]，即0~max_len-1每个数都乘以div_term，通过广播完成；然后经过sin或cos，shape不变；后使用切片，对于pe的偶数位置使用sin编码，奇数位置使用cos编码
```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```
5. 继续增加维度，由[max_len, d_model]变为[1, max_len, d_model]，增加batch size维度，方便forward广播
```python
pe = pe.unsqueeze(0)
```
6. pe不需要计算梯度，因此注册到缓冲区
```python
self.register_buffer('pe', pe)
```

##### forward
输入x，shape为[batch_size, seq_len]。直接从[1, max_len, d_model]的max_len个中截取前seq_len个返回
```python
return self.pe[:, :x.size(1)]
```

#### Segment编码
```bert/model/embedding/segment.py```   
单层的nn.Embedding(num_embeddings=3, embedding_dim=hidden, padding_idx=0)

对segment_info进行编码，即段编码，区分当前的单词是来自句子前半段、后半段、还是padding

#### 求和
分别获取上述三种编码并求和
```python
x = self.token(x) + self.position(x) + self.segment(segment_info)
```

### Encoder
```bert/model/transformer/encoder.py```   
每层Encoder由多头自注意力、前馈网络再搭配LayerNorm、Dropout构成，核心为前两者。Encoder层的好处是输出的数据维度与输入的数据维度完全一致，因此多层Encoder可以堆叠，这就构成了Bert的主体结构。

#### 多头自注意力
```bert/model/transformer/attention.py```
自注意力相较于交叉注意力，区别在于前者的QKV来自同一个输入，而后者的Q来自一个输入KV来自另一个输入。

QKV分别通过三个线性层对输入x进行变换得到。

自注意力计算的核心公式为    

$$ output=softmax(\frac{Q \cdot K^T}{\sqrt{dim}}) \cdot V $$

#### 前馈网络
```bert/model/utils/feed_forward.py```
由两个线性层并配合激活函数和LayerNorm构成。

核心要点是第一个线性层之后，特征通道数增加；第二个线性层后，特征通道数又变回为原来的样子。

## BERTLM模型
```bert/model/bert.py/BERTLM```

这个模型是对BERT模型的进一步封装。在上面的介绍中，Bert只用来提取特征，但是并未对两个训练任务做出预测。BERTLM通过进一步的封装，实现了对两个任务的预测输出。

### MaskedLanguageModel
在shape为[batch_size, seq_len, hidden]的x上应用线性层，变为[batch_size, seq_len, vocab_size]，并对最后一个维度使用softmax。

此时，为序列中的每一个token都预测出了一个词表中的单词。在```train.py```中使用nn.NLLLoss(ignore_index=0)对预测的单词和bert_label进行损失计算，ignore_index=0表示忽略bert_label中为0的项（即没有mask掉的那些）

### NextSentencePrediction

在shape为[batch_size, seq_len, hidden]的x上应用线性层，变为[batch_size, seq_len, 2]，并只使用序列中的第0个（因为这个任务只需要预测子句来源是否一致，只需要一个输出即可），即[:,0]，应用softmax后返回。

在```train.py```中使用nn.NLLLoss()对预测的二分类结果和is_next_label进行损失计算