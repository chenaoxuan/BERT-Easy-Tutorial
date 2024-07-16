<!-- TOC -->
* [代码文档](#代码文档)
  * [数据集示例](#数据集示例)
  * [任务一：随机mask并预测重建](#任务一随机mask并预测重建)
    * [随机mask的实现方式](#随机mask的实现方式)
    * [样例](#样例)
      * [构建输入](#构建输入)
      * [构建目标](#构建目标)
      * [segment_label](#segment_label)
  * [任务二：预测两个子句是否来自同一个句子](#任务二预测两个子句是否来自同一个句子)
    * [随机选择子句的实现方式](#随机选择子句的实现方式)
    * [样例](#样例-1)
      * [构建输入](#构建输入-1)
      * [构建目标](#构建目标-1)
  * [Bert](#bert)
    * [输入](#输入)
    * [生成mask](#生成mask)
    * [TokenEmbedding](#tokenembedding)
    * [PositionalEmbedding](#positionalembedding)
      * [初始化](#初始化)
      * [forward](#forward)
    * [SegmentEmbedding](#segmentembedding)
<!-- TOC -->

# 代码文档
## 数据集示例
示例句子（中间用\t分割，分为前半段和后半段，这是为BERT的任务二准备的）：  
Welcome to the	the jungle  
I can stay	here all night

建立词表vocab为（需要额外引入一些特殊的token）：  
{'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4, 'the': 5, 'I': 6, 'Welcome': 7, 'all': 8, 'can': 9, 'here': 10, 'jungle': 11, 'night': 12, 'stay': 13, 'to': 14}

## 任务一：随机mask并预测重建
### 随机mask的实现方式
以15%的概率随机mask，使用BERT预测mask掉的值，具体逻辑为：
```python
import random
sentence='Welcome to the'
toekns=[] # 转为index
output_label=[] # bert需要预测的target

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
通过这种方式构造出tokens，即index序列；和output_label，即bert的下游任务需要预测的target。

### 样例
#### 构建输入
1. 将Welcome to the	the jungle转为词表的index序列：7 14 5 和 5 11  
2. 现在随机mask掉，序列变为：4 14 5 和 5 10 （即将7变为mask token，将11变为随机token）    
3. 合并序列，并添加起始token，序列变为：3 4 14 5 2 5 10 2（需要注意的是，在前半句前后添加sos 3和eos 2，但是后半句只在最后添加eos 2）
4. 填充到预定义长度：3 4 14 5 2 5 10 2 0 0 0 0 0 0 0 0 0 0 0 0（此处预设长度为20）

至此，使用tokens构建完了bert任务一的输入序列，即dataset.py中的bert_input（注意，只是展示了前后两端来自同一个句子的情况，也有可能为Welcome to the和here all night构成两个子句，但是这并不影响整个流程）

#### 构建目标
在构建x中可知，将7变为mask token 4，将11变为随机token 10，这两个变化需要bert将其重新预测为原来的值，因此构建目标target为0 7 0 0 0 0 11 0 0 0 0 0 0 0 0 0 0 0 0 0（注意，sos和eos的位置用padding的0代替）

至此，生成了bert任务一的目标序列，dataset.py中的bert_label

#### segment_label
表明当前的单词来自哪里，例如：1 1 1 1 1 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0  
若为1，则表示token来自前半段（前半个子句）；若为2，则表示token来自后半段（后半个子句）；若为0，则表示token来自padding

## 任务二：预测两个子句是否来自同一个句子
在一个完整句子中，通过\t拆分为两端（两个子句），任务二由50%的概率选择不是同一个句子中的子句，让bert进行预测

### 随机选择子句的实现方式
```python
t1, t2 = get_corpus_line(index) # 根据index读取一个完整的句子，通过\t分为t1和t2
if random.random() > 0.5:
    return t1, t2, 1 # 50%的概率返回来自同一个句子的两个子句，并标记为1
else:
    return t1, get_random_line(), 0 # 50%的概率返回来自不同句子的两个子句，并标记为0
```
### 样例
#### 构建输入
即任务一构建输入中的bert_input，具体代码可查看dataset.y中的__getitem__方法

#### 构建目标
若bert_input由来自不同句子的两个t1和t2构成，则为1；否则为0.

至此，构建为了dataset.py中的is_next_label

## Bert
只关注forward部分

### 输入
- x：shape为[batch_size, seq_len]的序列，seq_len默认设置为了20，表示随机mask和padding后的index序列
- segment_info：shape为[batch_size, seq_len]的序列，seq_len默认设置为了20，值为1表示当前token来自前半句，值为2表示当前token来自后半句，值为0表示当前token为padding的

### 生成mask
1. 由于padding用0填充，因此(x > 0)表示生成一个shape同样为[batch_size, seq_len]的bool类型的序列，>0的为True，否则填充的位置为False，表示不可见
2. unsqueeze(1)在维度1扩展，生成[batch_size, 1, seq_len]的序列
3. repeat(1, x.size(1), 1)，在扩充出来的维度重复seq_len次，生成[batch_size, seq_len, seq_len]的序列
4. unsqueeze(1)在继续维度1扩展，生成[batch_size, 1, seq_len, seq_len]的序列，这一步是提前扩充出来，为后面的multi-head做准备
```python
mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
```

### TokenEmbedding
**TokenEmbedding**
即单层的nn.Embedding(num_embeddings=len(vocab), embedding_dim=hidden, padding_idx=0)

将x [batch_size, seq_len]]的整数序列编码为转换为密集向量表示[batch_size, seq_len, hidden]

注意，padding_idx=0表示指定整数序列值0是padding填充，对应的向量编码（无论随着训练参数更新了多少次）也全部为0。

### PositionalEmbedding
BERT使用表示绝对位置的正余弦编码，可以在init方法中提前预处理出一个shape为[max_len, hidden(即d_model)的位置编码map，然后在forward中只截取前token_len个返回即可。因此，传参初始化时，必须传递hidden，可以不传递max_len

#### 初始化
核心公式

$$ PE(pos,2i)=sin(pos/10000^{2i/dim}) $$

$$ PE(pos,2i+1)=cos(pos/10000^{2i/dim}) $$

其中pos表示单词在token序列中的位置，取值范围0 ~ token_len；i表示维度的位置，取值0 ~ dim；dim表示维度长度

1. 首先创建[max_len, d_model]的全0tensor，后续在这上面做修改
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

#### forward
输入x，shape为[batch_size, seq_len]。直接从[1, max_len, d_model]的max_len个中截取前seq_len个返回
```python
return self.pe[:, :x.size(1)]
```

### SegmentEmbedding

单层的nn.Embedding(num_embeddings=3, embedding_dim=hidden, padding_idx=0)

对segment_info进行编码，即段编码，区分当前的单词是来自句子前半段、后半段、还是padding
