<!-- TOC -->
* [Tutorial](#tutorial)
  * [Dataset Description](#dataset-description)
  * [Build vocabulary](#build-vocabulary)
  * [Task 1: Randomly mask and predict reconstruction](#task-1-randomly-mask-and-predict-reconstruction)
    * [Example](#example)
      * [Construct Input](#construct-input)
      * [Construct Target](#construct-target)
      * [segment_label](#segment_label)
    * [The specific implementation of random mask](#the-specific-implementation-of-random-mask)
  * [Task 2: Predict whether two sub-sentences come from the same sentence](#task-2-predict-whether-two-sub-sentences-come-from-the-same-sentence)
    * [Example](#example-1)
      * [Construct Input](#construct-input-1)
      * [Construct Target](#construct-target-1)
    * [The specific implementation of randomly selecting sub-sentences](#the-specific-implementation-of-randomly-selecting-sub-sentences)
  * [BERT model](#bert-model)
    * [Input](#input)
    * [Generate mask](#generate-mask)
    * [Three types of embedding](#three-types-of-embedding)
      * [TokenEmbedding](#tokenembedding)
      * [PositionEmbedding](#positionembedding)
        * [Initialization](#initialization)
        * [forward](#forward)
      * [SegmentEmbedding](#segmentembedding)
      * [Summation](#summation)
    * [Encoder](#encoder)
      * [Multi-Head-Self-Attention](#multi-head-self-attention)
      * [Feedforward-Network](#feedforward-network)
  * [BERTLM model](#bertlm-model)
    * [MaskedLanguageModel](#maskedlanguagemodel)
    * [NextSentencePrediction](#nextsentenceprediction)
<!-- TOC -->


# Tutorial
## Dataset Description
The dataset used in the project is ```data/corpus.txt```, which is an extremely simple dataset consisting of only two lines of sentences. In each line of the sentence, it is further divided into two sub-sentences by \t to indicate the context of the sentence (the first half can also be considered as a question and the second half as an answer).

i.e.：  
Welcome to the [\t] the jungle  
I can stay [\t] here all night

## Build vocabulary
 
Establish a corresponding Vocab for ```data/corpus.txt``` (requiring additional special tokens). The essence of the Vocab is a dictionary that maps English words to numbers:   
{'< pad >': 0, '< unk >': 1, '< eos >': 2, '< sos >': 3, '< mask >': 4, 'the': 5, 'I': 6, 'Welcome': 7, 'all': 8, 'can': 9, 'here': 10, 'jungle': 11, 'night': 12, 'stay': 13, 'to': 14}    
Where < pad > denotes padding, < unk > denotes unknown, < sos > denotes the start of a sentence, < eos > denotes the end of a sentence, < mask > denotes the word is masked.

This step is completed by ```prepare_vocab.py```, and the vocabulary is ultimately saved as ```data/vocab```.

## Task 1: Randomly mask and predict reconstruction
The Bert paper proposed two training tasks.

The first task is to randomly mask some words in the sentence, and then have Bert predict what these masked words actually are.

### Example
#### Construct Input
1. Convert 'Welcome to the jungle' to the index sequence of the Vocab: 7 14 5 and 5 11
2. Now, a part of the sequence is randomly masked, and the sequence becomes: 4, 14, 5, and 5, 10 (i.e. 7 is changed to the mask token, and 11 is changed to a random token)
3. Merge the sequence and add start and end tokens, the sequence becomes: 3 4 14 5 2 5 10 2 (note that sos 3 and eos 2 are added before and after the first half of the sentence, but the second half of the sentence is only added eos 2)
4. Fill to predefined length: 3 4 14 5 2 5 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (the preset length here is 20)

At this point, the input sequence for Bert task one has been constructed, which is the ```bert_input``` in ```bert/dataset/dataset.py``` (note that it only shows the case where the two ends come from the same sentence, and it is also possible to use a combination of 'Welcome to the' and 'here all night' in task two, but this does not affect the entire process)

#### Construct Target
In constructing input, it can be seen that changing 7 to mask token 4 and 11 to random token 10 requires Bert to re-predict them to their original values. Therefore, the target for constructing is 0 7 0 0 0 0 0 0 11 0 0 0 0 0 0 0 0 0 0 0 (note that the positions of sos and eos are replaced by padding 0)

At this point, the target sequence for Bert task one has been generated, which is the ```bert_label``` in ```bert/dataset/dataset.py```.

#### segment_label
Indicate where the current word comes from, for example: 1 1 1 1 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0   
If it is 1, it means that the token comes from the first half of the paragraph (first half sub-sentence); If it is 2, it means that the token comes from the second half (second half sub-sentence); If it is 0, it means that the token comes from padding. This item needs to be used in the encoding of the Bert model.


### The specific implementation of random mask
Randomly mask with a 15% probability, the specific code is:
```python
import random
sentence='Welcome to the'
toekns=[] # Convert to index
output_label=[] # The target Bert needs to predict 

for i,word in enumerate(sentence):
    prob = random.random() # Randomly select a number from 0 to 1
    if prob < 0.15:
        prob /= 0.15
        if prob < 0.8: # 80% probability of using mask filling
            tokens[i] = vocab.mask_index
        elif prob < 0.9: # 10% chance to use completely random filling
            tokens[i] = random.randrange(len(vocab))
        else: # 10% probability remains unchanged, that is, the mask is removed. Note that although there is no mask here, the output label still needs to make predictions
            toekns[i]=Search the Vocal-index corresponding to word
        output_label[i]=Search the Vocal-index corresponding to word
    else: # 不mask
        toekns[i]=Search the Vocal-index corresponding to word
        output_label[i]=0 # Indicates that there is currently no mask
```
Construct tokens in this way, which are the positional index sequences of English words in the Vocab dictionary; And the target that Bert's task one needs to predict is the output_label.

## Task 2: Predict whether two sub-sentences come from the same sentence

By using \t, a sentence in a row of the dataset can be split into two sub-sentences. The second task has a 50% probability of selecting sub-sentences that do not come from the same sentence, allowing Bert to predict their source.

### Example
#### Construct Input
1. Read a whole row from the dataset based on the input line index, and divide it into two sub-sentences using \t, such as t1=Welcome to the, t2=the jungle
2. There is a 50% chance to replace t2 with a sub-sentence from another row, such as replacing t2 with "here all night"

For specific code, please refer to the __ getitem__  method in ```bert/dataset/dataset.py```

#### Construct Target
If bert_input consists of t1 and t2 from the same sentence, then the target is 1; Otherwise, it is 0.

At this point, it is constructed as is_next_label in ```bert/dataset/dataset.py```

### The specific implementation of randomly selecting sub-sentences
```python
t1, t2 = get_corpus_line(index) # Read a complete sentence based on the index (line number) and divide it into t1 and t2 using \t
if random.random() > 0.5:
    return t1, t2, 1 # 50% chance to return two sub-sentence from the same sentence, marked as 1
else:
    return t1, get_random_line(), 0 # 50% chance to return two sub-sentence from the different sentence, marked as 0
```

## BERT model
```bert/model/bert.py/BERT```

From the main structure perspective, Bert first performs three types of embedding on the input:
1. TokenEmbedding, encoding the input dictionary index sequence into dense feature embeddings
2. PositionEmbedding，generate position encoding for sequences to distinguish the position of tokens
3. SegmentEmbedding，encode segment_label mentioned earlier to distinguish sentence sources

Then, a multi-layer Transformer Encoder structure is applied to extract features

We will only focus on the forward part of the Bert model below
### Input
- x: The sequence with a shape of [batch_size, seq_len], where seq_len is set to 20 by default, represents the index sequence after random masking and padding
- segment_info：The sequence with a shape of [batch_size, seq_len]. A value of 1 indicates that the current token is from the first half of the sentence, a value of 2 indicates that the current token is from the second half of the sentence, and a value of 0 indicates that the current token is from padding

### Generate mask
1. Since padding is filled with 0, (x>0) indicates the generation of a sequence of boolean type with the same shape as [batch_size, seq_len]. A sequence with >0 is True, otherwise the padding position is False, indicating invisibility
2. unscueze(1) extends in dimension 1 to generate a sequence of [batch_size, 1, seq_len]
3. repeat(1, x.size(1), 1), repeat seq_len times in the expanded dimension to generate a sequence of [batch_size, seq_len, seq_len]
```python
mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
```
### Three types of embedding
#### TokenEmbedding
```bert/model/embedding/token.py```   
TokenEmbedding is a single layer nn.Embedding(num_embeddings=len(vocab), embedding_dim=hidden, padding_idx=0)

Encode the integer sequence of x [batch_size, seq_len] into a dense vector representation [batch_size, seq_len, hidden]

Note that padding idx=0 indicates that the specified integer sequence value of 0 is padding, and the corresponding vector encoding (regardless of how many times the training parameters are updated) is also all 0.

#### PositionEmbedding
```bert/model/embedding/position.py```   

Bert uses sine and cosine embedding to represent absolute positions. In the init method, a position encoding map with a shape of [max_len, hidden (i.e. d_model)] can be preprocessed in advance, and then only the first seq_len returns can be truncated in the forward. Therefore, when initializing parameters, it is necessary to pass hidden instead of max_len

##### Initialization
Core formula

$$ PE(pos,2i)=sin(pos/10000^{2i/dim}) $$

$$ PE(pos,2i+1)=cos(pos/10000^{2i/dim}) $$

Among them, pos represents the position of the word in the token sequence, with a value range of 0~seq_len; i represents the position of the dimension, with values ranging from 0 to dim; dim represents the dimension length

1. First, create a full 0-tensor for [max_len, d_model], and then make modifications on it
```python
pe = torch.zeros(max_len, d_model).float()
```
2. Generate a sequence of 0~max_len-1, and then expand the dimension to become [max_len, 1], where each digit in 0~max_len-1 is on a line, indicating the position of the word in the token sequence, unsqueeze is for subsequent broadcasting
```python
position = torch.arange(0, max_len).float().unsqueeze(1)
```
3. Using log and exp to calculate the score part, the shape of div_term is [d_model//2]
```python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
```
Formula reasoning:

$$ e^{(2i) \cdot -(log(10000)/ dim)}=\frac{1}{e^{\frac{2i \cdot log(10000)}{dim}}} $$

Extract the denominator

$$ e^{\frac{2i \cdot log(10000)}{dim}}=e^{log(10000^{\frac{2i}{dim}})}=10000^{\frac{2i}{dim}} $$

4. Calculate sine and cosine. pe is the full 0-tensor of [max_len, d_model]; The sequence position with position [max_len, 1]; div_term is [d_model//2]. Firstly, after position * div_term, the shape becomes [max_len, d_model//2], which means that each number from 0 to max_len-1 is multiplied by div_term and completed through broadcasting; Then, after passing through sin or cos, the shape remains unchanged; Afterwards, use slicing and use sin encoding for even positions of pe, and cos encoding for odd positions
```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```
5. Continue to increase the dimension from [max_len, d_model] to [1, max_len, d_model]， Increase batch size dimension to facilitate forward broadcasting
```python
pe = pe.unsqueeze(0)
```
6. pe does not need to calculate gradients, so it is registered in the buffer
```python
self.register_buffer('pe', pe)
```

##### forward
Input x with a shape of [batch_size, seq_len]. Directly extract the first seq_len from the max_len of [1, max_len, d_model] and return them
```python
return self.pe[:, :x.size(1)]
```

#### SegmentEmbedding
```bert/model/embedding/segment.py```   
A single layer nn.Embedding(num_embeddings=3, embedding_dim=hidden, padding_idx=0)

Encode segment_info, to distinguish whether the current word comes from the first half, second half, or padding of the sentence

#### Summation
Obtain the above three embeddings separately and sum them up
```python
x = self.token(x) + self.position(x) + self.segment(segment_info)
```

### Encoder
```bert/model/transformer/encoder.py```   
Each layer of Encoder consists of multi-head-self-attention, feedforward-network combined with LayerNorm and Dropout, with the core being the first two. The advantage of the Encoder layer is that the output data dimension is exactly the same as the input data dimension, so multiple layers of Encoders can be stacked, which constitutes the main structure of Bert.

#### Multi-Head-Self-Attention
```bert/model/transformer/attention.py```
The difference between self-attention and cross-attention is that the former's QKV come from the same input, while the latter's Q comes from one input and KV come from another input.

QKV are obtained by transforming the input x through three linear layers.

The core formula for self-attention calculation is 

$$ output=softmax(\frac{Q \cdot K^T}{\sqrt{dim}}) \cdot V $$

#### Feedforward-Network
```bert/model/utils/feed_forward.py```
Composed of two linear layers combined with activation function and LayerNorm.

The core point is that after the first linear layer, the number of feature channels increases; After the second linear layer, the number of feature channels returns to its original state.

## BERTLM model
```bert/model/bert.py/BERTLM```

This model is a further encapsulation of the BERT model. In the above introduction, Bert was only used to extract features, but did not make predictions for the two training tasks. BERTLM achieves predictive output for two tasks through further encapsulation.

### MaskedLanguageModel
Apply a linear layer on x with a shape of [batch_size, seq_len, hidden], which becomes [batch_size, seq_len, vocab_size]， And use softmax for the last dimension.

At this point, a word in the vocabulary has been predicted for each token in the sequence. Use nn.NLLLoss(ignore_index=0) in ```train.py``` performs loss calculation on the predicted words and bert_label, where ignore_index=0 means ignoring items in bert_label that are 0 (i.e. those that are not masked)

### NextSentencePrediction

Apply a linear layer on x with a shape of [batch_size, seq_len, hidden], which becomes [batch_size, seq_len, 2], and only use the 0th one in the sequence (because this task only needs to predict whether the source of the sub-sentences is same, only one output is needed), i.e. [:, 0], apply softmax and return.

Use nn.NLLLoss() in ```train.py``` performs loss calculation on the predicted binary classification results and is_next_label