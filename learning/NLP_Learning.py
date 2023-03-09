import collections  # 用来統計詞頻
from collections import defaultdict
import math
from math import log2
import os
import re
import random
import sys
import zipfile

import jieba
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential, Model
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from matplotlib import pylab
import numpy as np
from numpy.linalg import svd as SVD
import pandas as pd
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import tensorflow as tf

import nltk   # 用来分词
from nltk import ne_chunk
from nltk import word_tokenize
import nltk.corpus
from nltk.text import TextCollection

# 載入NLTK的範例文句
try:
    from nltk.book import text1, text2, text3
except LookupError:
    nltk.download('gutenberg')
    nltk.download('genesis')
    nltk.download('inaugural')
    nltk.download('nps_chat')
    nltk.download('webtext')
    nltk.download('treebank')
    from nltk.book import text1, text2, text3
# *** Introductory Examples for the NLTK Book ***
# Loading text1, ..., text9 and sent1, ..., sent9
# Type the name of the text or sentence to view it.
# Type: 'texts()' or 'sents()' to list the materials.
# text1: Moby Dick by Herman Melville 1851
# text2: Sense and Sensibility by Jane Austen 1811
# text3: The Book of Genesis
# text4: Inaugural Address Corpus
# text5: Chat Corpus
# text6: Monty Python and the Holy Grail
# text7: Wall Street Journal
# text8: Personals Corpus
# text9: The Man Who Was Thursday by G . K . Chesterton 1908

# region Demo1: nltk token + Keras Embedding
# https://ithelp.ithome.com.tw/articles/10193924
# scikit-learn 須升級至 0.19
# pip install -U scikit-learn
# 在 python 執行 nltk.download(), 下載 data

# 探索數據分析(EDA)
# 計算訓練資料的字句最大字數
maxlen = 0
word_freqs = collections.Counter()  # 看起來是 default-dict
num_recs = 0
with open('NLP/Sentiment1_training.txt', 'r+', encoding='utf8') as f:
    for line in f:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ', maxlen)
print('nb_words ', len(word_freqs))


# 準備數據
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word_index["PAD"] = 0
word_index["UNK"] = 1
index2word = {v: k for k, v in word_index.items()}
X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)

# 讀取訓練資料，將每一單字以 dictionary 儲存
with open('NLP/Sentiment1_training.txt', 'r+', encoding='UTF-8') as content:
    for i, line in enumerate(content):
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word_index:
                seqs.append(word_index[word])
            else:
                seqs.append(word_index["UNK"])
        X[i] = seqs
        try:
            y[i] = int(label)
        except:
            print(label)

# 字句長度不足補空白        
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# 資料劃分訓練組及測試組
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型構建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# region build model
model = Sequential()
# 加『嵌入』層
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
# 加『LSTM』層
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
# binary_crossentropy:二分法
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# endregion

print(model.summary())

# 模型訓練
model.fit(Xtrain, 
          ytrain, 
          batch_size=BATCH_SIZE, 
          epochs=NUM_EPOCHS,
          validation_data=(Xtest, ytest))


# 預測
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('預測', '真實', '句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1, MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))

# 模型存檔 >> creates a HDF5 file 'model.h5'
model.save('SaveModel/Sentiment1.h5')

# 自己輸入測試
INPUT_SENTENCES = ['I love it.', 'It is so boring.', 'I love it althougn it is so boring.']
XX = np.empty(len(INPUT_SENTENCES), dtype=list)

# 轉換文字為數值
i = 0
for sentence in INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word_index:
            seq.append(word_index[word])
        else:
            seq.append(word_index['UNK'])
    XX[i] = seq
    i += 1


XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# 預測，並將結果四捨五入，轉換為 0 或 1
labels = [int(round(x[0])) for x in model.predict(XX)]
label2word = {1: '正面', 0: '負面'}

# 顯示結果
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
# endregion


# region Demo2: Keras Tokenizer + Embedding
# https://ithelp.ithome.com.tw/articles/10194633
# 參數設定
BASE_DIR = './data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# 將 glove.6B.100d.txt 檔案轉成 dict，key:單字, value:詞向量
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
# 顯示the單字的詞向量
print(len(embeddings_index["the"]))

# 讀取訓練資料檔，包含 20_newsgroup 資料夾下所有子目錄及檔案，共20類 
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

# 20類的代號及名稱                
print(labels_index)
print('Found %s texts.' % len(texts))

# 將訓練資料的單字轉成向量
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# 將訓練字句截長補短
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 將訓練資料分為訓練組及驗證組
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_test = data[-num_validation_samples:]
y_test = labels[-num_validation_samples:]

# region build model
# 轉成 Embedding 層的 input vector
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# 載入預訓模型，trainable = False 表示不重新計算
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# 訓練模型
print('Training model.')
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Dropout(0.25)(x)

x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.25)(x)

x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.25)(x)

x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
# endregion

# summarize the model
print(model.summary())

train_history = model.fit(x_train, 
                          y_train,
                          batch_size=128,
                          epochs=20,
                          validation_split=0.2)


# 模型存檔
model.save('./SaveModel/embedding.h5')  # creates a HDF5 file 

# evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Accuracy: %f' % (accuracy*100))
# endregion


# region Demo4：命名實體識別(Named Entity Recognition，NER)
# https://ithelp.ithome.com.tw/articles/10194822
# NER是一種解析文件並標註各個實體類別的技術，例如人、組織、地點...等。
# 除了詞性(Part Of Speech, POS)外，也標註出 PERSON、ORGANIZATION 等命名實體，有助於文件的解析。
sent = "Mark is studying at Stanford University in California"
print(ne_chunk(nltk.pos_tag(word_tokenize(sent)), binary=False))
# (S
#  (PERSON Mark/NNP)
#  is/VBZ
#  studying/VBG
#  at/IN
#  (ORGANIZATION Stanford/NNP University/NNP)
#  in/IN
#  (GPE California/NNP))
# endregion


# region Demo5
# https://ithelp.ithome.com.tw/articles/10194822
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
        print(nltk.sem.rtuple(rel))
# [ORG: 'WHYY'] 'in' [LOC: 'Philadelphia']
# [ORG: 'McGlashan &AMP; Sarrail'] 'firm in' [LOC: 'San Mateo']
# [ORG: 'Freedom Forum'] 'in' [LOC: 'Arlington']
# [ORG: 'Brookings Institution'] ', the research group in' [LOC: 'Washington']
# [ORG: 'Idealab'] ', a self-described business incubator based in' [LOC: 'Los Angeles']
# [ORG: 'Open Text'] ', based in' [LOC: 'Waterloo']
# [ORG: 'WGBH'] 'in' [LOC: 'Boston']
# [ORG: 'Bastille Opera'] 'in' [LOC: 'Paris']
# [ORG: 'Omnicom'] 'in' [LOC: 'New York']
# [ORG: 'DDB Needham'] 'in' [LOC: 'New York']
# [ORG: 'Kaplan Thaler Group'] 'in' [LOC: 'New York']
# [ORG: 'BBDO South'] 'in' [LOC: 'Atlanta']
# [ORG: 'Georgia-Pacific'] 'in' [LOC: 'Atlanta']
# endregion


# region Demo6
# E:\E-Block\Google 雲端硬碟\j32u4ukh\tensorflow\tensorflow\examples\udacity
# http://localhost:8888/notebooks/5_word2vec.ipynb
# Download the data from the source website if necessary.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    path = "data/" + filename
    if not os.path.exists(path):
        filename, _ = urlretrieve(url + filename, path)
    else:
        print("File has existed.")
    
    # 檢查文件大小，或許是為了避免下載到遭修改的文件
    statinfo = os.stat(path)    
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % path)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + path + '. Can you get to it with a browser?')
  
    return path


path = maybe_download('text8.zip', 31344016)


def read_data(path):
    with zipfile.ZipFile(path) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(path)
print('Data size %d' % len(words)) # Data size 17005207

print(type(words))
print(words[:5])
# Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # ex: {'a': 1, 'anarchism': 4, 'term': 2, 'UNK': 0, 'as': 3}
    
    # data: 將文字轉為數字(該文字在字典中的 index )
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        
        data.append(index)
    
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    # ex: {0: 'UNK', 1: 'as', 2: 'a', 3: 'term', 4: 'anarchism'}
    return data, count, dictionary, reverse_dictionary


'''用來查看程式碼內容，不是主程序'''
_size = 5
text = ['anarchism', 'originated', 'as', 'a', 'term', 'as', 'a', 'term', 'as']
result = [['UNK', -1]]
counter = collections.Counter(text)
# counter.most_common(n) >>> 取出 counter 中最多的 n 個
most_counter = counter.most_common(_size - 1)
result.extend(most_counter)
# check area
print(counter)
# Counter({'as': 3, 'a': 2, 'term': 2, 'anarchism': 1, 'originated': 1})
print("==========")
print(most_counter)
# [('as', 3), ('a', 2), ('term', 2), ('anarchism', 1)]
print("==========")
print(result)
# [['UNK', -1], ('as', 3), ('a', 2), ('term', 2), ('anarchism', 1)]
print("==========")
for word, _ in result:
    print(word, _)

'''用來查看程式碼內容，不是主程序'''
dictionary = dict()
for word, _ in result:
    dictionary[word] = len(dictionary)

values = dictionary.values()
keys = dictionary.keys()
_zip = zip(values, keys)
reverse_dictionary = dict(_zip)
# check
print(dictionary)
# {'a': 1, 'anarchism': 4, 'term': 2, 'UNK': 0, 'as': 3}
print("==========")
print(values)
# dict_values([1, 4, 2, 0, 3])
print("==========")
print(keys)
# dict_keys(['a', 'anarchism', 'term', 'UNK', 'as'])
print("==========")
_zip = zip(values, keys)
print(list(_zip))
# [(2, 'a'), (4, 'anarchism'), (3, 'term'), (0, 'UNK'), (1, 'as')]
print("==========")
print(reverse_dictionary)
# {0: 'UNK', 1: 'as', 2: 'a', 3: 'term', 4: 'anarchism'}

data, result, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', result[:5])
print('Sample data', data[:10])
# Hint to reduce memory.
del words
# Function to generate a training batch for the skip-gram model.
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
#    所謂斷言（Assertion），指的是程式進行到某個時間點，斷定其必然是某種狀態，
#    具體而言，也就是斷定該時間點上，某變數必然是某值，或某物件必具擁有何種特性值。
#    當程式執行到 assert 時，必須符合後面的條件，否則跳出例外
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
#    print("Origin data_index:", data_index)
    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # [ skip_window target skip_window ]
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
#    print("batch:", batch)
#    print("labels:")
#    print(labels)
#    print("span:", span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
#    print("Middle data_index:", data_index)
    
#    slide = batch_size // num_skips
#    print("滑動次數>>batch_size // num_skips:", slide)
    for i in range(batch_size // num_skips):
        # target label at the center of the buffer
        target = skip_window
        targets_to_avoid = [skip_window]        
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)            
            targets_to_avoid.append(target)
#            print("===== Round:", i * num_skips + j, " =====")
#            print("batch:", batch)
#            print("buffer:", buffer)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            
#            print("target index:", target)
#            print("batch[", i * num_skips + j, "] = buffer[", skip_window, "], value:", buffer[skip_window])
#            print("labels[", i * num_skips + j, ", 0] = buffer[", target, "], value:", buffer[target])
#            print("targets_to_avoid:", targets_to_avoid)
#            print("batch:", batch)
#            print("labels:")
#            print(labels)
#        print("==========")
            
        # data = [1, 2, 3, 4, 5]
        # buffer maxlen=span >>> 持續 append 的話，會產生滑動的效果，[1, 2, 3] >>> [2, 3, 4] >>> [3, 4, 5]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
#    print("Current data_index:", data_index)
    return batch, labels


'''用來查看程式碼內容，不是主程序'''
num_skips, skip_window = 2, 1
batch, labels = generate_batch(batch_size=4, num_skips=num_skips, skip_window=skip_window)
print()
print('with num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
print('\tbatch:', [reverse_dictionary[bi] for bi in batch])
print('\tlabels:', [reverse_dictionary[li] for li in labels.reshape(-1)])


# '''用來查看程式碼內容，不是主程序'''
def myDiv(A, B):
    try:
        if A % B != 0:
            raise AssertionError
        else:
            return A / B
    except AssertionError as ae:
        print(ae)


print(myDiv(A=4, B=2))
ae = myDiv(A=5, B=2)
print(ae)

'''用來查看程式碼內容，不是主程序'''
_skip_window = 2
_span = 2 * _skip_window + 1
_buffer = collections.deque(maxlen=_span)
print(_buffer)
print("==========")
_data_index = 0
for _ in range(_span + 2):
    _buffer.append(data[_data_index])
    _data_index = (_data_index + 1) % len(data)
    print(_data_index)
print("==========")
print(_buffer)
print("==========")

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    # data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

# data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
# with num_skips = 2 and skip_window = 1:
#    batch: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']
#    labels: ['as', 'anarchism', 'originated', 'a', 'as', 'term', 'of', 'a']
# with num_skips = 4 and skip_window = 2:
#    batch: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']
#    labels: ['anarchism', 'originated', 'a', 'term', 'as', 'term', 'originated', 'of']

'''用來查看程式碼內容，不是主程序'''
print("num_skips:", num_skips, ", skip_window:", skip_window)
print(type(batch))
print(batch.shape)
print(batch)
print("==========")
print(type(labels))
print(labels.shape)
print(labels)
print("==========")
print('batch:', [reverse_dictionary[bi] for bi in batch])
print('labels:', [reverse_dictionary[li] for li in labels.reshape(-1)])

'''用來查看程式碼內容，不是主程序'''
array = np.ndarray(shape=(5,), dtype=np.int32)
print(array)

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64    # Number of negative examples to sample.

'''用來查看程式碼內容，不是主程序'''
print(range(10))
print("==========")
sample = random.sample(range(10), 2)
print(sample)
print("==========")
print(np.array(sample))

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # Variables.
    # tf.random_uniform(shape, min, max, ...)
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # tf.truncated_normal(shape, mean, stddev, ...) 截斷式常態分配，截斷在2倍標準差
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Model.
    # Look up embeddings for inputs. 选取一个张量里面索引对应的元素
    # 取出 embeddings 中的元素，根據 train_dataset 中的 index
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                                     biases=softmax_biases,
                                                     inputs=embed,
                                                     labels=train_labels,
                                                     num_sampled=num_sampled,
                                                     num_classes=vocabulary_size))
    
    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities 
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    
    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

'''用來查看程式碼內容，不是主程序'''
a = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
print(a)
# tf.nn.sampled_softmax_loss(
#    weights,
#    biases,
#    labels,
#    inputs,
#    num_sampled, 每批隨機抽樣的類數。
#    num_classes, 可能的類數。
#    num_true=1,  每個訓練示例的目標類數。
#    sampled_values=None,
#    remove_accidental_hits=True,
#    partition_strategy='mod',
#    name='sampled_softmax_loss',
#    seed=None
# )

num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
            
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                # number of nearest neighbors
                top_k = 8 
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
# endregion


# region Demo7: tf-idf
# https://ccjou.wordpress.com/2009/11/04/svd-%E6%96%BC%E8%B3%87%E8%A8%8A%E6%AA%A2%E7%B4%A2%E8%88%87%E6%96%87%E6%9C
# %AC%E6%90%9C%E5%B0%8B%E7%9A%84%E6%87%89%E7%94%A8/
string = '''
考慮下面這個小型語料庫，稱為「我的語料庫」，它包含選自本網站的 5 份文件：
“特殊矩陣 (1)：冪零矩陣”、“特殊矩陣 
(2)：正規矩陣”、“特殊矩陣 
(3)：么正矩陣（酉矩陣）”、“特殊矩陣 
(4)：Householder 矩陣”和“特殊矩陣 
(5)：冪等矩陣”。
我們挑選的 10 個字詞依序是：
「秩」、「行列式」、「行空間」、「正交」、「特徵值」、
「對稱」、「對角化」、「投影」、「相似」和「正定」。
表一整理了「我的語料庫」裡每個字詞於各文件中出現的次數 (n_{ij})，
每份文件的字詞總數 (\sum_k n_{kj})，以及包含某字詞的文件數目 (df_i)。
字詞「正定」並未出現於任何文件中，我們可以將它由字詞—文件關聯矩陣中刪除。
從表一的的統計資料可計算出 TF-IDF，
「我的語料庫」的 9\times 5 階字詞—文件關聯矩陣如下：
'''

patten = re.compile("文件")
result = patten.findall(string)
print(result)
print(len(result))


def n_termInDoc(term, doc):
    patten = re.compile(term)
    result = patten.findall(doc)
    return len(result)


print(n_termInDoc(term="文件", doc=string))

terms = ["秩", "行列式", "行空間", "正交", "特徵值", "對稱", "對角化", "投影", "相似"]

df = pd.DataFrame()
for i in range(1, 6):
    doc = "doc%d" % i
    path = "data/tf_idf/%s.txt" % doc
    dictionary = defaultdict(int)
    with open(path, "r") as f:
        for line in f.readlines():
            for term in terms:
                dictionary[term] += n_termInDoc(term, line)
    df[doc] = pd.Series(dictionary)
print(df)

series = df["doc1"]
print(series)
print(series.sum())

_sum = []
for col in df.columns:
    series = df[col]
    _sum.append(series.sum())
print(_sum)

df.loc["Sum"] = _sum
print(df)

series = df.apply(lambda x: x > 0)
#      doc1   doc2   doc3   doc4   doc5
# 對稱   False   True  False   True   True
# 對角化   True   True   True   True   True
# 投影   False  False  False   True   True
# 正交   False   True   True   True   True
# 特徵值   True   True   True   True   True
# 相似    True   True  False   True  False
# 秩     True   True  False  False   True
# 行列式   True  False   True  False  False
# 行空間   True  False  False  False   True
# Sum   True   True   True   True   True

row = df.iloc[0, :]
print(type(row))
print(row)

row_value = row.values
print(np.where(row_value > 0))
print(len(np.where(row_value > 0)[0]))


def n_df(row):
    return len(np.where(row > 0)[0])


print("n_df(row):", n_df(row))
series = df.apply(lambda row: n_df(row), axis=1)

df["df"] = df.apply(lambda row: n_df(row), axis=1)
print(df)

df_value = df.values
sum_array = df_value.sum(axis=0).astype(np.float32)
df_array = np.array([n_df(row) for row in df_value]).astype(np.float32)

tf_frame = df_value.copy().astype(np.float32)
print(tf_frame)

for col in range(tf_frame.shape[1]):
    tf_frame[:, col] = tf_frame[:, col] / sum_array[col]
print(tf_frame)

n = df_value.shape[1]
idf_array = np.array([[log2(n / dfi) + 1 for dfi in df_array]]).T
print(idf_array.shape)
print(idf_array)

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
b = np.array([[10, 20, 30]])

print("a*b")
print(a*b)
print("np.multiply(a, b)")
print(np.multiply(a, b))
print("np.multiply(a, b.T)")
print(np.multiply(a, b.T))
print("b.shape", b.shape)
print("b.T.shape", b.T.shape)

tf_idf = np.multiply(tf_frame, idf_array)
print("tf_idf.shape", tf_idf.shape)
print(tf_idf)
print(df_value)


A = np.mat([[1, 2, 3],
            [4, 5, 6]])

U, sigma, VT = SVD(A)

print("===== %s =====" % "U")
print(U)
print("===== %s =====" % "sigma")
print(sigma)
print("===== %s =====" % "VT")
print(VT)

# [m, n] >>> [m, m] * [m, n] * [n, n]
s, v, d = SVD(tf_idf)
print("===== %s =====" % "s")
print(s.shape)
print(s)
print("===== %s =====" % "v")
print(v.shape)
print(v)
print("===== %s =====" % "d")
print(d.shape)
print(d)

v_matrix = np.zeros((tf_idf.shape[0], tf_idf.shape[1]))
print(v_matrix.shape)
print(v_matrix)

for i in range(tf_idf.shape[1]):
    v_matrix[i, i] = v[i]
print(v_matrix)

svd = np.dot(np.dot(s, v_matrix), d)
print(svd)
# endregion


# region Demo: 句子相似度(詞向量平均作為句向量，是較差的方法，句向量才是較佳的方法)
# https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
dim = 0
word_vecs = {}

""" 開啟詞向量檔案 """
# 之後可以從 word_vecs 這個dict中取得詞向量
with open('data/cna.cbow.cwe_p.tar_g.512d.0.txt', encoding='utf8') as f:
    for line in f:
        # 假設我們的詞向量有300維
        # 由word以及向量中的元素共301個
        # 以空格分隔組成詞向量檔案中一行
        tokens = line.strip().split()

        # 第一行是兩個整數，分別代表有幾個詞向量，以及詞向量維度
        if len(tokens) == 2:
            dim = int(tokens[1])  # 詞向量維度
            continue

        word = tokens[0]
        vec = np.array([float(t) for t in tokens[1:]])
        word_vecs[word] = vec

'''查看word_vecs(詞向量字典)'''
print(len(word_vecs))  # 158566
print(type(word_vecs.keys()))  # <class 'dict_keys'>
demo_keys = list(word_vecs.keys())[:5]
print(demo_keys)  # ['較窄', '駕崩', '林德嘉', '區對', '接龍']
print(len(word_vecs['接龍']))  # 512
print(type(word_vecs['接龍']))  # <class 'numpy.ndarray'>
print(word_vecs['接龍'].shape)  # (512,)
print(word_vecs['接龍'].max())  # 5.088133
print(word_vecs['接龍'].min())  # -4.802558

max_value, min_value = 0, 0
for key in word_vecs.keys():
    array = word_vecs[key]
    max_value = max(max_value, array.max())
    min_value = min(min_value, array.min())
print("max_value:%f, min_value:%f" % (max_value, min_value))

# 示範比賽題目
# 我們要從answers中挑出應該接在dialogue之後的短句
dialogue = "如果飛機在飛行當中打一個小洞的話 會不會影響飛行的安全呢"
answers = [
    "其實狗搖尾巴有很多種方式 高興搖尾巴 生氣也搖尾巴",
    "如果這個洞的話經過仔細的設計的話 應該不至於造成太大問題",
    "所以只要依照政府規定 在採收前十天不要噴灑農藥",
    "靜電才是加油站爆炸的元凶 手機不過是代罪羔羊",
    "我們可以用表面張力及附著力的原理 來測試看看",
    "不過蝦子死亡後 身體會釋放出有毒素的體液 可能造成水的變質"]

# dialogue 斷詞後的詞，有在word_vecs(詞向量字典)的個數，用來取向量平均之用
emb_cnt = 0

avg_dlg_emb = np.zeros((dim,))  # dim = 512
# jieba.cut 會把dialogue作分詞
# 對於有在word_vecs裡面的詞我們才把它取出
# 最後詞向量加總取平均，作為句子的向量表示
for word in jieba.cut(dialogue):
    if word in word_vecs:
        avg_dlg_emb += word_vecs[word]
        emb_cnt += 1
avg_dlg_emb /= emb_cnt
print(avg_dlg_emb.shape)

emb_cnt = 0
max_idx = -1
max_sim = -10
# 在六個回答中，每個答句都取詞向量平均作為句向量表示
# 我們選出與dialogue句子向量表示cosine similarity最高的短句
for idx, ans in enumerate(answers):
    avg_ans_emb = np.zeros((dim,))
    for word in jieba.cut(ans):
        if word in word_vecs:
            avg_ans_emb += word_vecs[word]
            emb_cnt += 1

    # 其實就是 cos 值，用來衡量兩個向量之間的距離
    sim = (np.dot(avg_dlg_emb, avg_ans_emb) / np.linalg.norm(avg_dlg_emb) / np.linalg.norm(avg_ans_emb))
    print("Ans#%d: %f" % (idx, sim))

    if sim > max_sim:
        max_idx = idx
        max_sim = sim

print("Answer:%d" % max_idx)
# endregion

