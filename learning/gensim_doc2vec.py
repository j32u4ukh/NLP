import collections
import os
import random
import re
import time

import gensim
import jieba
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
from pprint import pprint
import smart_open


# region Demo: Doc2Vec 句向量
# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'
print("gensim.__path__[0]: ", gensim.__path__[0])
# gensim.__path__[0]:  C:\Users\ETlab\.conda\envs\py36\lib\site-packages\gensim
print("os.sep: ", os.sep)
# os.sep:  \
print("lee_train_file: ", lee_train_file)
# C:\Users\ETlab\.conda\envs\py36\lib\site-packages\gensim\test\test_data\lee_background.cor
print("lee_test_file: ", lee_test_file)
# C:\Users\ETlab\.conda\envs\py36\lib\site-packages\gensim\test\test_data\lee.cor


# region read_corpus function
def read_corpus(fname, tokens_only=False):
    # gensim.utils.simple_preprocess 效果類似斷詞
    # gensim.models.doc2vec.TaggedDocument 將斷詞結果與 index 綑綁在一起
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # tokens_only=False
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


min_index = 0
min_length = 25000
min_content = ""

max_index = 0
max_length = 0
max_content = ""

with smart_open.open(lee_train_file, encoding="iso-8859-1") as file:
    for index, text in enumerate(file):
        length = len(text)

        # 最短的句子
        if length < min_length:
            min_length = length
            min_index = index
            min_content = text

        # 最長的句子
        if length > max_length:
            max_length = length
            max_index = index
            max_content = text

print("min index:{}, length:{}".format(min_index, min_length))  # min index:207, length:292
print("max index:{}, length:{}".format(max_index, max_length))  # max index:250, length:3838

print(min_content)
# Geoff Huegill has continued his record-breaking ways
# at the World Cup short course swimming
# in Melbourne, bettering the Australian record in the 100 metres butterfly.
# Huegill beat fellow Australian Michael Klim, backing up
# after last night setting a world record in the 50 metres butterfly.


preprocess = gensim.utils.simple_preprocess(min_content)
print(preprocess)
# ['geoff', 'huegill', 'has', 'continued', 'his', 'record', 'breaking', 'ways',
# 'at', 'the', 'world', 'cup', 'short', 'course', 'swimming',
# 'in', 'melbourne', 'bettering', 'the', 'australian', 'record', 'in', 'the', 'metres', 'butterfly',
# 'huegill', 'beat', 'fellow', 'australian', 'michael', 'klim', 'backing', 'up',
# 'after', 'last', 'night', 'setting', 'world', 'record', 'in', 'the', 'metres', 'butterfly']

print(gensim.models.doc2vec.TaggedDocument(preprocess, [min_index]))
# TaggedDocument(
# ['geoff', 'huegill', 'has', 'continued', 'his', 'record', 'breaking', 'ways',
# 'at', 'the', 'world', 'cup', 'short', 'course', 'swimming',
# 'in', 'melbourne', 'bettering', 'the', 'australian', 'record', 'in', 'the', 'metres', 'butterfly',
# 'huegill', 'beat', 'fellow', 'australian', 'michael', 'klim', 'backing', 'up',
# 'after', 'last', 'night', 'setting', 'world', 'record', 'in', 'the', 'metres', 'butterfly'],
# [207])
# endregion

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

print(type(train_corpus))  # <class 'list'>
print(len(train_corpus))   # 300
print(train_corpus[min_index])
# TaggedDocument(
# ['geoff', 'huegill', 'has', 'continued', 'his', 'record', 'breaking', 'ways',
# 'at', 'the', 'world', 'cup', 'short', 'course', 'swimming',
# 'in', 'melbourne', 'bettering', 'the', 'australian', 'record', 'in', 'the', 'metres', 'butterfly',
# 'huegill', 'beat', 'fellow', 'australian', 'michael', 'klim', 'backing', 'up',
# 'after', 'last', 'night', 'setting', 'world', 'record', 'in', 'the', 'metres', 'butterfly'],
# [207])

# 建立空物件
# model.vocabulary.min_count 可取得此處的 min_count
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=7)


# region build_vocab 遍歷一次文本建立詞典
# sol 1
# update=False 一次性建立詞典(優點:min_count 不會分批次計算；缺點:對記憶體消耗高)
model.build_vocab(train_corpus)

# sol 2
# 批次建立詞典(優點:對記憶體消耗低；缺點: min_count 也會分批次計算)
length = len(train_corpus)
step = 27
sum_len = 0
for _from in range(0, length, step):
    to = _from + step
    if to > length:
        to = length
    print(_from, to)
    corpus = train_corpus[_from: to]
    sum_len += len(corpus)
    if _from == 0:
        model.build_vocab(corpus)
    else:
        model.build_vocab(corpus, update=True)
# endregion

# model.wv.vocab 詞彙表
vocabulary_dict = model.wv.vocab
print(len(model.wv.vocab.keys()))

for num, key in enumerate(vocabulary_dict.keys()):
    print(key, vocabulary_dict[key])
    if num == 5:
        break
# refuse Vocab(count:2, index:2747, sample_int:4294967296)
# television Vocab(count:11, index:807, sample_int:4294967296)
# damaging Vocab(count:2, index:2748, sample_int:4294967296)
# ship Vocab(count:4, index:1750, sample_int:4294967296)
# mouth Vocab(count:2, index:2749, sample_int:4294967296)

dict_value = vocabulary_dict["refuse"]
print(type(dict_value))  # <class 'gensim.models.keyedvectors.Vocab'>
print(dict_value)        # Vocab(count:2, index:2747, sample_int:4294967296)


# 因應批次建立詞典，事後過濾頻率較低的詞
def selectDict():
    a = {"b": 1,
         "c": 2,
         "d": 3,
         "e": 2,
         "f": 1}
    A = pd.Series(a)
    A.sort_values(axis=0, ascending=False)
    print(A)
    # b    1
    # c    2
    # d    3
    # e    2
    # f    1
    # dtype: int64

    B = A.where(A > 1)
    print(B)
    B = B.dropna()
    print(B)
    C = B.to_dict()
    print(C)


# region Training model
length = len(train_corpus)
step = 27
sum_len = 0
start = time.time()
for epoch in range(model.epochs):
    for _from in range(0, length, step):
        to = _from + step
        if to > length:
            to = length
        corpus = train_corpus[_from: to]
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print("train: %d - %d" % (_from, to))
#        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
print("Cost time:", time.time() - start)
# endregion


# model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
vec = model.infer_vector("what do you want to do".split(" "))
print(type(vec))
print(vec.shape)


def similarity(v1, v2):
    # 若輸入句子，則先轉換成句向量
    if type(v1) == str:
        v1 = model.infer_vector(v1.split(" "))
        v2 = model.infer_vector(v2.split(" "))
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


vec1 = model.infer_vector("good girl".split(" "))
vec2 = model.infer_vector("good woman".split(" "))
vec3 = model.infer_vector("bad girl".split(" "))

sim12 = similarity(vec1, vec2)
sim13 = similarity(vec1, vec3)
sim23 = similarity(vec2, vec3)
print("sim12:", sim12)  # sim12: -0.063614674
print("sim13:", sim13)  # sim13: 0.24054623
print("sim23:", sim23)  # sim23: 0.061400477
print("sim:", similarity("good girl", "good boy"))              # sim: 0.031871706
print("sim:", similarity("i was hit by Jason", "i hit Jason"))  # sim: 0.12129312

# 評估模型
print(train_corpus[207])
print("="*30)
print(train_corpus[207].words)
print(len(model.docvecs))

ranks = []
second_ranks = []
sims = None
doc_id = None
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))    
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)    
    second_ranks.append(sims[1])

print(type(sims))
print(len(sims))
print("doc_id:", doc_id)
doc_id = 0
sims = sims[:10]
for docid, sim in sims:
    print(docid, sim)
print("*"*50)
array = [docid for docid, sim in sims]
print(type(array))
print(array)
print(array.index(17))
collections.Counter(ranks)  # Results vary between runs due to random seeding and very small corpus

doc_id = 299
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

# Testing the Model
# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
# endregion


# region Demo4:Doc2Vec VS Keras Embbeding
# mini_token
# filetype: train / test
def read_files(filetype):
    def binaryLabels(n_pos, n_neg):
        one = np.ones((n_pos,))
        zero = np.zeros((n_neg,))    
        return np.r_[one, zero]
    
    def rm_tags(text):
        # 將符合正規表達式的部分換成空字串
        rm_tag = re.compile(r'<[^>]+>')
        return rm_tag.sub('', text)
    
    path = "data/aclImdb/"
    all_texts = []
    
    # get all file under positive_path
    # ex: "data/aclImdb/train/pos/"
    # ex: "data/aclImdb/test/pos/"    
    positive_path = path + filetype + "/pos/"
    files = os.listdir(positive_path)
    n_pos = 0
    for file in files:
        n_pos += 1
        file_name = os.path.join(positive_path, file)
        with open(file_name, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
        if n_pos % 500 == 0:
            print("read pos-%s file, files number:%5d" % (filetype, n_pos))
    
    # get all file under negative_path
    # ex: "data/aclImdb/train/neg/"
    # ex: "data/aclImdb/test/neg/"
    negative_path = path + filetype + "/neg/"
    files = os.listdir(negative_path)
    n_neg = 0
    for file in files:
        n_neg += 1
        file_name = os.path.join(negative_path, file)
        with open(file_name, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
        if n_neg % 500 == 0:
            print("read neg-%s file, files number:%5d" % (filetype, n_neg))
    
    all_labels = binaryLabels(n_pos, n_neg)
    print("===== %s files finish been read, files number:%d =====" % (filetype, n_pos + n_neg))
    
    return all_texts, all_labels


train_text, y_train = read_files("train")
test_text, y_test = read_files("test")

print(type(test_text))  # <class 'list'>
print(len(test_text))   # 25000
min_index = 0
min_length = 25000
for index, text in enumerate(test_text):
    if len(text) < min_length:
        min_length = len(text)
        min_index = index
print(min_index, min_length)  # 21499 32
print(test_text[min_index])
# Read the book, forget the movie!

# 產生句向量
docVector_model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=20, epochs=40)
docVector_model.build_vocab(train_corpus)
docVector_model.train(train_corpus, total_examples=docVector_model.corpus_count, epochs=docVector_model.epochs)


def similarity(v1, v2):
    # 若輸入句子，則先轉換成句向量
    if type(v1) == str:
        v1 = docVector_model.infer_vector(v1.split(" "))
        v2 = docVector_model.infer_vector(v2.split(" "))
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


vec1 = docVector_model.infer_vector("good girl".split(" "))
vec2 = docVector_model.infer_vector("good woman".split(" "))
vec3 = docVector_model.infer_vector("bad girl".split(" "))

sim12 = similarity(vec1, vec2)
sim13 = similarity(vec1, vec3)
sim23 = similarity(vec2, vec3)
print("sim12:", sim12)
print("sim13:", sim13)
print("sim23:", sim23)

seq = train_text[0]
s = gensim.utils.simple_preprocess(seq)
print(s)

vec = docVector_model.infer_vector(s)
print(type(vec))
# <class 'numpy.ndarray'>
print(vec.shape)
# (512,)
print(vec)


"""Keras Embedding"""


def kerasEmbedding(train_text, y_train, test_text, y_test):
    # 1. num_words：決定單詞數量 (出現次數最多的 N 個單詞)
    input_dim = 2000
    # 但計算時全部都會需要計算，因此 token.word_index token.word_counts token.index_word 等，都是包含所有詞的字典
    token = Tokenizer(num_words = input_dim)
    # 2. 形塑 token
    token.fit_on_texts(train_text)
    # 3. 將影評文字轉為數字list
    x_train_seq = token.texts_to_sequences(train_text)
    x_test_seq = token.texts_to_sequences(test_text)
    # 4. 讓數字list長度都變100
    x_train = sequence.pad_sequences(x_train_seq, maxlen = 100)
    x_test = sequence.pad_sequences(x_test_seq, maxlen = 100)
    
    # 建立網路
    model = Sequential()
    # 建立Embedding層，將數字list變成向量list(將文字應設為多維度空間的向量，語意類似的向量在該空間的距離也較近)
    # 我們自行決定將它設為32維資料
    # 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    model.add(Embedding(input_dim=input_dim, output_dim=32, input_length = 100))
    model.add(Flatten())
    model.add(Dropout(0.35))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # 編譯網路
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # 開始訓練
    model.fit(x_train,
              y_train,
              batch_size = 100,
              epochs = 10,
              verbose = 2,
              validation_split = 0.2)
    
    # 衡量模型表現
    scores = model.evaluate(x_test, y_test, verbose = 1)
    print(scores)


def docVector(train_text, y_train, test_text, y_test):
    def read_corpus(documents):
        for index, document in enumerate(documents):
            preprocess = gensim.utils.simple_preprocess(document)
            yield gensim.models.doc2vec.TaggedDocument(preprocess, [index])
    
    train_corpus = list(read_corpus(train_text))
#    test_corpus = list(read_corpus(test_text))
    
    # 產生句向量
    docVector_model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=20, epochs=40)
    print("=== 建立 Doc2Vec 物件 ===")
    docVector_model.build_vocab(train_corpus)
    print("=== 字典創建完成 ===")
    docVector_model.train(train_corpus, total_examples=docVector_model.corpus_count, epochs=docVector_model.epochs)
    print("=== 句向量訓練完成 ===")
    
    x_train = []    
    for index, seq in enumerate(train_text):
        s = gensim.utils.simple_preprocess(seq)    
        x_train.append(docVector_model.infer_vector(s))
        if (index + 1) % 500 == 0:
            print("進度：第 %d 句" % (index + 1))
    
    x_test = []    
    for index, seq in enumerate(test_text):
        s = gensim.utils.simple_preprocess(seq)    
        x_test.append(docVector_model.infer_vector(s))
        if (index + 1) % 500 == 0:
            print("進度：第 %d 句" % (index + 1))
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    print("x_train.shape:", x_train.shape, ", x_test.shape:", x_test.shape)
    
    # 建立網路
    model = Sequential()
    model.add(Dense(input_dim = 512, units = 256, activation = "relu"))
    model.add(Dropout(0.35))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    # 編譯網路
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    
    # 開始訓練
    model.fit(x_train,
              y_train,
              batch_size = 100,
              epochs = 10,
              verbose = 2,
              validation_split = 0.2)
    
    # 衡量模型表現
    scores = model.evaluate(x_test, y_test, verbose = 1)
    print(scores)


# kerasEmbedding accuracy = 0.82084
kerasEmbedding(train_text, y_train, test_text, y_test)

# docVector accuracy = 0.83544
docVector(train_text, y_train, test_text, y_test)
'''句向量表現得比 Embedding 好~~~'''


'''Keras Embedding'''
# 1. 決定詞頻低標 (出現次數應大於低標才會被納入 token)
# num_words：None或整数，处理的最大单词数量。若被设置为整数，则分词器将被限制为待处理数据集中最常见的num_words个单词
# 但計算時全部都會需要計算，因此 token.word_index token.word_counts token.index_word 等，都是包含所有詞的字典
token = Tokenizer(num_words = 20)
# 2. 形塑 token
token.fit_on_texts(train_text)
# token.word_index >>> 字典 key:word; value:index
# token.word_counts >>> 字典 key:word; value:counts
# token.index_word >>> 字典 key:index; value:word
number = 21
for index in range(1, number + 1):
    try:
        word = token.index_word[index]
        counts = token.word_counts[word]
        print("%d\t%s\t%d" % (index, word, counts))
    except:
        print("no index %d" % index)
# 1       the     330431
# 2       and     160190
# 3       a       159715
# 4       of      143511
# 5       to      132341
# 6       is      103717
# 7       in      92723
# 8       it      77721
# 9       i       77682
# 10      this    74921
# 11      that    67177
# 12      was     47377
# 13      as      44789
# 14      movie   43467
# 15      with    43218
# 16      for     43098
# 17      but     40888
# 18      film    38565
# 19      on      33814
# 20      you     30424
# 21      not     30077

x_test_seq = token.texts_to_sequences(test_text)
print(len(x_test_seq))
# print(test_text[0])
# print(x_test_seq[0])
print(len(test_text[0]))
print(len(x_test_seq[0]))
# endregion
