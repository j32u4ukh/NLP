import os
import re
import tarfile  # 用於解壓縮
import time
import urllib.request

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import numpy as np

# download dataset from IMDb
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
file = "data/aclImdb_v1.tar.gz"
if not os.path.isfile(file):
    result = urllib.request.urlretrieve(url, file)
    print("Download: ", result)

if not os.path.exists("data/aclImdb"):  # 判斷解壓縮目錄是否存在
    tfile = tarfile.open(file, 'r:gz')  # 開啟解壓縮檔
    result = tfile.extractall('data/')  # 解壓縮檔案至data目錄


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
    file_list = []
    
    # get all file under positive_path
    # ex: "data/aclImdb/train/pos/"
    # ex: "data/aclImdb/test/pos/"    
    positive_path = path + filetype + "/pos/"
    files = os.listdir(positive_path)
    for f in files:
        file_list += [positive_path + f]
        if len(file_list) % 1000 == 0:
            print("read %s file, files number:%5d" % (filetype, len(file_list)))
    n_pos = len(files)
    
    # get all file under negative_path
    # ex: "data/aclImdb/train/neg/"
    # ex: "data/aclImdb/test/neg/"
    negative_path = path + filetype + "/neg/"
    files = os.listdir(negative_path)
    for f in files:
        file_list += [negative_path + f]
        if len(file_list) % 1000 == 0:
            print("read %s file, files number:%5d" % (filetype, len(file_list)))
    n_neg = len(files)
    
    all_labels = binaryLabels(n_pos, n_neg)
    all_texts = []
    
    for file in file_list:
        with open(file, encoding = 'utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    
    print("read:", filetype, ", files amounts: ", len(file_list))
    return all_texts, all_labels


def originLabels(n_pos, n_neg):
    # 利用 numpy 產生 label 資料
    # 原方法：all_labels = ([1] * n_pos + [0] * n_neg)
    return ([1] * n_pos + [0] * n_neg)


def binaryLabels(n_pos, n_neg):
    one = np.ones((n_pos,))
    zero = np.zeros((n_neg,))    
    return np.r_[one, zero]


n_pos, n_neg = 5000000, 5000000
start = time.time()
originLabels(n_pos, n_neg)
print("originLabels Cost time:", time.time() - start)
start = time.time()
binaryLabels(n_pos, n_neg)
print("binaryLabels Cost time:", time.time() - start)

train_text, y_train = read_files("train")
test_text, y_test = read_files("test")
print(type(train_text))
print(len(train_text))
# train_text.shape = (25000, ?) 25000個文本，每個文本的長度不一定

# 篩選出現次數最多的 2000字
token = Tokenizer(num_words = 2000)
token.fit_on_texts(train_text)
print(token.word_index)
print(type(token))

#  將影評文字轉為數字list
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
print(type(x_train_seq))
# <class 'list'>
seq = np.array(x_train_seq)
# np.array(x_train_seq) >> 只把整個 list 變 np.array 內部還是 list，因此 seq.shape 只有 (25000,)，實際上是 2維資料
print(seq.shape)
# (25000,)

seq0 = seq[0]
print(type(seq0))
# <class 'list'>
print(len(seq0))
# 106
print(seq0)
# [308, 6, 3, 1068, 208, 8, 29, 1, 168, 54, 13, 45, 81, 40, 391, 109, 137, 13,
# 57, 149, 7, 1, 481, 68, 5, 260, 11, 6, 72, 5, 631, 70, 6, 1, 5, 1, 1530, 33, 
# 66, 63, 204, 139, 64, 1229, 1, 4, 1, 222, 899, 28, 68, 4, 1, 9, 693, 2, 64, 
# 1530, 50, 9, 215, 1, 386, 7, 59, 3, 1470, 798, 5, 176, 1, 391, 9, 1235, 29, 
# 308, 3, 352, 343, 142, 129, 5, 27, 4, 125, 1470, 5, 308, 9, 532, 11, 107, 
# 1466, 4, 57, 554, 100, 11, 308, 6, 226, 47, 3, 11, 8, 214]

# 讓數字list長度都變100
x_train = sequence.pad_sequences(x_train_seq, maxlen = 100)
x_test = sequence.pad_sequences(x_test_seq, maxlen = 100)

# <class 'numpy.ndarray'>
print(type(x_train))
# (25000, 100)
print(x_train.shape)

model = Sequential()
# 建立Embedding層，將數字list變成向量list(將文字應設為多維度空間的向量，語意類似的向量在該空間的距離也較近)
# 我們自行決定將它設為32維資料
# 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# embedding_1 (Embedding)
model.add(Embedding(input_dim = 2000, output_dim = 32, input_length = 100))
# dropout_1 (Dropout)
model.add(Dropout(0.2))  
# flatten_1 (Flatten)
model.add(Flatten())
# dense_1 (Dense)
model.add(Dense(units = 256, activation = 'relu'))
# dropout_2 (Dropout)
model.add(Dropout(0.35))
# dense_2 (Dense)
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

train_history = model.fit(x_train,
                          y_train,
                          batch_size = 100,
                          epochs = 10, 
                          verbose = 2,
                          validation_split = 0.2)

scores = model.evaluate(x_test, y_test, verbose = 1)
# accuracy:scores[1] = 0.80988
print(scores)

predict = model.predict_classes(x_test)
predict_classes = predict.reshape(-1)

SentimentDict = {1: '正面的',
                 0: '負面的'}


def displayTestSentiment(i):
    print(x_test[i])
    print("label: ", SentimentDict[y_test[i]], ", predict: ", SentimentDict[predict_classes[i]])


# 儲存訓練過的模型中的weight >>> 此種方法需要自己在搭建相同架構的網路，缺點是較麻煩，優點是可進行 transfer learning(遷移學習)
# model.save_weights("SaveModel/KerasImdbMlp.h5")

# 儲存訓練過的模型
model.save("SaveModel/KerasModel-Imdb.h5")

# 匯入保存好的模型

try:
    model = load_model('SaveModel/KerasModel-Imdb.h5')
    print("載入模型成功，繼續訓練模型")
except:
    print("載入模型失敗，開始訓練一個新的模型")


'''測試'''

input_text = '''Remakes of original films always make me cringe - you can never beat an original, even if it is a bad one.

I'm not fully against this remake but I feel with Disney and their budget they could have done and made it a lot better.

My main gripe were the accents - fairly dodgey especially in the songs. Some can pull it off but Ewan Mcgregor isn't remotely French in this film, Emma Watsons singing has been extremely enhanced and takes away from her natural voice.

As a story, it's tried and tested through the original - even if there is a plot flaw (not saying there is one) it is a Disney classic and the charm of that is there overpowering the flaws.

Could have done better though Disney...'''

'''翻譯
原始電影的翻拍總讓我感到畏縮 - 你永遠不會打敗原作，即使它是一個糟糕的原創。
我並不完全反對這個翻拍，但我覺得他們可以做迪斯尼和他們的預算並且做得更好。
我的主要抱怨是口音 - 特別是在歌曲中相當眩暈。
有些人可以把它拉下來，但Ewan Mcgregor在這部電影中不是法國人，Emma Watsons的歌聲極度增強，並且不再是她的自然聲音。
作為一個故事，它通過原始的嘗試和測試 - 即使有一個情節缺陷（不是說有一個）它是迪斯尼經典，其魅力在於壓倒了這些缺陷。
迪士尼可以做得更好......
'''

input_seq = token.texts_to_sequences([input_text])
# length: 114
print("length:", len(input_seq[0]))
# input_seq[0]:只有一筆資料，但他本身是 2維資料
print(input_seq[0])

# 讓數字list長度都變100
pad_input_seq = sequence.pad_sequences(input_seq, maxlen = 100)

print(len(pad_input_seq[0]))
print(pad_input_seq[0])

predict_result = model.predict_classes(pad_input_seq)
print(predict_result)        # [[1]]
print(predict_result[0][0])  # 1

predict_score = model.predict(pad_input_seq)
# [[0.99867153]]
print(predict_score)


def predictReview(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen = 100)
    predict_result = model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])


predictReview('''The only reason why this movie got anyone to go to it is because of Emma Watson. Which is a bit funny 
because she CAN'T sing. Auto tune was her savior. Really pathetic of Disney. They don't care about quality anymore, 
just sales.''')
predictReview('''Emma Watson was amazing in it and it really did bring back memories from when I watched the cartoon 
as a child. My niece and nephews, all under the age of 10, all love the movie and watch it at least Twice a week.''')


'''查看 embedding 層'''
# model.input = <tf.Tensor 'embedding_1_input_1:0' shape=(?, 100) dtype=float32>
# model.get_layer('embedding_1').output =
# <tf.Tensor 'embedding_1_1/embedding_lookup:0' shape=(?, 100, 32) dtype=float32>
embedding_1 = Model(inputs=model.input, outputs=model.get_layer('embedding_1').output)
embedding_1_output = embedding_1.predict(pad_input_seq)

# <class 'numpy.ndarray'>
print(type(embedding_1_output))
# (1, 100, 32)
print(embedding_1_output.shape)
print(embedding_1_output[0, 0].shape)
print(embedding_1_output[0, 0])
print(model.input)


def hiddenLayerOutput(layer_name):
    hidden_layer = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return hidden_layer.predict(pad_input_seq)


embedding_1 = hiddenLayerOutput("embedding_1")
dropout_1 = hiddenLayerOutput("dropout_1")
flatten_1 = hiddenLayerOutput("flatten_1")
dense_1 = hiddenLayerOutput("dense_1")
dropout_2 = hiddenLayerOutput("dropout_2")
dense_2 = hiddenLayerOutput("dense_2")

print(embedding_1.shape)
print(dropout_1.shape)
print(flatten_1.shape)
print(dense_1.shape)
print(dropout_2.shape)
print(dense_2.shape)

print("embedding_1")
print(model.get_layer('embedding_1').output)
# Tensor("embedding_1_1/embedding_lookup:0", shape=(?, 100, 32), dtype=float32)
print("dropout_1")
print(model.get_layer('dropout_1').output)
# Tensor("dropout_1_1/cond/Merge:0", shape=(?, 100, 32), dtype=float32)
print("flatten_1")
print(model.get_layer('flatten_1').output)
# Tensor("flatten_1_1/Reshape:0", shape=(?, ?), dtype=float32)
print("dense_1")
print(model.get_layer('dense_1').output)
# Tensor("dense_1_1/Relu:0", shape=(?, 256), dtype=float32)
print("dropout_2")
print(model.get_layer('dropout_2').output)
# Tensor("dropout_2_1/cond/Merge:0", shape=(?, 256), dtype=float32)
print("dense_2")
print(model.get_layer('dense_2').output)
# Tensor("dense_2_1/Sigmoid:0", shape=(?, 1), dtype=float32)

print(model.layers)
# [<keras.layers.embeddings.Embedding at 0x264b3255a20>,
# <keras.layers.core.Dropout at 0x264b32554e0>,
# <keras.layers.core.Flatten at 0x264b39bd128>,
# <keras.layers.core.Dense at 0x264b39bd2e8>,
# <keras.layers.core.Dropout at 0x264b3a09400>,
# <keras.layers.core.Dense at 0x264b3a094e0>]

print(model.get_layer('embedding_1'))
# <keras.layers.embeddings.Embedding at 0x264b3255a20>

# 使用詞彙量較多的字典來進行預測，以提升預測準確率
token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)
#  將影評文字轉為數字list
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
# 讓數字list長度都變380
x_train = sequence.pad_sequences(x_train_seq, maxlen = 380)
x_test = sequence.pad_sequences(x_test_seq, maxlen = 380)

# 建立模型
model = Sequential()
model.add(Embedding(input_dim=3800, 
                    output_dim=32, 
                    input_length=380))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())

train_history = model.fit(x_train, 
                          y_train, 
                          batch_size=100,
                          epochs=10,
                          verbose=2,
                          validation_split=0.2)

scores = model.evaluate(x_test, y_test, verbose=1)
# accuracy: scores[1] = 0.83244
print(scores)

# RNN Learning


# 建立模型
model = Sequential()
model.add(Embedding(input_dim=3800,
                    output_dim=32,
                    input_length=380))
model.add(Dropout(0.35))
model.add(SimpleRNN(units=16))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

train_history = model.fit(x_train, 
                          y_train, 
                          batch_size=100,
                          epochs=10,
                          verbose=2,
                          validation_split=0.2)

scores = model.evaluate(x_test, y_test, verbose=1)
# accuracy:scores[1] = 0.8428
print(scores)

# LSTM Learning


# 建立模型
model = Sequential()
model.add(Embedding(input_dim=3800, output_dim=32, input_length=380))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

train_history = model.fit(x_train, 
                          y_train, 
                          batch_size=100,
                          epochs=10,
                          verbose=2,
                          validation_split=0.2)

scores = model.evaluate(x_test, y_test, verbose=1)
# accuracy:scores[1] = 0.8484
print(scores)

# 建立模型
model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))

# model.input
# <tf.Tensor 'embedding_1_input_1:0' shape=(?, 100) dtype=float32>
model.add(Embedding(input_dim=3800, output_dim=32, input_length=380))
# model.get_layer('embedding_1').output
# <tf.Tensor 'embedding_1_1/embedding_lookup:0' shape=(?, 100, 32) dtype=float32>

model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
