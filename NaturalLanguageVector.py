import gc
import json
import os

from gensim.models import word2vec
import jieba
import numpy as np
import psutil
import tensorflow as tf


# 自然語言處理(Natural Language Processing, NLP)
# 自然語言向量(Natural Language Vector, NLV)
class NLV:
    def __init__(self):
        self.root = "Crescent/data/"
        self.jieba_path = os.path.join(self.root, "NLP/jieba_dict")

        jieba.set_dictionary(os.path.join(self.jieba_path, 'dict.txt.big'))
        stopwords_path = os.path.join(self.jieba_path, "stopwords.txt")
        self.stopword_set = set()
        with open(stopwords_path, 'r', encoding='utf-8') as stopwords:
            for num, line in enumerate(stopwords):
                self.stopword_set.add(line.strip('\n'))


class WordVector(NLV):
    def __init__(self, _path=None):
        super().__init__()
        self.model = None
        if _path is None:
            self.loadModel()
        else:
            self.loadModel(_path)

    # 目前詞和字向量混用，之後希望可以區分
    def loadModel(self, _path='Crescent/SaveModel/word2vec.model'):
        self.model = word2vec.Word2Vec.load(_path)

    def sentence2Vector(self, _sentence):
        _temp_result = jieba.cut(_sentence, cut_all=False)
        _result = [_temp for _temp in _temp_result if _temp not in self.stopword_set]

        _sentence_vector = []
        for _res in _result:
            try:
                _sentence_vector.append(self.model.wv[_res])
            except KeyError:
                # print("word {} not in vocabulary".format(_res))
                continue

        return np.array(_sentence_vector)


class SentenceVector(NLV):
    def __init__(self, _input_size=250, _output_size=250, _step=10, _cell_size=32, _lr=0.02, _epoch=100):
        super().__init__()
        # Hyper Parameters
        self.INPUT_SIZE = _input_size
        self.STEP = _step
        self.OUTPUT_SIZE = _output_size

        # placeholder
        self.tf_x = None
        self.tf_y = None

    def buildModel(self):
        with tf.variable_scope("placeholder", reuse=tf.AUTO_REUSE):
            self.tf_x = tf.placeholder(tf.float32, [None, self.STEP, self.INPUT_SIZE])
            self.tf_y = tf.placeholder(tf.float32, [None, self.STEP, self.OUTPUT_SIZE])

        with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
            # RNN   CELL_SIZE=32
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.CELL_SIZE)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)  # very first hidden state

            #                                batch=1   TIME_STEP=10   CELL_SIZE=32
            # outputs.shape TensorShape([Dimension(1), Dimension(10), Dimension(32)])
            outputs, self.final_state = tf.nn.dynamic_rnn(
                rnn_cell,  # cell you have chosen
                self.tf_x,  # input
                initial_state=self.init_state,  # the initial hidden state
                time_major=False  # False: (batch, time step, input); True: (time step, batch, input)
            )

        with tf.variable_scope("dense", reuse=tf.AUTO_REUSE):
            # shape (1, 10, 32) >> (1*10, 32) for 矩陣運算
            # (batch * TIME_STEP=1*10, CELL_SIZE=32)
            # reshape 3D output to 2D for fully connected layer
            outs2D = tf.reshape(outputs, [-1, self.CELL_SIZE])
            # net_outs2D.shape = TensorShape([Dimension(10), Dimension(1)])
            net_outs2D = tf.layers.dense(outs2D, self.OUTPUT_SIZE)
            # (batch=1, TIME_STEP=10, INPUT_SIZE=1)
            self.outs = tf.reshape(net_outs2D, [-1, self.TIME_STEP, self.OUTPUT_SIZE])  # reshape back to 3D

        with tf.variable_scope("loss_and_train", reuse=tf.AUTO_REUSE):
            # compute cost
            self.loss = tf.losses.mean_squared_error(labels=self.tf_y,
                                                     predictions=self.outs)
            self.train = tf.train.AdamOptimizer(self.LR).minimize(self.loss)


class WikiVector(NLV):
    def __init__(self):
        super().__init__()
        # 建立字典：key 第幾篇；value 以詞向量呈現的文章
        self.wiki = {}
        # load data
        self.dataset = None
        # 原始文本路徑
        self.wiki_texts = os.path.join(self.root, "NLP/wiki_texts.txt")
        # 字向量文件路徑
        self.wiki_word_vector = os.path.join(self.root, "NLP/wiki_word_vector")

    def wordVector(self, _word_vector):
        _round = 0
        with open(self.wiki_texts, "r", encoding="utf-8") as _wiki:
            # 依序取得每篇文章(345589 篇)
            for _index, _content in enumerate(_wiki):
                # _start = time.time()

                # 分解成詞句向量後的文章 的 容器
                _content_container = []
                # 將文章分為一句一句
                _lines = _content.split(" ")
                # 依序取得每一個句子
                for _sub_index, _line in enumerate(_lines):
                    # 以詞向量形式呈現"句向量" shape=(N, 250)
                    _sentence_vector = _word_vector.sentence2Vector(_line)
                    if len(_sentence_vector) > 0:
                        # 保留原始文具和分解後向量的字典
                        dictionary = {"raw": _line,
                                      "sentence_vector": _sentence_vector.tolist()}
                        _content_container.append(dictionary)

                # print("content {} size: {}".format(_index, getMemorySize(_content_container)))
                self.wiki[_index] = _content_container
                # print("cost time: {}".format(time.time() - _start))
                # print("wiki size: {}".format(getMemorySize(self.wiki)))
                del _content_container
                gc.collect()

                # region 記憶體管理
                # 21篇 = 1024byte = 1K >> process cost: 246M
                # 340篇 = 10240byte = 10K >> process cost: 211M
                if self.wiki.__sizeof__() >= 1024 * 10:
                    _round += 1
                    _path = os.path.join(self.wiki_word_vector, "{}.txt".format(_round))
                    with open(_path, "w", encoding="utf-8") as output:
                        json_string = json.dumps(self.wiki)
                        output.write(json_string)

                    del self.wiki
                    gc.collect()
                    self.wiki = {}
                    # 20190922 06:53
                    # round:227, index:77633, wiki size: 216, memory cost: 0G 39M 12K 0btye
                    yield "round:{}, index:{}, wiki size: {}".format(_round, _index, self.wiki.__sizeof__())
                # endregion

                yield "round:{}, index:{}".format(_round, _index)

    def load(self, _file_name):
        _path = os.path.join(self.wiki_word_vector, _file_name)
        with open(_path, "r", encoding="utf-8") as _file:
            _dataset = _file.read()
            self.dataset = json.loads(_dataset)

    def readData(self):
        for _index, _document in enumerate(self.dataset.values()):
            for _sub_index, _line in enumerate(_document):
                _raw = _line["raw"]
                _vector = _line["sentence_vector"]
                yield _index, _sub_index, _raw, _vector

    @staticmethod
    def checkDocument(_document):
        for _index, _line in enumerate(_document):
            _raw = _line["raw"]
            _vector = _line["sentence_vector"]
            yield _index, _raw, _vector


def wordVectorTest():
    _wv = WordVector()
    _sentence = "馮克斯是模仿果實能力者"
    _sentence_vector = _wv.sentence2Vector(_sentence)
    print("sentence_vector")
    print(_sentence_vector)


def fromWikiToVector():
    _wv = WordVector()
    _wiki_vector = WikiVector()

    # 透過詞向量，將維基百科轉為向量
    # 20190922 06:53
    # round:227, index:77633, wiki size: 216, memory cost: 0G 39M 12K 0btye
    for i in _wiki_vector.wordVector(_wv):
        print("{}, memory cost: {}".format(i, getSize(psutil.Process(os.getpid()).memory_info().rss)))


def getMemorySize(_variable):
    _size = _variable.__sizeof__()
    return getSize(_size)


def getSize(_size):
    _byte = _size % 1024
    _size = int(_size / 1024)
    _kb = _size % 1024
    _size = int(_size / 1024)
    _mb = _size % 1024
    _size = int(_size / 1024)
    # _gb = _size % 1024
    _gb = 0
    return "{}G {}M {}K {}btye".format(_gb, _mb, _kb, _byte)


def sizeTest1():
    _wv = WordVector()
    print("wv size: {}".format(_wv.__sizeof__()))
    test = "測試"
    vector = _wv.sentence2Vector(test)
    print("vector shape: {}".format(vector.shape))
    print("test size: {}".format(test.__sizeof__()))
    print("vector size: {}".format(vector.__sizeof__()))


def loadWikiDataset(_dataset_number):
    _wv = WordVector()
    _wiki_vector = WikiVector()

    _file_name = "{}.txt".format(_dataset_number)

    _wiki_vector.load(_file_name)

    _np_vector = None
    for _index, _sub_index, _raw, _vetor in _wiki_vector.readData():
        print("index:", _index)
        print("sub_index:", _sub_index)
        print("raw:", _raw)
        print("vetor:", _vetor)
        _np_vector = np.array(_vetor)
        break

    print("np_vector")
    print(_np_vector.shape)  # (N, 250)
    print(_np_vector)


def newsTitleTest():
    _news1 = "台泥董座專訪／張安平 注入環保新美學"
    _news2 = "網路、手機普及化 學者：通勤時間應納入工時 - 財經 - 中時電子報"
    _wv = WordVector()
    _vector1 = _wv.sentence2Vector(_news1)
    _vector2 = _wv.sentence2Vector(_news2)

    return _vector1, _vector2


if __name__ == "__main__":
    vector1, vector2 = newsTitleTest()
    jieba_list = jieba.cut("台泥董座專訪／張安平 注入環保新美學")
    for j in jieba_list:
        print(j, end=", ")
    print()
    jieba_list = jieba.cut("網路、手機普及化 學者：通勤時間應納入工時 - 財經 - 中時電子報")
    for j in jieba_list:
        print(j, end=", ")
