import logging
import os

from gensim import models
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
import jieba
from opencc import OpenCC


root = "Crescent/data/NLP/"


def pathCheck():
    _path = os.path.join(root, "zhwiki-20190901-pages-articles.xml.bz2")
    print("path:", _path)


# 簡體字 tokens 轉 繁體字 tokens
def simpleToken2TraditionalToken():
    """可利用 sys.argv，從 cmd 傳入檔案路徑，或直接賦予"""
    # if len(sys.argv) != 2:
    #     print("Usage: python3 " + sys.argv[0] + " wiki_data_path")
    #     exit()
    # sys.argv[1]
    _path = os.path.join(root, "zhwiki-20190901-pages-articles.xml.bz2")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    _wiki_corpus = WikiCorpus(_path, dictionary={})

    # OpenCC 用於中文簡轉繁
    _cc = OpenCC('s2t')
    _output_path = os.path.join(root, "wiki_texts.txt")
    with open(_output_path, 'w', encoding='utf-8') as output:
        # _wiki_corpus.get_texts()：迭代每一篇文章，它所回傳的是一個 tokens list
        for _num, _text in enumerate(_wiki_corpus.get_texts()):
            # _cc.convert：將簡體字轉繁體字
            output.write(_cc.convert(' '.join(_text)) + '\n')
            if _num % 10000 == 0:
                logging.info("已處理 {} 篇文章".format(_num + 1))
        logging.info("end @ 已處理 {} 篇文章".format(_num + 1))


def jiebaTest(_sentence="颱風天就是要泛舟啊不然要幹嘛"):
    jieba.set_dictionary(os.path.join(root, 'jieba_dict/dict.txt.big'))
    seg_list = jieba.cut(_sentence, cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))


def mulitOpenWithTest():
    _input_path = os.path.join(root, "jieba_dict/stopwords.txt")
    _output_path = os.path.join(root, "mulitOpenWithTest.txt")
    with open(_input_path, "r", encoding='utf-8') as _input, open(_output_path, "w", encoding='utf-8') as _output:
        for _num, _line in enumerate(_input):
            _content = "{}\t{}\t{}\n".format(_num, _line.strip("\n"), "=====")
            _output.write(_content)

            if (_num + 1) % 1000 == 0:
                print("line {}".format(_num + 1))
        print("end @ line {}".format(_num + 1))


# 將維基百科內容進行斷詞
def participleByJieba():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # dict.txt.big 和 stopwords.txt皆需自行下載，並非包含於套件當中
    # jieba custom setting.
    jieba.set_dictionary(os.path.join(root, 'jieba_dict/dict.txt.big'))

    # load stopwords set
    _stopwords_path = os.path.join(root, "jieba_dict/stopwords.txt")
    _stopword_set = set()
    with open(_stopwords_path, 'r', encoding='utf-8') as _stopwords:
        for _num, _line in enumerate(_stopwords):
            _stopword_set.add(_line.strip('\n'))

    _input_path = os.path.join(root, "wiki_texts.txt")
    _output_path = os.path.join(root, "wiki_seg.txt")
    with open(_input_path, 'r', encoding='utf-8') as _input, open(_output_path, 'w', encoding='utf-8') as _output:
        for _num, _line in enumerate(_input):
            _line = _line.strip('\n')
            _words = jieba.cut(_line, cut_all=False)
            for _word in _words:
                if _word not in _stopword_set:
                    _output.write(_word + ' ')
            _output.write('\n')

            if (_num + 1) % 1000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (_num + 1))

        logging.info("end @ 已完成前 %d 行的斷詞" % (_num + 1))


def wordToVec():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    _path = os.path.join(root, "wiki_seg.txt")
    _sentences = word2vec.LineSentence(_path)
    _model = word2vec.Word2Vec(_sentences, size=250)

    # 保存模型，供日後使用
    _model.save("Crescent/SaveModel/word2vec.model")

    # 模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")
    # ex: model = models.Word2Vec.load('SaveModel/word2vec.model')


def demo():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    model = models.Word2Vec.load('Crescent/SaveModel/word2vec.model')
    print("提供 3 種測試模式\n")
    print("輸入一個詞，則去尋找前10個該詞的相似詞")
    print("輸入兩個詞，則去計算兩個詞的餘弦相似度")
    print("輸入三個詞，進行類比推理")

    #    查詢詞向量
    #    model['computer']
    #    array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

    while True:
        try:
            query = input(">>>")
            if query == "exit()":
                print("exit...")
                break

            q_list = query.split()
            if len(q_list) == 1:
                print("相似詞前 10 排序")
                res = model.most_similar(q_list[0], topn=10)
                for item in res:
                    print(item[0] + "\t" + str(item[1]))
            elif len(q_list) == 2:
                print("計算 Cosine 相似度")
                res = model.similarity(q_list[0], q_list[1])
                print(res)
            else:
                print("%s之於%s，如%s之於" % (q_list[0], q_list[2], q_list[1]))
                res = model.most_similar([q_list[0], q_list[1]], [q_list[2]], topn=10)
                for item in res:
                    print(item[0] + "," + str(item[1]))

            print("----------------------------")
        except Exception as e:
            print(repr(e))


if __name__ == "__main__":
    # pathCheck()
    # simpleToken2TraditionalToken()
    # jiebaTest()
    # mulitOpenWithTest()
    # participleByJieba()
    # wordToVec()
    demo()



