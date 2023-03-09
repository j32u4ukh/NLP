from gensim.models import word2vec
import logging
import jieba


def sentenceToWord(inputFile, outputFile):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    jieba.set_dictionary('jieba_dict/dict.txt.big')
    stopword_set = set()
    with open('jieba_dict/stopwords.txt', 'r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword = stopwords.readline()
            stopword_set.add(stopword.strip('\n'))
    
    output = open('./data/' + outputFile, 'w', encoding='utf-8')
    with open('./data/' + inputFile, 'r', encoding='utf-8') as content:
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 100 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()
    return outputFile


def wordToVec(outputFile, dim, modelName):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    sentences = word2vec.LineSentence("./data/" + outputFile)
    model = word2vec.Word2Vec(sentences, size=dim, min_count=2)
    
    
#    class gensim.models.word2vec.Word2Vec(sentences=None, size=100, 
#                                          alpha=0.025, window=5, 
#                                          min_count=5, 
#                                          max_vocab_size=None, 
#                                          sample=0.001, seed=1, 
#                                          workers=3, min_alpha=0.0001, 
#                                          sg=0, hs=0, negative=5, 
#                                          cbow_mean=1, 
#                                          hashfxn=<built-in function hash>, 
#                                          iter=5, null_word=0, 
#                                          trim_rule=None, sorted_vocab=1, 
#                                          batch_words=10000)
#    sentences:當然了，這是要訓練的句子集，沒有他就不用跑了
#    size:這表示的是訓練出的詞向量會有幾維
#    alpha:機器學習中的學習率，這東西會逐漸收斂到 min_alpha
#    sg:這個不是三言兩語能說完的，sg=1表示採用skip-gram,sg=0 表示採用cbow
#    window:還記得孔乙己的例子嗎？能往左往右看幾個字的意思
#    workers:執行緒數目，除非電腦不錯，不然建議別超過 4
#    min_count:若這個詞出現的次數小於min_count，那他就不會被視為訓練對象

    # 保存模型，供日後使用
    model.save("./SaveModel/" + modelName + ".model")
    return "./SaveModel/" + modelName + ".model"


outputFile = sentenceToWord("mayday.txt", "mayday_words.txt")
modelName = wordToVec(outputFile, 10, "MayDayVec")
model = word2vec.Word2Vec.load(modelName)

print(model['巨大'])

for i in model.wv.vocab.keys():
    print(i)

print(model.similarity("狂放", "貪玩"))
print(model.most_similar("志明"))
