import os

import jieba
from gensim import corpora, models, similarities
from gensim.corpora import MmCorpus

# https://medium.com/pyladies-taiwan/%E4%BB%A5-jieba-%E8%88%87-gensim-%E6%8E%A2%E7%B4%A2%E6%96%87%E6%9C%AC%E4%B8%BB%E9%A1%8C-%E4%BA%94%E6%9C%88%E5%A4%A9%E4%BA%BA%E7%94%9F%E7%84%A1%E9%99%90%E5%85%AC%E5%8F%B8%E6%AD%8C%E8%A9%9E%E5%88%86%E6%9E%90-ii-fdf5d3708662

jieba.set_dictionary(os.path.join("Crescent/data/NLP", 'jieba_dict/dict.txt.big'))

# load stopwords set
stopwords_path = os.path.join("Crescent/data/NLP", "jieba_dict/stopwords.txt")
stopword_set = set()
with open(stopwords_path, 'r', encoding='utf-8') as stopwords:
    for num, line in enumerate(stopwords):
        stopword_set.add(line.strip('\n'))

root = "Crescent/data/news"
texts = []
files = os.listdir(root)

for file in files:
    if file != "key_word":
        path = os.path.join(root, file)
        news_files = os.listdir(path)
        for news_file in news_files:
            file_path = os.path.join(path, news_file)
            with open(file_path, "r", encoding="utf-8") as input_file:
                text = []
                # region one file
                for line in input_file:
                    line = line.strip('\n')
                    words = jieba.cut(line, cut_all=False)
                    for word in words:
                        if word not in stopword_set:
                            text.append(word)
                # endregion
                texts.append(text)

# print(texts)
# 建立本次文檔的語料庫(字典)
dictionary = corpora.Dictionary(texts)
# 字典摘要
print(dictionary)

# 字典存檔
dictionary.save("Crescent/data/NLP/stock_news_test.dict")

for word, index in dictionary.token2id.items():
    print(word + " id:" + str(index))

    limit_index = 200
    if index < limit_index:
        continue
    if index == limit_index + 10:
        break

# 將corpus序列化
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize("Crescent/data/NLP/stock_news_test.mm", corpus)

# 載入語料庫
dict_path = "Crescent/data/NLP/stock_news_test.dict"
mm_path = "Crescent/data/NLP/stock_news_test.mm"
if os.path.exists(dict_path):
    dictionary = corpora.Dictionary.load(dict_path)
    corpus = corpora.MmCorpus(mm_path)
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

# 創建 tfidf model
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# 創建 LSI model 潛在語義索引
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
corpus_lsi = lsi[corpus_tfidf]  # LSI潛在語義索引
lsi_path = "Crescent/data/NLP/stock_news_test.lsi"
lsi.save(lsi_path)
lsi_mm_path = "Crescent/data/NLP/lsi_stock_news_test.mm"
corpora.MmCorpus.serialize(lsi_mm_path, corpus_lsi)
print("LSI topics:")
lsi.print_topics(2)

# 「好好」這首歌詞在各主題的佔比
# 基於tf-idf-> lsi 的文本相似度分析
doc = """想 把 你 寫成 一首歌 想養 一隻 貓 想要 回到 每個 場景 撥慢 每 隻 錶 我們 在 小孩 和 大人 的 轉角 蓋 一座 城堡 我們 
好好 好 到 瘋 掉 像 找回 失散多年 雙胞 生命 再長 不過 煙火 落下 了 眼角 世界 再大 不過 你 我 凝視 的 微笑 在 所有 流逝 風景 
與 人群 中 你 對 我 最好 一切 好好 是否 太好 沒有 人 知道 你 和 我 背著 空空 的 書包 逃出 名為 日常 的 監牢 忘 了 要 
長大 忘 了 要 變老 忘 了 時間 有腳 最 安靜 的 時刻 回憶 總是 最 喧囂 最 喧囂 的 狂歡 寂寞 包圍 著 孤島 還以 為 馴服 想念 
能 陪伴 我 像 一隻 家貓 它 就 窩 在 沙發 一角 卻 不肯 睡著 你 和 我 曾 有 滿滿的 羽毛 跳 著名 為 青春 的 舞蹈 不 
知道 未來 不 知道 煩惱 不知 那些 日子 會 是 那麼 少 時間 的 電影 結局 才 知道 原來 大人 已 沒有 童謠 最後 的 叮嚀 
最後 的 擁抱 我們 紅著 眼笑 我們 都 要 把 自己 照顧 好 好 到 遺憾 無法 打擾 好好 的 生活 好好 的 變老 好好 假裝 我 已經 
把 你 忘掉 """

# 把字典中的語料庫轉為詞包
vec_bow = dictionary.doc2bow(doc.split())
# 用前面建好的 lsi 去計算這一篇歌詞
vec_lsi = lsi[vec_bow]
print(vec_lsi)

# 建立索引
index = similarities.MatrixSimilarity(lsi[corpus])
# index.save("lyrics/lyrics_mayday.index")

# 相似度
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)
