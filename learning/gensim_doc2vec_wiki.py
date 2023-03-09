import multiprocessing
import os
import time
from pprint import pprint

import jieba
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html

root = "Crescent/data/NLP/"


# region WikiDoc2Vec
class WikiDoc2Vec:
    def __init__(self):
        pass
# endregion


# region TaggedWikiDocument
class TaggedWikiDocument:
    def __init__(self, _wiki_path):
        self.wiki_path = _wiki_path
        self.wiki = None

        # jieba custom setting.
        jieba.set_dictionary(os.path.join(root, 'jieba_dict/dict.txt.big'))

        # load stopwords set
        _stopwords_path = os.path.join(root, "jieba_dict/stopwords.txt")
        self.stopword_set = set()
        with open(_stopwords_path, 'r', encoding='utf-8') as _stopwords:
            for _num, _line in enumerate(_stopwords):
                self.stopword_set.add(_line.strip('\n'))

    def __iter__(self):
        with open(self.wiki_path, "r", encoding='utf-8') as wiki:
            for wiki_index, content in enumerate(wiki):
                lines = content.split(" ")
                title = lines[0]
                yield TaggedDocument(content[1:], [title])

    def pre_process(self, _line):
        _output = []
        _line = _line.strip('\n')
        _words = jieba.cut(_line, cut_all=False)
        for _word in _words:
            if _word not in self.stopword_set:
                _output.append(_word)

        return _output


wiki_path = os.path.join(root, "wiki_texts.txt")
documents = TaggedWikiDocument(wiki_path)

print(type(documents))

for document in documents:
    doc = list(document)
    print(len(doc))
    print(len(doc[1]))
    print(doc[1])
    break

# region 建立與訓練模型
# Preprocessing
# To set the same vocabulary size with original paper. We first calculate the optimal min_count parameter.
cores = multiprocessing.cpu_count()
print(cores)

models = [
    # dm=0 >>> PV-DBOW
    Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=19, epochs=10, workers=cores),
    # dm=1 >>> PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8, min_count=19, epochs=10, workers=cores),
]

models[0].build_vocab(documents)
print(str(models[0]))

models[1].reset_from(models[0])
print(str(models[1]))

for model in models:
    start = time.time()
    model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
    print(str(model), " cost time:", (time.time() - start))
    print("model.corpus_count", model.corpus_count)
    print("model.iter", model.iter)
# endregion

for model in models:
    print(str(model))
    print(model.docvecs.most_similar(positive=['經濟學'], topn=20))

for model in models:
    print(str(model))
    print(model.docvecs.most_similar(positive=["政治學"], topn=20))

for model in models:
    print(str(model))
    pprint(model.docvecs.most_similar(positive=["法學"], topn=10))

for model in models:
    print(str(model))
    vec = [model["經濟學"] - model["政治學"] + model["法學"]]
    pprint([m for m in model.docvecs.most_similar(vec, topn=11) if m[0] != "經濟學"])


# 儲存兩個模型，雖然兩個感覺都不太好
model = models[0]
model_path = "Crescent/SaveModel/doc2vec_PVDBOW_wiki.model"
model.save(model_path)
model_doc2vec_PVDBOW_wiki = Doc2Vec.load(model_path)
# 產生句向量
print(model_doc2vec_PVDBOW_wiki.infer_vector(["經濟", "海嘯"]))

for i, key in enumerate(model_doc2vec_PVDBOW_wiki.docvecs.doctags.keys()):
    print(i, key)
    if i == 10:
        break

model = models[1]
model_path = "Crescent/SaveModel/doc2vec_PVDM_wiki.model"
model.save(model_path)
model_doc2vec_PVDM_wiki = Doc2Vec.load(model_path)
# 產生句向量
print(model_doc2vec_PVDM_wiki.infer_vector(["經濟", "海嘯"]))

for i, key in enumerate(model_doc2vec_PVDM_wiki.docvecs.doctags.keys()):
    print(i, key)
    if i == 10:
        break
# endregion
