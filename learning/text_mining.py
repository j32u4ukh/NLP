import nltk
from nltk import TextCollection
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

# region Demo1: tf-idf
# https://ithelp.ithome.com.tw/articles/10194822
# 『詞頻』（term frequency，tf）是單字在文件中出現的次數，
# 『逆向檔案頻率』（inverse document frequency，idf）是『所有文件數』除以『有出現該單字的檔案數』，
# 其中 idf 分母通常會加1，避免除以0的問題，另外為避免極端值影響過大，
# 故取 log，tf-idf公式如下：
# tf-idf = tf * idf。


def demo1():
    # 使用 TextCollection class 計算 tf-idf
    # input為 text1, text2, text3
    tf_idf = TextCollection([text1, text2, text3])

    # 計算 tf，例如，book在text3的tf
    # tf = text.count(term) / len(text) = 2.233937985881512e-05
    book_tf = tf_idf.tf("book", text3)
    print("book_tf:", book_tf)

    # 計算 idf，例如，book在text1、text2、text3的idf
    # idf = (log(len(self._texts) / matches) if matches else 0.0) = 1.0986122886681098
    Book_idf = tf_idf.idf("Book")  # 大小寫有差別
    print("Book_idf:", Book_idf)
# endregion


# region Document-term matrix
def demo2():
    docs = ['why hello there', 'omg hello pony', 'she went there? omg']
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    print(df)
# endregion


# region Document-term matrix + SVD
def demo3():
    # docs = ['why hello there',
    #         'omg hello world',
    #         'she went there? omg',
    #         "hello world",
    #         "hello beautiful world"]

    docs = [" ".join("台泥"),
            " ".join("台泥張安平"),
            " ".join("張安平"),
            " ".join("亞泥"),
            " ".join("亞洲水泥")]

    # docs = ["台泥",
    #         "台泥張安平",
    #         "張安平",
    #         "亞泥",
    #         "亞洲水泥".split()]

    vec = CountVectorizer(token_pattern=u'(?u)\w+')
    x = vec.fit_transform(docs)
    terms = vec.get_feature_names()
    df = pd.DataFrame(x.toarray(), columns=terms)
    trem_document = df.T.values
    print("trem_document", trem_document.shape)
    print(trem_document)
    term_concept, concept_concept, concept_doc = np.linalg.svd(trem_document, full_matrices=True)
    print("term_concept", term_concept.shape)
    print(term_concept)
    print("concept_concept", concept_concept.shape)
    print(concept_concept)
    print("concept_doc", concept_doc.shape)
    print(concept_doc)
# endregion


if __name__ == "__main__":
    demo3()
