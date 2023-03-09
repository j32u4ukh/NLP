import nltk

"""
萃取法(Extractive Method)：從本文中挑選重要的字句，集合起來，成為摘要。
抽象法(Abstractive Method)：瞭解本文大意後，自動產生摘要。
"""

# region Demo1：萃取法(Extractive Method)
# https://ithelp.ithome.com.tw/articles/10195046
# 它是使用 NLTK library，未牽涉到 Neural Network

# 本文，大意是歐巴馬卸任
news_content = '''At noon on Friday, 55-year old Barack Obama became a federal retiree.
His pension payment will be $207,800 for the upcoming year, about half of his presidential salary.
Obama and every other former president also get seven months of "transition" services to help adjust to post-presidential life. 
The ex-Commander in Chief also gets lifetime Secret Service protection as well as allowances for things such as travel, office expenses, communications and health care coverage.
All those extra expenses can really add up. 
In 2015 they ranged from a bit over $200,000 for Jimmy Carter to $800,000 for George W. Bush, according to a government report. 
Carter doesn't get health insurance because you have to work for the federal government for five years to qualify.
'''

# 分詞、標註、NER、打分數，依分數高低排列句子
results = []
# 利用 nltk.sent_tokenize 分割句子，應該是用句點去區分
for sent_no, sentence in enumerate(nltk.sent_tokenize(news_content)):
    # nltk.word_tokenize 用於英文斷詞
    # ex: ['All', 'those', 'extra', 'expenses', 'can', 'really', 'add', 'up', '.']
    no_of_tokens = len(nltk.word_tokenize(sentence))
    # 利用 nltk.pos_tag 標記詞性與詞的類別
    # ex: [('All', 'PDT'), ('those', 'DT'), ('extra', 'JJ'), ('expenses', 'NNS'), 
    #      ('can', 'MD'), ('really', 'RB'), ('add', 'VB'), ('up', 'RP'), ('.', '.')]
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # 計算句子中屬於名詞的數量
    no_of_nouns = len([word for word, pos in tagged if pos in ["NN", "NNP"] ])
    # Use NER to tag the named entities. 產生樹結構 ex:
#    (S
#      All/PDT
#      those/DT
#      extra/JJ
#      expenses/NNS
#      can/MD
#      really/RB
#      add/VB
#      up/RP
#      ./.)
    ners = nltk.ne_chunk(tagged, binary=False)
    # hasattr() 函数用于判断对象是否包含对应的属性。>>>判斷該樹結構是否有 label
    no_of_ners = len([chunk for chunk in ners if hasattr(chunk, 'label')])
    # (有標籤數量 + 名詞數量) / 總句長
    score = (no_of_ners + no_of_nouns) / float(no_of_tokens)
    results.append((sent_no, no_of_tokens, no_of_ners, no_of_nouns, score, sentence))

# 依重要性順序列出句子
for sent in sorted(results, key=lambda x: x[4], reverse=True):
    print(sent[5])

for sent_no, sentence in enumerate(nltk.sent_tokenize(news_content)):
    print("===== %d =====" % sent_no)
    token = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    no_of_nouns = len([word for word, pos in tagged if pos in ["NN", "NNP"] ])
    ners = nltk.ne_chunk(tagged, binary=False)
    for chunk in ners:
        print("chunk:", chunk)
        print(len(chunk))
        print(type(chunk))
        print("hasattr(chunk, \'label\'):", hasattr(chunk, 'label'))
        print(".....")
    no_of_ners = len([chunk for chunk in ners if hasattr(chunk, 'label')])
#    print(no_of_ners)    
    
    print()
# endregion


########################################
#   Demo2：抽象法(Abstractive Method)   #
########################################
# https://ithelp.ithome.com.tw/articles/10195209
# 作法通常分兩階段：
# 1.熔合(fusion)：連貫萃取出的重要字句。
# 2.壓縮(compression)：將字句精簡，變成摘要。
'''沒有實際內容'''
