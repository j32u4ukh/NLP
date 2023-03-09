import logging

import jieba
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from opencc import OpenCC
import numpy as np

"""類神經網路-實作英翻中"""

"""簡轉繁"""
path = "cmn.txt"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
line_num = 0
# s2t_array = []
cc = OpenCC('s2t')      
with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        with open("translation_texts2.txt", 'a', encoding='utf-8') as output:
            output.write(cc.convert(line))
        # s2t_array.append(cc.convert(line))
        line_num += 1
        if line_num % 1000 == 0:
            logging.info("已處理 %d 句" % line_num)


"""文字處理"""
jieba_file = 'data/jieba_dict/'
# 處理斷詞
jieba.set_dictionary(jieba_file + 'dict.txt.big')
stopword_set = set()
with open(jieba_file + 'stopwords.txt', 'r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword = stopwords.readline()
        stopword_set.add(stopword.strip('\n'))

output = open('translation_seg.txt', 'w', encoding='utf-8')
with open(path, 'r', encoding='utf-8') as content :
    for texts_num, line in enumerate(content):
        line = line.strip('\n')
        words = jieba.cut(line, cut_all = False)
        for word in words:
            if word not in stopword_set:
                output.write(word + ' ')
        output.write('\n')

        if (texts_num + 1) % 5000 == 0:
            logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
output.close()


"""開始翻譯"""
# 參數設定
# Batch size for training.
batch_size = 64 
# Number of epochs to train for. 
epochs = 100
# Latent dimensionality of the encoding space.  
latent_dim = 256
# Number of samples to train on.
num_samples = 10000  
# 設定訓練資料檔路徑
data_path = 'translation_texts2.txt' 
# 讀取資料檔，並將所有單字整理為字典，分別為英文及中文字典
# 注意，英文為字母的集合，非單字(Word)
# 將 input語言 翻譯成 target語言
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path, encoding='utf8').read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
# 字典排序
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

# 計算編碼器、解碼器的最大長度
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
# 以dict儲存字典單字及序號
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# 設定編碼器、解碼器input起始值(均為0矩陣)
encoder_input_data = np.zeros((len(input_texts),
                               max_encoder_seq_length,
                               num_encoder_tokens),
                              dtype='float32')
decoder_input_data = np.zeros((len(input_texts),
                               max_decoder_seq_length,
                               num_decoder_tokens),
                              dtype='float32')
decoder_target_data = np.zeros((len(input_texts),
                                max_decoder_seq_length,
                                num_decoder_tokens),
                               dtype='float32')

# 設定 encoder_input、decoder_input對應的順序
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# 建立 encoder LSTM 隱藏層
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# 捨棄 output，只保留記憶狀態 h 及 c
encoder_states = [state_h, state_c]

# 建立 decoder LSTM 隱藏層
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# decoder 記憶狀態不會在訓練過程使用，只會在推論(Inference)使用
decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                    return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 定義模型，由 encoder_input_data 及 decoder_input_data
# 轉換為 decoder_target_data 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 訓練
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], 
          decoder_target_data, batch_size=batch_size,
          epochs=epochs, validation_split=0.2)

# 儲存模型及結果
model.save('../SaveModel/English2Chinese.h5')

# 推論(Inference)
# 過程如下:
# 1) encode input and retrieve initial decoder state
# 編碼輸入並檢索初始解碼器狀態
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# 以這個初始狀態運行一步解碼器，並以“序列開始”標記作為目標。
# Output will be the next target token
# 輸出將是下一個目標令牌
# 3) Repeat with the current target token and current states
# 重複當前目標令牌和當前狀態

# 定義編碼器取樣模型
encoder_model = Model(encoder_inputs, encoder_states)

# 定義解碼器的input
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 定義解碼器 LSTM 模型
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)

# 以編碼器的記憶狀態 h 及 c 為解碼器的記憶狀態
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 建立反向的 dict，才能透過查詢將數值轉回文字
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# 模型預測，並取得翻譯結果(中文)
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[int(sampled_token_index)]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# 測試100次
for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('*')
    print('Input sentence:', input_texts[seq_index])
    try:
        print('Decoded sentence:', decoded_sentence)
    except:
        # 出現亂碼，以?取代
        print('Decoded sentence:', 
              decoded_sentence.encode('ascii', 'replace'))
        # print("error:", sys.exc_info()[0])
