import logging

import keras
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf

from NNTools.utils import getLogger


INPUT_SIZE = 250
STEP = 50
BATCH_SIZE = 1
LR = 0.005

ENCODER1 = 50
ENCODER2 = 50
DECODER1 = 50
DECODER2 = 50
OUTPUT_SIZE = 250


def tensorflowModel(_level=logging.DEBUG):
    # 產生 logger 物件
    logger = getLogger("tensorflowModel")
    logger.level = _level

    with tf.variable_scope("placeholder", reuse=tf.AUTO_REUSE):
        # 自編碼的輸入輸出應為相同數據，但為清楚區分，仍分別給予命名。
        tf_x = tf.placeholder(tf.float32, [None, STEP, INPUT_SIZE])
        tf_y = tf.placeholder(tf.float32, [None, STEP, OUTPUT_SIZE])

        logger.debug("tf_x.shape:", tf_x.shape)
        logger.debug("tf_y.shape:", tf_y.shape)

    # region Encode 1
    """LSTM權重數量計算公式
    = 4 × [h（h + i）+ h]
    
    h: hidden layer units
    i: input layer units
    """
    rnn_e1 = []
    init_e1 = []
    outputs_e1 = []
    final_state_e1 = []

    # 這裡利用迴圈，將相同的 input 輸入到 3 個 dynamic_rnn 當中。
    # 50 + 50 + 50 權重數量 = 3 * 4 * [50（50 + 250）+ 50] =  3,0100
    # 150          權重數量 = 4 * [150（150 + 250）+ 150]  = 24,0600
    # 透過將 150 個 units 分為 3 * 50 個 units，可以有效降低權重數量。
    for i in range(3):
        scope_name = "encode1-{}".format(i)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # RNN   ENCODER1 = 50
            temp_rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=ENCODER1)
            logger.debug("[{}] rnn_cell.shape:{}".format(scope_name, temp_rnn_cell.shape))
            rnn_e1.append(temp_rnn_cell)

            # len(init_state) = 2
            # init_state[0].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(ENCODER1)])
            # init_state[1].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(ENCODER1)])
            temp_init_state = rnn_e1[i].zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
            logger.debug("len(temp_init_state): {}".format(len(temp_init_state)))
            logger.debug("temp_init_state[0].shape: {}".format(temp_init_state[0].shape))
            logger.debug("temp_init_state[1].shape: {}".format(temp_init_state[1].shape))
            init_e1.append(temp_init_state)

            # outputs.shape TensorShape([Dimension(BATCH_SIZE), Dimension(STEP), Dimension(ENCODER1)])
            # temp_final_state = (h_c, h_n)
            # h_c.shape TensorShape([Dimension(BATCH_SIZE), Dimension(ENCODER1)])
            # h_n.shape TensorShape([Dimension(BATCH_SIZE), Dimension(ENCODER1)])
            temp_outputs, temp_final_state = tf.nn.dynamic_rnn(
                rnn_e1[i],                   # cell you have chosen
                tf_x,                           # input
                initial_state=init_e1[i],   # the initial hidden state
                time_major=False                # False: (batch, time step, input); True: (time step, batch, input)
            )

            logger.debug("temp_outputs.shape: {}".format(temp_outputs.shape))
            logger.debug("temp_final_state.shape: {}".format(temp_final_state.shape))

            # 將輸出存在陣列中，以供後續的層做存取
            # outputs_e1.shape should be:TensorShape([3, Dimension(BATCH_SIZE), Dimension(STEP), Dimension(ENCODER1)])
            if len(outputs_e1) == 3:
                # outputs_e1 初始化完成，根據索引更新數值即可
                outputs_e1[i] = temp_outputs
            else:
                # 將數值依序加入 outputs_e1 以進行初始化
                outputs_e1.append(temp_outputs)

            if len(final_state_e1) == 3:
                # final_state_e1 初始化完成，根據索引更新數值即可
                final_state_e1[i] = temp_final_state
            else:
                # 將數值依序加入 final_state_e1 以進行初始化
                final_state_e1.append(temp_final_state)
    # endregion

    # 利用 concatenate 將 outputs_e1 的 3 個層拼接起來
    # outputs_e1.shape maybe:TensorShape([Dimension(BATCH_SIZE), Dimension(STEP), 3 * Dimension(ENCODER1)])
    encode1 = tf.keras.layers.concatenate(outputs_e1)
    logger.debug("encode1.shape:{}".format(encode1.shape))

    with tf.variable_scope("encode2", reuse=tf.AUTO_REUSE):
        # RNN   ENCODER2 = 50
        rnn_e2 = tf.nn.rnn_cell.LSTMCell(num_units=ENCODER2)
        logger.debug("[encode2] rnn_e2.shape:{}".format(rnn_e2.shape))

        # len(init_state) = 2
        # init_state[0].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(ENCODER2)])
        # init_state[1].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(ENCODER2)])
        init_state_e2 = rnn_e2.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)  # very first hidden state
        logger.debug("len(init_state):{}".format(len(init_state_e2)))
        logger.debug("init_state_e2[0].shape:{}".format(init_state_e2[0].shape))
        logger.debug("init_state_e2[1].shape:{}".format(init_state_e2[1].shape))

        # outputs.shape TensorShape([Dimension(batch=5), Dimension(TIME_STEP=10), Dimension(CELL_SIZE=32)])
        outputs_e2, final_state_e2 = tf.nn.dynamic_rnn(
            rnn_e2,  # cell you have chosen
            encode1,  # input
            initial_state=init_state_e2,  # the initial hidden state
            time_major=False  # False: (batch, time step, input); True: (time step, batch, input)
        )

        logger.debug("outputs_e2.shape: {}".format(outputs_e2.shape))
        logger.debug("final_state_e2.shape: {}".format(final_state_e2.shape))

    with tf.variable_scope("sentence_vector", reuse=tf.AUTO_REUSE):
        logger.debug("===== code layer =====")
        sentence_vector = outputs_e2[:, -1, :]
        logger.debug("sentence_vector.shape: {}".format(sentence_vector.shape))

        # deep copy of sentence_vector
        outs = tf.identity(sentence_vector)
        logger.debug("tf.identity(sentence_vector).shape: {}".format(outs.shape))
        outs = tf.reshape(outs, [-1, ENCODER2])
        logger.debug("tf.reshape(outs, [-1, ENCODER2]).shape: {}".format(outs.shape))

        # region 參考 tensorflow_backend.py def repeat(x, n)
        repeat_outs = tf.expand_dims(outs, 1)
        logger.debug("repeat_outs.shape: {}".format(repeat_outs.shape))
        pattern = tf.stack([1, STEP, 1])
        logger.debug("pattern: {}".format(pattern))
        # repeat outs STEP times
        repeat_layer = tf.tile(repeat_outs, pattern)
        logger.debug("repeat_layer.shape: {}".format(repeat_layer.shape))
        # endregion

    with tf.variable_scope("decode1", reuse=tf.AUTO_REUSE):
        # RNN   ENCODER2 = 50
        rnn_d1 = tf.nn.rnn_cell.LSTMCell(num_units=DECODER1)
        logger.debug("[decode1] rnn_d1.shape: {}".format(rnn_d1.shape))

        # len(init_state) = 2
        # init_state[0].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(DECODER1)])
        # init_state[1].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(DECODER1)])
        init_state_d1 = rnn_d1.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)  # very first hidden state
        logger.debug("len(init_state_d1):{}".format(len(init_state_d1)))
        logger.debug("init_state_d1[0].shape:{}".format(init_state_d1[0].shape))
        logger.debug("init_state_d1[1].shape:{}".format(init_state_d1[1].shape))

        # outputs.shape TensorShape([Dimension(batch=5), Dimension(TIME_STEP=10), Dimension(DECODER1)])
        outputs_d1, final_state_d1 = tf.nn.dynamic_rnn(
            rnn_d1,  # cell you have chosen
            repeat_layer,  # input
            initial_state=init_state_d1,  # the initial hidden state
            time_major=False  # False: (batch, time step, input); True: (time step, batch, input)
        )

        logger.debug("outputs_d1.shape: {}".format(outputs_d1.shape))
        logger.debug("final_state_d1.shape: {}".format(final_state_d1.shape))

    rnn_d2 = []
    init_state_d2 = []
    outputs_d2 = []
    final_state_d2 = []
    for i in range(3):
        scope_name = "decode2-{}".format(i)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # RNN   ENCODER1 = 50
            temp_rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=DECODER2)
            logger.debug("[{}] rnn_cell.shape:{}".format(scope_name, temp_rnn_cell.shape))
            rnn_d2.append(temp_rnn_cell)

            # len(init_state) = 2
            # init_state[0].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(DECODER2)])
            # init_state[1].shape = TensorShape([Dimension(BATCH_SIZE), Dimension(DECODER2)])
            temp_init_state = rnn_d2[i].zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
            logger.debug("len(temp_init_state): {}".format(len(temp_init_state)))
            logger.debug("temp_init_state[0].shape: {}".format(temp_init_state[0].shape))
            logger.debug("temp_init_state[1].shape: {}".format(temp_init_state[1].shape))
            init_state_d2.append(temp_init_state)

            # temp_outputs.shape TensorShape([Dimension(BATCH_SIZE), Dimension(STEP), Dimension(DECODER2)])
            temp_outputs, temp_final_state = tf.nn.dynamic_rnn(
                rnn_d2[i],                   # cell you have chosen
                outputs_d1,                           # input
                initial_state=init_state_d2[i],   # the initial hidden state
                time_major=False                # False: (batch, time step, input); True: (time step, batch, input)
            )

            logger.debug("temp_outputs.shape: {}".format(temp_outputs.shape))
            logger.debug("temp_final_state.shape: {}".format(temp_final_state.shape))

            # last_step_outputs.shape should be:TensorShape([Dimension(BATCH_SIZE), 1, Dimension(DECODER2)])
            # last_step_outputs = temp_outputs[:, -1, :]
            # logger.debug("last_step_outputs.shape: {}".format(last_step_outputs.shape))

            # flatten_outputs.shape should be:TensorShape([Dimension(BATCH_SIZE), Dimension(DECODER2)])
            # flatten_outputs = tf.reshape(last_step_outputs, [-1, DECODER2])
            # logger.debug("flatten_outputs.shape: {}".format(flatten_outputs.shape))

            # outputs_d2.shape should be:TensorShape([3, Dimension(BATCH_SIZE), Dimension(STEP), Dimension(DECODER2)])
            if len(outputs_d2) == 3:
                # outputs_d2 初始化完成，根據索引更新數值即可
                outputs_d2[i] = temp_outputs
            else:
                # 將數值依序加入 outputs_d2 以進行初始化
                outputs_d2.append(temp_outputs)

            if len(final_state_d2) == 3:
                # final_state_d2 初始化完成，根據索引更新數值即可
                final_state_d2[i] = temp_final_state
            else:
                # 將數值依序加入 final_state_d2 以進行初始化
                final_state_d2.append(temp_final_state)

    with tf.variable_scope("loss_and_train", reuse=tf.AUTO_REUSE):
        # https://blog.csdn.net/xwd18280820053/article/details/72867818
        # encode 的時候同樣的數據分別進入三個隱藏層，因此 decode 的時候也會要有三個同樣的數據(三個並排，不是形成矩陣)
        # labels.shape should be: TensorShape([3, Dimension(BATCH_SIZE), Dimension(STEP), Dimension(DECODER2)])
        labels = tf.tile(tf_y, [3, 1, 1, 1])
        logger.debug("labels.shape: {}".format(labels.shape))

        # labels.shape should be: TensorShape([Dimension(BATCH_SIZE), Dimension(STEP), 3 * Dimension(DECODER2)])
        labels = tf.reshape(labels, [BATCH_SIZE, STEP, -1])
        logger.debug("labels.shape: {}".format(labels.shape))

        # outputs_d2.shape should be:TensorShape([3, Dimension(BATCH_SIZE), Dimension(STEP), Dimension(DECODER2)])
        # 網路輸出 outputs_d2 ，利用 tf.reshape 調整維度，
        # TensorShape([Dimension(BATCH_SIZE), Dimension(STEP), 3 * Dimension(DECODER2)])。
        outs = tf.reshape(outputs_d2, [BATCH_SIZE, STEP, -1])
        # compute cost
        loss = tf.losses.mean_squared_error(labels=labels,
                                            predictions=outs)
        train = tf.train.AdamOptimizer(LR).minimize(loss)


def kerasSequential():
    # 產生 logger 物件
    logger = getLogger("kerasSequential")

    _model = Sequential()
    _model.add(LSTM(125, activation='relu', input_shape=(STEP, INPUT_SIZE), return_sequences=True))
    _model.add(LSTM(50, activation='relu', return_sequences=False))
    _model.add(RepeatVector(STEP))
    _model.add(LSTM(50, activation='relu', return_sequences=True))
    _model.add(LSTM(125, activation='relu', return_sequences=True))
    _model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
    _model.compile(optimizer='adam', loss='mse')
    _model.summary()

    return _model


def kerasModel():
    # 產生 logger 物件
    logger = getLogger("kerasModel")

    # Input shape 不包含 batch
    _input = Input(shape=(STEP, INPUT_SIZE), dtype='float32')

    _layer1_1 = LSTM(50, activation='relu', name='layer1_1', return_sequences=True)(_input)
    _layer1_2 = LSTM(50, activation='relu', name='layer1_2', return_sequences=True)(_input)
    _layer1_3 = LSTM(50, activation='relu', name='layer1_3', return_sequences=True)(_input)
    print("layer1:", _layer1_3.shape)

    _merge_layer = keras.layers.concatenate([_layer1_1, _layer1_2, _layer1_3])
    print("merge_layer:", _merge_layer.shape)

    # 編碼結果
    _layer2 = LSTM(50, activation='relu', name='layer2', return_sequences=False)(_merge_layer)
    print("layer2:", _layer2.shape)

    # 上一層為 1 維資料，為了輸入到 LSTM 當中，需拼接多次編碼結果，使維度與編碼前相同
    _layer2_sequences = RepeatVector(STEP)(_layer2)

    _layer3 = LSTM(50, activation='relu', name='layer3', return_sequences=True)(_layer2_sequences)

    _layer4_1 = LSTM(50, activation='relu', name='layer4_1', return_sequences=False)(_layer3)
    _layer4_2 = LSTM(50, activation='relu', name='layer4_2', return_sequences=False)(_layer3)
    _layer4_3 = LSTM(50, activation='relu', name='layer4_3', return_sequences=False)(_layer3)

    _model = Model(inputs=_input, outputs=[_layer4_1, _layer4_2, _layer4_3])

    return _model


if __name__ == "__main__":
    pass
