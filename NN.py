from keras.layers import *
import keras.backend as K
from keras.layers import Layer
import tensorflow as tf
from keras.layers import Dense
from keras import layers, Model
from keras import regularizers
import numpy as np

from tensorflow.keras.initializers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape







class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn



class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1)  # Add dimension (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # Add dimension (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # shape = (batch, seq_len, 2)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


def build_model(n_steps_in,n_steps_out, n_feature, n_output, D_MODEL,optimizer):
    # inp = Input(shape = (SEQ_LEN, feat_size))

    time_embedding = Time2Vector(n_steps_in)

    in_seq = Input(shape=(n_steps_in, n_feature))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x, self_attn = EncoderLayer(
        d_model=D_MODEL,
        d_inner_hid=128,
        n_head=64,
        d_k=64,
        d_v=64,
        dropout=0)(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    final = Dense(128, activation="relu")(conc)
    dense = Dense(n_output, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.L2(1e-4),name = 'output_layer')(final)
    output = Reshape((1,3))(dense)
    model = Model(inputs=in_seq, outputs=output)
    model.compile(
        loss="MSE",
        optimizer=optimizer,
        metrics=['mae'])

    return model




def inverse_transform(y_test, yhat):
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    yhat_reshaped = yhat.reshape(-1, yhat.shape[-1])
    yhat_inverse = scaler.inverse_transform(yhat_reshaped)
    y_test_inverse = scaler.inverse_transform(y_test_reshaped)
    return yhat_inverse, y_test_inverse

def evaluate_forecast(y_test_inverse, yhat_inverse):
    mse_ = tf.keras.losses.MeanSquaredError()
    mae_ = tf.keras.losses.MeanAbsoluteError()
    mape_ = tf.keras.losses.MeanAbsolutePercentageError()
    mae = mae_(y_test_inverse,yhat_inverse)
    print('mae:', mae)
    mse = mse_(y_test_inverse,yhat_inverse)
    print('mse:', mse)
    mape = mape_(y_test_inverse,yhat_inverse)
    print('mape:', mape)


def Loss_D(y_true, y_pred):
    p = 0.8
    mse_loss = tf.reduce_mean(tf.math.squared_difference(y_true[:, 0, 0], y_pred[:, 0, 0])
                              + tf.math.squared_difference(y_true[:, 1, 0], y_pred[:, 1, 0])
                              + tf.math.squared_difference(y_true[:, 0, 1], y_pred[:, 0, 1])
                              + tf.math.squared_difference(y_true[:, 1, 1], y_pred[:, 1, 1])
                              + tf.math.squared_difference(y_true[:, 0, 2], y_pred[:, 0, 2])
                              + tf.math.squared_difference(y_true[:, 1, 2], y_pred[:, 1, 2])
                              + tf.math.squared_difference(y_true[:, 0, 3], y_pred[:, 0, 3])
                              + tf.math.squared_difference(y_true[:, 1, 3], y_pred[:, 1, 3])
                              + tf.math.squared_difference(y_true[:, 0, 4], y_pred[:, 0, 4])
                              + tf.math.squared_difference(y_true[:, 1, 4], y_pred[:, 1, 4])
                              + tf.math.squared_difference(y_true[:, 0, 5], y_pred[:, 0, 5])
                              + tf.math.squared_difference(y_true[:, 1, 5], y_pred[:, 1, 5]))

    # print((df2[3][1]-0.4)*4-(df2[0][1]-0.45)*45+(df2[0][0]-0.45)*45)
    # print((df2[4][1]-0.5)*8-(df2[1][1]-0.5)*52+(df2[1][0]-0.5)*52)
    # print((df2[5][1]-0.3)*5-(df2[2][1]+0.05)*45+(df2[2][0]+0.05)*45)

    lossDx = tf.reduce_mean(tf.math.squared_difference((y_pred[:, 1, 0] - 0.45) * 45. - (y_pred[:, 0, 0] - 0.45) * 45.,
                                                       (y_pred[:, 1, 3] - 0.4) * 4))
    lossDy = tf.reduce_mean(tf.math.squared_difference((y_pred[:, 1, 1] - 0.5) * 52. - (y_pred[:, 0, 1] - 0.5) * 52.,
                                                       (y_pred[:, 1, 4] - 0.5) * 8))
    lossDz = tf.reduce_mean(tf.math.squared_difference((y_pred[:, 1, 2] + 0.05) * 45. - (y_pred[:, 0, 2] + 0.05) * 45.,
                                                       (y_pred[:, 1, 5] - 0.3) * 5))
    # dy_loss = tf.cast(fx,tf.float32)
    #     if y_pred[2]<0: fy=y_pred[2]+1-y_pred[1]
    #     else: fy=y_pred[2]-y_pred[1]
    #     fy = tf.cond(tf.less(y_pred[:,2],0), lambda: tf.add(tf.subtract(y_pred[:,2], y_pred[:,1]),1),
    #            lambda: tf.subtract(y_pred[:,2], y_pred[:,1]))
    loss_total = p * mse_loss + ((1 - p) * lossDx + (1 - p) * lossDy + (1 - p) * lossDz) / 3
    # loss = tf.reduce_mean(loss_total)

    return loss_total


