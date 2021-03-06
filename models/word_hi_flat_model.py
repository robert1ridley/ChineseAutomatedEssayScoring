import tensorflow.keras.layers as layers
from tensorflow import keras
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention


def build_word_flat_att(vocab_size, maxlen, lengths_count, longest_title, configs, embedding_weights):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS

    word_input = layers.Input(shape=(maxlen,), dtype='int32', name='word_input')
    x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxlen,
                         weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = layers.Dropout(dropout_prob, name='drop_x')(x_maskedout)
    zcnn = layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid', name='zcnn')(drop_x)
    avg_zcnn = Attention()(zcnn)

    essay_length_input = layers.Input(shape=(1,), dtype='int32', name='essay_length_input')
    length_embedding = layers.Embedding(output_dim=embedding_dim, input_dim=lengths_count, input_length=1,
                                        weights=None, mask_zero=True, name='length_embedding')(essay_length_input)

    title_input = layers.Input(shape=(longest_title,), dtype='int32', name='title_input')
    title_x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=longest_title,
                               weights=embedding_weights, mask_zero=True, name='title_x')(title_input)
    title_hz_lstm = layers.LSTM(lstm_units, return_sequences=True, name='title_hz_lstm')(title_x)

    title_essay_attention = layers.Attention()([avg_zcnn, title_hz_lstm])
    comb_avg_hz_lstm = Attention(name='comb_avg_hz_lstm')(title_essay_attention)

    length_layer = layers.Flatten()(length_embedding)
    conc = layers.Concatenate()([avg_zcnn, comb_avg_hz_lstm, length_layer])

    y = layers.Dense(units=1, activation='sigmoid', name='y_att')(conc)

    model = keras.Model(inputs=[word_input, title_input, essay_length_input], outputs=y)

    model.summary()

    model.compile(loss='mse', optimizer='rmsprop')

    return model
