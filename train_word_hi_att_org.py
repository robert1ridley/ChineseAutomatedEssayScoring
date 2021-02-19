import sys
import time
import json
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from configs.configs import OrgConfigs
from utils import get_data, get_org_data_sents_and_words, create_vocab, org_data_essay_to_ids, \
    convert_org_data_scores_to_new_scores, pad_hierarchical_text_sequences, pad_flat_text_sequences, \
    scale_down_scores, load_word_embedding_dict, build_embedd_table
from models.word_hi_att_model import build_word_hi_att_text_only
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from evaluators.evaluator import Evaluator


def main():
    configs = OrgConfigs()

    data_path = configs.DATA_PATH
    data = get_data(data_path)
    random.shuffle(data)
    if configs.DEBUG:
        data, longest_sent_count, longest_sent, longest_title = get_org_data_sents_and_words(data[:100])
    else:
        data, longest_sent_count, longest_sent, longest_title = get_org_data_sents_and_words(data)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.1, random_state=42)

    word_vocab = create_vocab(train_data, configs)
    vocab_path = configs.MODEL_OUTPUT_PATH + "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as fp:
        json.dump(word_vocab, fp)
    train_titles, train_texts, train_scores = org_data_essay_to_ids(train_data, word_vocab)
    dev_titles, dev_texts, dev_scores = org_data_essay_to_ids(dev_data, word_vocab)
    test_titles, test_texts, test_scores = org_data_essay_to_ids(test_data, word_vocab)

    train_scores_y = convert_org_data_scores_to_new_scores(train_scores)
    dev_scores_y = convert_org_data_scores_to_new_scores(dev_scores)
    test_scores_y = convert_org_data_scores_to_new_scores(test_scores)

    train_titles_X = pad_flat_text_sequences(train_titles, longest_title)
    dev_titles_X = pad_flat_text_sequences(dev_titles, longest_title)
    test_titles_X = pad_flat_text_sequences(test_titles, longest_title)

    train_texts_X = pad_hierarchical_text_sequences(train_texts, longest_sent_count, longest_sent)
    dev_texts_X = pad_hierarchical_text_sequences(dev_texts, longest_sent_count, longest_sent)
    test_texts_X = pad_hierarchical_text_sequences(test_texts, longest_sent_count, longest_sent)

    train_texts_X = train_texts_X.reshape((train_texts_X.shape[0], train_texts_X.shape[1] * train_texts_X.shape[2]))
    dev_texts_X = dev_texts_X.reshape((dev_texts_X.shape[0], dev_texts_X.shape[1] * dev_texts_X.shape[2]))
    test_texts_X = test_texts_X.reshape((test_texts_X.shape[0], test_texts_X.shape[1] * test_texts_X.shape[2]))

    train_scores_y_scaled = scale_down_scores(train_scores_y)
    dev_scores_y_scaled = scale_down_scores(dev_scores_y)
    test_scores_y_scaled = scale_down_scores(test_scores_y)

    embedding_path = configs.EMBEDDING_PATH
    embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
    print('embedd_dict complete')
    embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
    embed_table = [embedd_matrix]

    train_inputs = [train_texts_X]
    dev_inputs = [dev_texts_X]
    test_inputs = [test_texts_X]

    model = build_word_hi_att_text_only(len(word_vocab), longest_sent_count, longest_sent, configs,
                                        embedding_weights=embed_table)
    model_path = configs.MODEL_OUTPUT_PATH + "org_word_hi_att_text_only_model"
    evaluator = Evaluator(dev_inputs, dev_scores_y_scaled, test_inputs, test_scores_y_scaled)
    evaluator.evaluate(model, -1, print_info=True)
    epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE
    layers_dict = {'ZeroMaskedEntries': ZeroMaskedEntries, 'Attention': Attention}
    for ii in range(epochs):
        print('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time.time()
        model.fit(train_inputs,
                  train_scores_y_scaled, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        print("Training one epoch in %.3f s" % tt_time)
        best = evaluator.evaluate(model, ii + 1)
        if best:
            print("SAVING TO ", model_path)
            tf.keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True, save_format='h5')
            rec_model = tf.keras.models.load_model(model_path, custom_objects=layers_dict)
            np.testing.assert_allclose(
                model.predict(test_inputs), rec_model.predict(test_inputs)
            )

    evaluator.print_final_info()


if __name__ == '__main__':
    main()
