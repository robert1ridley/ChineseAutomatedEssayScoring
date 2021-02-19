import sys
import time
import random
from sklearn.model_selection import train_test_split
from configs.configs import OrgConfigs
from utils import get_data, get_pos_org_data_sents_and_words, create_pos_vocab, org_pos_data_essay_to_ids, \
    convert_org_data_scores_to_new_scores, pad_hierarchical_text_sequences, scale_down_scores
from features import get_features
from models.word_hi_att_model import build_word_hi_att_text_only, build_features_model
from evaluators.evaluator import Evaluator


def main():
    configs = OrgConfigs()

    data_path = configs.DATA_PATH
    data = get_data(data_path)
    random.shuffle(data)
    if configs.DEBUG:
        data, longest_sent_count, longest_sent, longest_title = get_pos_org_data_sents_and_words(data[:100])
    else:
        data, longest_sent_count, longest_sent, longest_title = get_pos_org_data_sents_and_words(data)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.1, random_state=42)

    word_vocab = create_pos_vocab(train_data, configs)
    train_titles, train_texts, train_scores, train_raw = org_pos_data_essay_to_ids(train_data, word_vocab)
    dev_titles, dev_texts, dev_scores, dev_raw = org_pos_data_essay_to_ids(dev_data, word_vocab)
    test_titles, test_texts, test_scores, test_raw = org_pos_data_essay_to_ids(test_data, word_vocab)

    train_features, dev_features, test_features = get_features(train_raw, dev_raw, test_raw)

    train_scores_y = convert_org_data_scores_to_new_scores(train_scores)
    dev_scores_y = convert_org_data_scores_to_new_scores(dev_scores)
    test_scores_y = convert_org_data_scores_to_new_scores(test_scores)

    train_texts_X = pad_hierarchical_text_sequences(train_texts, longest_sent_count, longest_sent)
    dev_texts_X = pad_hierarchical_text_sequences(dev_texts, longest_sent_count, longest_sent)
    test_texts_X = pad_hierarchical_text_sequences(test_texts, longest_sent_count, longest_sent)

    train_texts_X = train_texts_X.reshape((train_texts_X.shape[0], train_texts_X.shape[1] * train_texts_X.shape[2]))
    dev_texts_X = dev_texts_X.reshape((dev_texts_X.shape[0], dev_texts_X.shape[1] * dev_texts_X.shape[2]))
    test_texts_X = test_texts_X.reshape((test_texts_X.shape[0], test_texts_X.shape[1] * test_texts_X.shape[2]))

    train_scores_y_scaled = scale_down_scores(train_scores_y)
    dev_scores_y_scaled = scale_down_scores(dev_scores_y)
    test_scores_y_scaled = scale_down_scores(test_scores_y)

    train_inputs = [train_features]
    dev_inputs = [dev_features]
    test_inputs = [test_features]

    # model = build_word_hi_att_text_only(len(word_vocab), longest_sent_count, longest_sent, configs,
    #                                     embedding_weights=None)
    model = build_features_model(train_features.shape[1])
    evaluator = Evaluator(dev_inputs, dev_scores_y_scaled, test_inputs, test_scores_y_scaled)
    evaluator.evaluate(model, -1, print_info=True)
    epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE
    for ii in range(epochs):
        print('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time.time()
        model.fit(train_inputs,
                  train_scores_y_scaled, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        print("Training one epoch in %.3f s" % tt_time)
        evaluator.evaluate(model, ii + 1)

    evaluator.print_final_info()


if __name__ == '__main__':
    main()
