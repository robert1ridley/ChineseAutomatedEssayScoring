import sys
import time
import random
from sklearn.model_selection import train_test_split
from configs.configs import Configs
from utils import get_data, get_words, create_vocab, essay_to_ids_flat, convert_original_scores_to_new_scores, \
    pad_flat_text_sequences, scale_down_scores, load_word_embedding_dict, \
    build_embedd_table, create_id_dict, convert_to_ids_array
from models.word_hi_flat_model import build_word_flat_att
from evaluators.evaluator import Evaluator


def main():
    configs = Configs()

    data_path = configs.DATA_PATH
    data = get_data(data_path)
    random.shuffle(data)
    if configs.DEBUG:
        data, longest_essay, longest_title = get_words(data[:300], configs)
    else:
        data, longest_essay, longest_title = get_words(data, configs)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.1, random_state=42)

    word_vocab = create_vocab(train_data, configs)
    train_titles, train_texts, train_scores, train_grades, train_lengths = essay_to_ids_flat(train_data, word_vocab)
    dev_titles, dev_texts, dev_scores, dev_grades, dev_lengths = essay_to_ids_flat(dev_data, word_vocab)
    test_titles, test_texts, test_scores, test_grades, test_lengths = essay_to_ids_flat(test_data, word_vocab)

    lengths_id_dict = create_id_dict(train_lengths)
    lengths_count = len(lengths_id_dict)
    train_lengths_X = convert_to_ids_array(train_lengths, lengths_id_dict)
    dev_lengths_X = convert_to_ids_array(dev_lengths, lengths_id_dict)
    test_lengths_X = convert_to_ids_array(test_lengths, lengths_id_dict)

    train_scores_y = convert_original_scores_to_new_scores(train_scores)
    dev_scores_y = convert_original_scores_to_new_scores(dev_scores)
    test_scores_y = convert_original_scores_to_new_scores(test_scores)

    train_titles_X = pad_flat_text_sequences(train_titles, longest_title)
    dev_titles_X = pad_flat_text_sequences(dev_titles, longest_title)
    test_titles_X = pad_flat_text_sequences(test_titles, longest_title)

    train_texts_X = pad_flat_text_sequences(train_texts, longest_essay)
    dev_texts_X = pad_flat_text_sequences(dev_texts, longest_essay)
    test_texts_X = pad_flat_text_sequences(test_texts, longest_essay)

    train_scores_y_scaled = scale_down_scores(train_scores_y)
    dev_scores_y_scaled = scale_down_scores(dev_scores_y)
    test_scores_y_scaled = scale_down_scores(test_scores_y)

    embedding_path = configs.EMBEDDING_PATH
    embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
    print('embedd_dict complete')
    embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
    embed_table = [embedd_matrix]

    train_inputs = [train_texts_X, train_titles_X, train_lengths_X]
    dev_inputs = [dev_texts_X, dev_titles_X, dev_lengths_X]
    test_inputs = [test_texts_X, test_titles_X, test_lengths_X]

    model = build_word_flat_att(len(word_vocab), longest_essay, lengths_count, longest_title,
                                configs, embedding_weights=embed_table)
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
