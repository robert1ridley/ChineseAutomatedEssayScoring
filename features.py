import numpy as np
from snownlp import SnowNLP
from stopwordsiso import stopwords
import sklearn.preprocessing as sk
stops = stopwords(["zh"])


def get_word_and_sent_features(essay):
    s = SnowNLP(essay)
    sentences = s.sentences

    # Average Sentence Length
    sent_words = [SnowNLP(sent).words for sent in sentences]
    sent_lengths = [len(sent) for sent in sent_words]
    sent_sum = sum(sent_lengths)
    sent_mean = sent_sum / len(sent_lengths)

    # Word Features
    word_lengths = []
    longest_word = -1
    num_stopwords = 0
    for sent in sent_words:
        for word in sent:
            word_lengths.append(len(word))
            if len(word) > longest_word:
                longest_word = len(word)
            if word in stops:
                num_stopwords += 1

    word_sum = sum(word_lengths)
    word_mean = word_sum / len(word_lengths)
    proportion_stopwords = num_stopwords / len(word_lengths)
    proportion_non_stop = 1 - proportion_stopwords

    # Sentiment Features
    sent_sentiments = [SnowNLP(sent).sentiments for sent in sentences]
    neg_sentence_count = [1 for sentiment in sent_sentiments if sentiment < 0.5]
    pos_sentence_count = [1 for sentiment in sent_sentiments if sentiment > 0.5]
    total_neg_sentences = len(neg_sentence_count)
    total_pos_sentences = len(pos_sentence_count)
    neg_sent_proportion = total_neg_sentences / len(sent_sentiments)
    pos_sent_proportion = total_pos_sentences / len(sent_sentiments)

    # Similarity Features
    sent_pairs = []
    for index, sent in enumerate(sentences):
        if index != len(sentences)-1:
            sent_pairs.append((sent, sentences[index+1]))

    sentence_similarities = []
    for pair in sent_pairs:
        s1 = SnowNLP(pair[0])
        sim = s1.sim(pair[1])
        sim_sum = sum(sim)
        sim_ave = sim_sum / len(sim)
        sentence_similarities.append(sim_ave)

    sent_sim_sum = sum(sentence_similarities)
    try:
        essay_wide_mean_sim = sent_sim_sum / len(sentence_similarities)
    except ZeroDivisionError:
        essay_wide_mean_sim = 0.0

    return [sent_mean, word_mean, longest_word, proportion_non_stop, neg_sent_proportion, pos_sent_proportion,
            essay_wide_mean_sim]


def calc_features(essay_sets):
    features_lists = []
    for essay_set in essay_sets:
        set_features = []
        for essay in essay_set:
            essay_features = get_word_and_sent_features(essay)
            set_features.append(essay_features)
        features_lists.append(set_features)
    return features_lists


def get_features(train, dev, test):
    all_set_features = calc_features([train, dev, test])
    train, dev, test = all_set_features[0], all_set_features[1], all_set_features[2]
    train = np.array(train)
    dev = np.array(dev)
    test = np.array(test)

    scaler = sk.MinMaxScaler()
    scaler = scaler.fit(train)
    # print(scaler)
    X_train = scaler.transform(train)
    X_dev = scaler.transform(dev)
    X_test = scaler.transform(test)

    return X_train, X_dev, X_test
