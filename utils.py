import re
import jieba
import numpy as np


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def get_data(file_location):
    raw_data = open(file_location, 'r')
    return raw_data.readlines()


def is_number(token):
    return bool(num_regex.match(token))


def convert_original_scores_to_new_scores(score_list):
    return np.array([[int(score) - 1] for score in score_list])


def scale_down_scores(scores_array, min_score=1, max_score=3):
    return (scores_array - min_score) / (max_score - min_score)


def rescale_scores(scores_array, min_score=1, max_score=3):
    rescaled = scores_array * (max_score - min_score) + min_score
    return np.around(rescaled).astype(int)


def convert_to_ids_array(in_list, id_dict):
    return np.array([[id_dict[item]] for item in in_list])


def create_id_dict(items_list):
    unique_items = set()
    for item in items_list:
        unique_items.add(item)

    item_to_id = {}
    for i, item in enumerate(unique_items):
        item_to_id[item] = i
    return item_to_id


def pad_flat_text_sequences(index_sequences, max_title_len):
    X = np.empty([len(index_sequences), max_title_len], dtype=np.int32)

    for i, essay in enumerate(index_sequences):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)
        for j in range(num):
            word_id = sequence_ids[j]
            X[i, j] = word_id
        length = len(sequence_ids)
        X[i, length:] = 0
    return X


def pad_hierarchical_text_sequences(index_sequences, max_sentnum, max_sentlen):
    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid
            X[i, j, length:] = 0

        X[i, num:, :] = 0
    return X


def shorten_sentence(sent, max_sentlen=50):
    new_tokens = []
    sent = sent.strip()
    words = jieba.cut(sent)
    tokens = [word for word in words]
    if len(tokens) > max_sentlen:
        split_keywords = ['因为', '但是', '所以', '不过', '因此', '此外', '可是', '从而', '不然', '无论如何', '由于']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    return new_tokens


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    sents = para.split("\n")
    sent_tokens = []
    for sent in sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlen=50)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def create_vocab(data, configs):
    vocab_size = configs.VOCAB_SIZE
    word_vocab_count = {}
    for essay in data:
        essay_title = essay['essay_title']
        essay_text = essay['essay_text']
        essay_title = jieba.cut(essay_title)
        for word in essay_title:
            try:
                word_vocab_count[word] += 1
            except KeyError:
                word_vocab_count[word] = 1
        for sentence in essay_text:
            for word in sentence:
                try:
                    word_vocab_count[word] += 1
                except KeyError:
                    word_vocab_count[word] = 1

    import operator
    sorted_word_freqs = sorted(word_vocab_count.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1

    word_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(word_vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        word_vocab[word] = index
        index += 1
    return word_vocab


def get_sents_and_words(data, configs):
    longest_sent_count = -1
    longest_sent = -1
    longest_title = -1
    essays = []
    for line in data:
        items = line.split('\t')
        essay_id = items[0]
        essay_title = items[2]
        essay_text = items[3]
        essay_text = essay_text.replace("PARAGRAPH", "")
        # KEEP ONLY ESSAYS SHORTER THAT 1200 CHARS AND ARE IN THE SPECIFIED AGE GROUPS
        if items[6] not in ["1200字", "1200字以上"] and items[5] in configs.AGE_GROUPS:
            essay = {}
            sentences = cut_sent(essay_text)
            score = items[4]
            if score == '5':
                score = '4'
            essay['essay_id'] = essay_id
            essay['essay_title'] = essay_title
            essay['essay_text'] = sentences
            essay['score'] = score
            essay['age'] = items[5]
            essay['length'] = items[6]
            essays.append(essay)
            if len(sentences) > longest_sent_count:
                longest_sent_count = len(sentences)
            for sentence in sentences:
                if len(sentence) > longest_sent:
                    longest_sent = len(sentence)
            title_words = list(jieba.cut(essay_title))
            if len(title_words) > longest_title:
                longest_title = len(title_words)
    return essays, longest_sent_count, longest_sent, longest_title


def get_words(data, configs):
    longest_essay = -1
    longest_title = -1
    essays = []
    for line in data:
        items = line.split('\t')
        essay_id = items[0]
        essay_title = items[2]
        essay_text = items[3]
        essay_text = essay_text.replace("PARAGRAPH", "")
        if items[6] not in ["1200字", "1200字以上"] and items[5] in configs.AGE_GROUPS:
            essay = {}
            essay_text = list(jieba.cut(essay_text))
            score = items[4]
            if score == '5':
                score = '4'
            essay['essay_id'] = essay_id
            essay['essay_title'] = essay_title
            essay['essay_text'] = essay_text
            essay['score'] = score
            essay['age'] = items[5]
            essay['length'] = items[6]
            essays.append(essay)
            if len(essay_text) > longest_essay:
                longest_essay = len(essay_text)
            title_words = list(jieba.cut(essay_title))
            if len(title_words) > longest_title:
                longest_title = len(title_words)
    return essays, longest_essay, longest_title


def essay_to_ids(essay_set, word_vocab):
    essay_titles = []
    essay_texts = []
    essay_scores = []
    student_grades = []
    essay_lengths = []
    num_hit, unk_hit, total = 0., 0., 0.
    for essay in essay_set:
        essay_title = essay['essay_title']
        essay_text = essay['essay_text']
        essay_score = essay['score']
        student_grade = essay['age']
        essay_length = essay['length']

        # TITLE
        title_ids = []
        for i, word in enumerate(jieba.cut(essay_title)):
            if is_number(word):
                title_ids.append(word_vocab['<num>'])
            elif word in word_vocab.keys():
                title_ids.append(word_vocab[word])
            else:
                title_ids.append(word_vocab['<unk>'])
        essay_titles.append(title_ids)

        # TEXTS
        sentences_list = []
        for sentence in essay_text:
            sentence_ids = []
            for word in sentence:
                if is_number(word):
                    sentence_ids.append(word_vocab['<num>'])
                    num_hit += 1
                elif word in word_vocab.keys():
                    sentence_ids.append(word_vocab[word])
                else:
                    sentence_ids.append(word_vocab['<unk>'])
                    unk_hit += 1
                total += 1
            sentences_list.append(sentence_ids)
        essay_texts.append(sentences_list)

        essay_scores.append(essay_score)
        student_grades.append(student_grade)
        essay_lengths.append(essay_length)
    print(' num hit: {}, total: {}, unkn hit: {}'.format(num_hit, total, unk_hit))
    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return essay_titles, essay_texts, essay_scores, student_grades, essay_lengths


def essay_to_ids_flat(essay_set, word_vocab):
    essay_titles = []
    essay_texts = []
    essay_scores = []
    student_grades = []
    essay_lengths = []
    num_hit, unk_hit, total = 0., 0., 0.
    for essay in essay_set:
        essay_title = essay['essay_title']
        essay_text = essay['essay_text']
        essay_score = essay['score']
        student_grade = essay['age']
        essay_length = essay['length']

        # TITLE
        title_ids = []
        for i, word in enumerate(jieba.cut(essay_title)):
            if is_number(word):
                title_ids.append(word_vocab['<num>'])
            elif word in word_vocab.keys():
                title_ids.append(word_vocab[word])
            else:
                title_ids.append(word_vocab['<unk>'])
        essay_titles.append(title_ids)

        # TEXTS
        essay_ids = []
        for word in essay_text:
            if is_number(word):
                essay_ids.append(word_vocab['<num>'])
                num_hit += 1
            elif word in word_vocab.keys():
                essay_ids.append(word_vocab[word])
            else:
                essay_ids.append(word_vocab['<unk>'])
                unk_hit += 1
            total += 1
        essay_texts.append(essay_ids)
        essay_scores.append(essay_score)
        student_grades.append(student_grade)
        essay_lengths.append(essay_length)
    print(' num hit: {}, total: {}, unkn hit: {}'.format(num_hit, total, unk_hit))
    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return essay_titles, essay_texts, essay_scores, student_grades, essay_lengths


def load_word_embedding_dict(embedding_path):
    print("Loading Embedding ...")
    embedd_dim = -1
    embedd_dict = dict()
    first_line = True
    embedding_file = open(embedding_path, 'r')
    lines = embedding_file.readlines()
    for line in lines:
        line = line.strip()
        if not first_line:
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
        first_line = False
    embedding_file.close()
    return embedd_dict, embedd_dim, True


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim])
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    oov_num = 0
    for word in word_alphabet:
        ww = word.lower() if caseless else word
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        embedd_table[word_alphabet[word], :] = embedd
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    print("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table