from collections import Counter
import numpy as np
import os
import pickle
import gensim


def load_data(fpath):
    inputs = []
    labels = []
    with open(fpath, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            text, label = line.strip().split("<SEP>")
            inputs.append(text.strip().split(" "))
            labels.append(label)

    return inputs, labels


def load_data_iwslt(fpath, label):
    inputs = []
    labels = []
    with open(fpath, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            text = line.strip()
            inputs.append(text.strip().split(" "))
            labels.append(label)

    return inputs, labels


def remove_stop_word(inputs, stop_word_path):
    all_words = [word for data in inputs for word in data]

    word_count = Counter(all_words)  # 统计词频，返回一个 dict
    sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)  # 由大到小排列

    words = [item[0] for item in sort_word_count]

    if stop_word_path:
        with open(stop_word_path, "r", encoding="utf8") as fr:
            stop_words = [line.strip() for line in fr.readlines()]
        words = [word for word in words if word not in stop_words]

    return words


def gen_vocab(words, labels, output_path='./data/iwslt/pretreat'):
    if os.path.exists(os.path.join(output_path, "word_to_index.pkl")) and \
            os.path.exists(os.path.join(output_path, "label_to_index.pkl")):

        with open(os.path.join(output_path, "word_to_index.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

        with open(os.path.join(output_path, "label_to_index.pkl"), "rb") as f:
            label_to_index = pickle.load(f)

        return word_to_index, label_to_index

    vocab = ["<PAD>", "<UNK>"] + words
    word_to_index = dict(zip(vocab, list(range(len(vocab)))))

    with open(os.path.join(output_path, "word_to_index.pkl"), "wb") as f:
        pickle.dump(word_to_index, f)

    unique_labels = list(set(labels))
    label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))
    with open(os.path.join(output_path, "label_to_index.pkl"), "wb") as f:
        pickle.dump(label_to_index, f)

    return word_to_index, label_to_index


def encode_sentence(inputs, word_to_index):
    inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence] for sentence in inputs]

    return inputs_idx


def padding(inputs, sequence_length):
    new_inputs = [sentence[:sequence_length]
                  if len(sentence) > sequence_length
                  else sentence + [0] * (sequence_length - len(sentence))
                  for sentence in inputs]

    return new_inputs


def encode_label(labels, label_to_index):
    labels_idx = [label_to_index[label] for label in labels]

    return labels_idx


def get_data(fpath, hp, label, outpath):
    inputs, labels = load_data_iwslt(fpath, label)

    words = remove_stop_word(inputs, hp.stop_word_path)

    word_to_index, label_to_index = gen_vocab(words, labels, output_path=outpath)

    inputs_idx = encode_sentence(inputs, word_to_index)

    inputs_idx = padding(inputs_idx, sequence_length=hp.sequence_length)

    labels_idx = encode_label(labels, label_to_index)

    return np.array(inputs_idx, dtype=int), np.array(labels_idx), label_to_index, len(word_to_index), words


def next_batch(x1, x2, y1, y2, batch_size):
    perm = np.arange(len(x1))
    np.random.shuffle(perm)
    x1 = x1[perm]
    x2 = x2[perm]
    y1 = y1[perm]
    y2 = y2[perm]

    num_batches = len(x1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x1 = np.array(x1[start: end], dtype="int32")
        batch_x2 = np.array(x2[start: end], dtype="int32")
        batch_y1 = np.array(y1[start: end], dtype="int32")
        batch_y2 = np.array(y2[start: end], dtype="int32")

        yield dict(x1=batch_x1, x2=batch_x2, y1=batch_y1, y2=batch_y2)


def getWordEmbedding(hp, words, model_path):
    wordVec = gensim.models.word2vec.Word2Vec.load(model_path)
    vocab = []
    wordEmbedding = []

    # 添加 "pad" 和 "UNK",
    vocab.append("<PAD>")
    vocab.append("<UNK>")
    wordEmbedding.append(np.zeros(hp.d_model))
    wordEmbedding.append(np.random.randn(hp.d_model))

    for word in words:
        try:
            vector = wordVec.wv[word]
            vocab.append(word)
            wordEmbedding.append(vector)
        except:
            print(word + "不存在于词向量中")

    return vocab, np.array(wordEmbedding)
