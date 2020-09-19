from data_load import *
import tensorflow as tf
from model import Transformer
from hparams import Hparams
from utils import *
import math
import numpy as np


hparams = Hparams
parser = hparams.parser
hp = parser.parse_args()

en_train_inputs, _, en_label_to_index, hp.en_vocab_size, en_words = \
    get_data(hp.en_fp, hp, 0, hp.en_pretreat_path)
de_train_inputs, _, de_label_to_index, hp.de_vocab_size, de_words = \
    get_data(hp.de_fp, hp, 1, hp.de_pretreat_path)

en_train_labels = np.array([0] * len(en_train_inputs))
de_train_labels = np.array([1] * len(de_train_inputs))

num_train_batches = calc_num_batches(len(en_train_inputs), hp.train_batch_size)

hp.num_class = 2

en_wordEmbedding = None
if hp.en_pretreat_path is not None:
    _, en_wordEmbedding = getWordEmbedding(hp, en_words, hp.en_w2v_model_path)

de_wordEmbedding = None
if hp.de_pretreat_path is not None:
    _, de_wordEmbedding = getWordEmbedding(hp, de_words, hp.de_w2v_model_path)

m = Transformer(hp, en_wordEmbedding, de_wordEmbedding)

X1 = tf.placeholder(tf.int32, shape=(None, hp.sequence_length), name="X_en")
X2 = tf.placeholder(tf.int32, shape=(None, hp.sequence_length), name="X_de")
y1 = tf.placeholder(tf.int32, shape=[None], name="y_en")
y2 = tf.placeholder(tf.int32, shape=[None], name="y_de")
loss, train_op, global_step, predictions, shuffle_y, en_memory, de_memory = m.train(X1, X2, y1, y2)
eval_loss, eval_prediction, sentence = m.eval(X1, X2, y1, y2)

saver = tf.train.Saver(max_to_keep=hp.epoch)

with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.ckpt_path)
    if ckpt is None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, ckpt)

    for epoch in range(hp.epoch):
        all_prediction = np.array([])
        all_label = np.array([])
        all_loss = 0.0
        print('--------Epoch {}--------'.format(epoch))
        batch_index = 0
        global_step_ = sess.run(global_step)
        for batch in next_batch(en_train_inputs, de_train_inputs, en_train_labels, de_train_labels,
                                batch_size=hp.train_batch_size):
            loss_, train_op_, global_step_, predictions_, shuffle_y_, en_memory_, de_memory_ = \
                sess.run([loss, train_op, global_step, predictions, shuffle_y, en_memory, de_memory],
                         feed_dict={X1: batch['x1'], X2: batch['x2'], y1: batch['y1'], y2: batch['y2']})
            real_epoch = math.ceil(global_step_ / num_train_batches)
            all_prediction = np.hstack((all_prediction, predictions_))
            all_label = np.hstack((all_label, shuffle_y_))
            all_loss += loss_ * len(batch['x1']) * 2

            acc = accuracy(predictions_, shuffle_y_)
            # print(predictions_)
            print('Batch: {}      Loss: {}     Accuracy: {}'.format(batch_index, loss_, acc))
            """
            print(cosine(en_memory_[0][0], de_memory_[0][0]))
            print(cosine(en_memory_[0][0], en_memory_[0][1]))
            print(cosine(en_memory_[0][0], en_memory_[1][0]))
            """
            # if batch_index > 2: break
            batch_index += 1
            if batch_index >= 5: break
        """
        # model_output = "imdb%dL" % real_epoch
        model_output = "iwslt"
        ckpt_name = os.path.join(hp.ckpt_path, model_output)
        saver.save(sess, ckpt_name)
        """
        # print(all_prediction)
        acc = accuracy(all_prediction, all_label)
        print('Loss {}     Accuracy {}'.format(all_loss / len(en_train_inputs) / 2, acc))

        batch_index = 0
        for batch in next_batch(en_train_inputs, de_train_inputs, en_train_labels, de_train_labels,
                                batch_size=hp.train_batch_size):
            eval_loss_, eval_prediction_, sentence_ = \
                sess.run([eval_loss, eval_prediction, sentence],
                         feed_dict={X1: batch['x1'], X2: batch['x2'], y1: batch['y1'], y2: batch['y2']})
            eval_acc = accuracy(eval_prediction_, np.hstack((batch['y1'], batch['y2'])))
            print('Eval Loss {}      Accuracy {}'.format(eval_loss_, eval_acc))
            batch_index += 1
            if batch_index >= 10: break

    sess.close()
