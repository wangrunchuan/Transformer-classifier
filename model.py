from modules import *
from flip_gradient import flip_gradient


class Transformer:

    def __init__(self, hp, en_wordEmbedding, de_wordEmbedding):
        self.hp = hp

        if en_wordEmbedding is not None:
            self.en_wordEmbedding = \
                tf.Variable(tf.cast(en_wordEmbedding, dtype=tf.float32, name="en_word2vec"), name="en_weight_mat")
        else:
            self.en_wordEmbedding = get_random_embeddings(self.hp.en_vocab_size, self.hp.d_model, zero_pad=True)

        if de_wordEmbedding is not None:
            self.de_wordEmbedding = \
                tf.Variable(tf.cast(de_wordEmbedding, dtype=tf.float32, name="de_word2vec"), name="de_weight_mat")
        else:
            self.de_wordEmbedding = get_random_embeddings(self.hp.de_vocab_size, self.hp.d_model, zero_pad=True)

        self.embeddedPosition = fixedPositionEmbedding(self.hp.sequence_length)
        # self.inputs = tf.placeholder(tf.int32, shape=(None, hp.sequence_length))
        # self.labels = tf.placeholder(tf.float32, shape=[None])

    def encode(self, x, batch_size, embeddings, training=True, scope_name="encoder"):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # x, seqlens, sents1 = x  # ([None], (), ())。vocab 编码的文本，长度，string 的文本。

            # src_masks
            src_masks = tf.math.equal(x, 0)  # (N, T1)。返回 x == 0 逐元素的真值。防止用于 padding 的 token 影响训练。

            # embedding
            enc = tf.nn.embedding_lookup(embeddings, x)  # (N, T1, d_model)。给定 id list 获得 embedding matrix

            enc *= self.hp.d_model**0.5  # scale

            embeddedPosition = np.expand_dims(self.embeddedPosition, 0).repeat(batch_size, axis=0)
            enc = tf.concat([enc, embeddedPosition], -1)  # (32, 256, 576)

            # enc += positional_encoding(enc, self.hp.sequence_length)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            # memory = enc
            # return memory, src_masks

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope=scope_name + "_multihead_attention")
                    # feed forward
                    # enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model + self.hp.sequence_length])  # d_ff=2048
                    enc = ff(enc, num_units=[self.hp.d_ff, enc.shape[-1]])  # d_ff=2048
        memory = enc
        return memory, src_masks

    def output(self, x, scope_name="fc"):
        l2Loss = 0

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            f = flip_gradient(x, l=1.0)
            # f = x
            output_w = tf.get_variable("output_w",
                                       shape=[x.get_shape()[-1].value, self.hp.num_class],
                                       initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.hp.num_class]), name="output_b")
            logits = tf.nn.xw_plus_b(f, output_w, output_b, name="logits")
            """
            F = tf.nn.xw_plus_b(x, output_w, output_b, name="F")
            print('******************************************')
            print(F.shape)
            f_ = flip_gradient(F, l=1.0)
            print(f_.shape)
            output_w_2 = tf.get_variable("output_w_2",
                                         shape=[f_.get_shape()[-1].value, self.hp.num_class],
                                         initializer=tf.contrib.layers.xavier_initializer())
            output_b_2 = tf.Variable(tf.constant(0.1, shape=[self.hp.num_class]), name="output_b_2")

            logits = tf.nn.xw_plus_b(f_, output_w_2, output_b_2, name="logits")
            """
            y_proba = tf.nn.softmax(logits)
            l2Loss += tf.nn.l2_loss(output_w)
            l2Loss += tf.nn.l2_loss(output_b)

        return logits, y_proba, output_w, l2Loss

    def train(self, x1, x2, y1, y2):
        en_memory, _ = self.encode(x1, self.hp.train_batch_size, self.en_wordEmbedding, scope_name="en_encoder")
        # de_memory, _ = self.encode(x2, self.hp.train_batch_size, self.de_wordEmbedding, scope_name="de_encoder")
        de_memory, _ = self.encode(x2, self.hp.train_batch_size, self.de_wordEmbedding, scope_name="en_encoder")

        en_sentence = tf.reshape(en_memory, [-1, self.hp.sequence_length * en_memory.shape[-1]])
        de_sentence = tf.reshape(de_memory, [-1, self.hp.sequence_length * de_memory.shape[-1]])
        sentence = tf.concat([en_sentence, de_sentence], 0)
        y = tf.concat([y1, y2], 0)

        perm = np.arange(int(sentence.shape[0]))
        np.random.shuffle(perm)

        sentence = tf.gather(sentence, perm, axis=0)
        y = tf.gather(y, perm, axis=0)

        # sentence = tf.transpose(sentence, perm=perm)
        # y = tf.transpose(y, perm=perm)

        logits, y_proba, output_w, l2Loss = self.output(sentence, "fc")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss") + l2Loss * 0.0
        predictions = tf.argmax(logits, axis=-1, name="predictions")

        global_step = tf.train.get_or_create_global_step()
        # lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)  # 学习率
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step, predictions, y, en_memory, de_memory

    def eval(self, x1, x2, y1, y2):
        en_memory, _ = self.encode(x1, self.hp.train_batch_size, self.en_wordEmbedding, scope_name="en_encoder")
        # de_memory, _ = self.encode(x2, self.hp.train_batch_size, self.de_wordEmbedding, scope_name="de_encoder")
        de_memory, _ = self.encode(x2, self.hp.train_batch_size, self.de_wordEmbedding, scope_name="en_encoder")

        en_sentence = tf.reshape(en_memory, [-1, self.hp.sequence_length * en_memory.shape[-1]])
        de_sentence = tf.reshape(de_memory, [-1, self.hp.sequence_length * de_memory.shape[-1]])
        sentence = tf.concat([en_sentence, de_sentence], 0)
        y = tf.concat([y1, y2], 0)

        logits, y_proba, output_w, l2Loss = self.output(sentence, "fc")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

        loss = tf.reduce_mean(xentropy) + l2Loss * self.hp.l2RegLambda
        predictions = tf.argmax(logits, axis=-1)

        return loss, predictions, sentence
