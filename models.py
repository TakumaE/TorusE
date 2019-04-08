import math
import tensorflow as tf


class BasicModel(object):
    def __init__(self, config, nent, nrel):
        super(BasicModel, self).__init__()
        self.config = config
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        self.emb_ent = tf.Variable(tf.random_uniform([nent, config.emb_dim], -0.5, 0.5), name="ent_emb")
        self.emb_rel = tf.Variable(tf.random_uniform([nrel, config.emb_dim], -0.5, 0.5), name="rel_emb")

        pos_he = tf.nn.embedding_lookup(self.emb_ent, self.pos_h)
        pos_re = tf.nn.embedding_lookup(self.emb_rel, self.pos_r)
        pos_te = tf.nn.embedding_lookup(self.emb_ent, self.pos_t)

        neg_he = tf.nn.embedding_lookup(self.emb_ent, self.neg_h)
        neg_re = tf.nn.embedding_lookup(self.emb_rel, self.neg_r)
        neg_te = tf.nn.embedding_lookup(self.emb_ent, self.neg_t)

        pos_score = self.scoring_func(pos_he, pos_re, pos_te)
        neg_score = self.scoring_func(neg_he, neg_re, neg_te)

        # Margin loss
        self.loss = tf.reduce_sum(
            tf.maximum(tf.subtract(tf.add(pos_score, self.config.margin), neg_score), 0.))

        # Testing
        self.r_score = self.scoring_func(pos_he, pos_re, self.emb_ent)
        self.l_score = self.scoring_func(self.emb_ent, pos_re, pos_te)

    def scoring_func(self, h, r, t):
        raise NotImplementedError


class TransE(BasicModel):
    def __init__(self, config, nent, nrel):
        super(TransE, self).__init__(config, nent, nrel)

    def scoring_func(self, h, r, t):
        d = tf.subtract(tf.add(h, r), t)
        if "l1" in self.config.reg:
            return tf.reduce_sum(tf.abs(d), 1)
        else:  # l2
            return tf.reduce_sum(tf.square(d), 1)


class TorusE(BasicModel):
    def __init__(self, config, nent, nrel):
        super(TorusE, self).__init__(config, nent, nrel)

    def scoring_func(self, h, r, t):
        d = tf.subtract(tf.add(h, r), t)
        d = d - tf.floor(d)
        d = tf.minimum(d, 1.0 - d)
        if "el2" in self.config.reg:
            return tf.reduce_sum(2 - 2 * tf.cos(2 * math.pi * d), 1) / 4
        elif "l2" in self.config.reg:
            return 4 * tf.reduce_sum(tf.square(d), 1)
        else:  # l1
            return 2 * tf.reduce_sum(tf.abs(d), 1)
