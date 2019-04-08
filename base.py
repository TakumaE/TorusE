import math
import random
from time import time
import utils as ul
import numpy as np


class LPResult(object):
    def __init__(self):
        self.mrr = 0
        self.mr = 0
        self.h1 = 0
        self.h3 = 0
        self.h10 = 0

    def __repr__(self):
        return "MRR:%.2f | MR:%.0f | HIT1:%.2f | HIT3:%.2f | HIT10:%.2f" % \
               (self.mrr, self.mr, self.h1, self.h3, self.h10)


def negative_sampling(data, i_batch, batch_size):
    pos_batch = data.train[i_batch * batch_size:(i_batch + 1) * batch_size, :]
    neg_batch = data.train[i_batch * batch_size:(i_batch + 1) * batch_size, :].copy()
    for n_triple in neg_batch:
        r = n_triple[1]
        bern = len(data.rel_h[r]) / (len(data.rel_t[r]) + len(data.rel_h[r]))
        if random.random() < bern:
            n_triple[2] = random.randrange(data.nent)
            while n_triple[2] in data.train_triples[(n_triple[0], r)]:
                n_triple[2] = random.randrange(data.nent)
        else:
            n_triple[0] = random.randrange(data.nent)
            while n_triple[2] in data.train_triples[(n_triple[0], r)]:
                n_triple[0] = random.randrange(data.nent)
    return pos_batch, neg_batch


def train(data, model, train_opt, config, sess, saver):
    if config.restore:
        print("Load Model: %s" % config.restore)
        saver.restore(sess, config.restore)
    else:
        print("Start training: ")
        rows, _ = data.train.shape
        batch_size = math.ceil(rows / config.nbatches)
        best_mrr = 0.
        best_epoch = -1
        last_valid = 0.
        bad_valid = 0
        for i in range(config.epoch):
            all_loss = 0.
            start_time = time()
            np.random.shuffle(data.train)
            for j in range(config.nbatches):
                pos_batch, neg_batch = negative_sampling(data, j, batch_size)
                _, batch_loss = sess.run([train_opt, model.loss],
                                         feed_dict={model.pos_h: pos_batch[:, 0],
                                                    model.pos_r: pos_batch[:, 1],
                                                    model.pos_t: pos_batch[:, 2],
                                                    model.neg_h: neg_batch[:, 0],
                                                    model.neg_r: neg_batch[:, 1],
                                                    model.neg_t: neg_batch[:, 2]})
                all_loss += batch_loss
            message = "Epoch %4d\tLoss:%.2f\t%.2fs" % (i, all_loss, time() - start_time)

            if (i + 1) % config.save_steps == 0:
                save_dir = "checkpoints/%s/%d/%s.ckpt" % (data.name, i, config.model.lower())
                ul.save_model(saver, sess, save_dir)
                message += "\tCheckpoint Saved "
            print(message)

            # Validate
            if (i + 1) % config.valid_steps == 0:
                i_mrr = valid(data, model, sess)
                if i_mrr > best_mrr:
                    best_mrr = i_mrr
                    best_epoch = i

                if i_mrr >= last_valid:
                    bad_valid = 0
                    last_valid = i_mrr
                else:
                    bad_valid += 1

                # Early stopping
                if bad_valid >= config.early_stopping:
                    print("Early stopping at epoch %d" % i)
                    break

        if best_epoch > 0:
            # Load best session
            print("Save best model in epoch %d with Valid(Mrr): %.4f" % (best_epoch, best_mrr))
            save_dir = "checkpoints/%s/%d/%s.ckpt" % (data.name, best_epoch, config.model.lower())
            saver.restore(sess, save_dir)
        else:
            print("Save the last model in epoch %d" % (config.epoch - 1))
        ul.save_model(saver, sess, config.save_dir)


def valid(data, model, sess):
    mrr = 0.
    valid_rows, _ = data.valid.shape
    # valid_rows = 10

    time_list = []
    for i in range(valid_rows):
        start = time()
        temp_h = data.valid[i, 0]
        temp_r = data.valid[i, 1]
        temp_t = data.valid[i, 2]

        res_score = sess.run(model.r_score, feed_dict={model.pos_h: [temp_h],
                                                       model.pos_r: [temp_r]})
        res_score = np.argsort(res_score)
        temp_hr = data.triples_t[(temp_h, temp_r)]
        rank = 0
        for j in range(data.nent):
            if res_score[j] == data.valid[i, 2]:
                mrr += 1 / (rank + 1)
                break
            if len(temp_hr) == 0 or res_score[j] not in temp_hr:
                rank += 1

        res_score = sess.run(model.l_score, feed_dict={model.pos_r: [temp_r],
                                                       model.pos_t: [temp_t]})
        res_score = np.argsort(res_score)
        temp_rt = data.triples_h[(temp_r, temp_t)]
        rank = 0
        for j in range(data.nent):
            if res_score[j] == data.valid[i, 0]:
                mrr += 1 / (rank + 1)
                break
            if len(temp_rt) == 0 or res_score[j] not in temp_rt:
                rank += 1
        time_list.append(time() - start)
        ul.print_progress(i, valid_rows, "Valid - %.2fs/triple" % np.mean(time_list))
    mrr = mrr / valid_rows / 2.
    print("\tMRR: %f" % mrr)
    return mrr


def test(data, model, sess):
    res_l = LPResult()
    res_r = LPResult()
    res_fl = LPResult()
    res_fr = LPResult()
    test_rows, test_cols = data.test.shape
    time_list = []
    for i in range(test_rows):
        start = time()
        temp_h = data.test[i, 0]
        temp_r = data.test[i, 1]
        temp_t = data.test[i, 2]

        temp_hr = data.triples_t[(temp_h, temp_r)]
        res_score = sess.run(model.r_score, feed_dict={model.pos_h: [temp_h],
                                                       model.pos_r: [temp_r]})
        res_score = np.argsort(res_score)
        rank = 0.
        for j in range(data.nent):
            if res_score[j] == temp_t:
                if rank < 10:
                    res_fr.h10 += 1
                if rank < 3:
                    res_fr.h3 += 1
                if rank < 1:
                    res_fr.h1 += 1
                res_fr.mr += rank + 1
                res_fr.mrr += 1 / (rank + 1)

                if j < 10:
                    res_r.h10 += 1
                if j < 3:
                    res_r.h3 += 1
                if j < 1:
                    res_r.h1 += 1
                res_r.mr += j + 1
                res_r.mrr += 1 / (j + 1)
                break
            if len(temp_hr) == 0 or res_score[j] not in temp_hr:
                rank += 1

        res_score = sess.run(model.l_score, feed_dict={model.pos_r: [temp_r],
                                                       model.pos_t: [temp_t]})
        res_score = np.argsort(res_score)
        temp_rt = data.triples_h[(temp_r, temp_t)]
        rank = 0.
        for j in range(data.nent):
            if res_score[j] == temp_h:
                if rank < 10:
                    res_fl.h10 += 1
                if rank < 3:
                    res_fl.h3 += 1
                if rank < 1:
                    res_fl.h1 += 1
                res_fl.mr += rank + 1
                res_fl.mrr += 1 / (rank + 1)

                if j < 10:
                    res_l.h10 += 1
                if j < 3:
                    res_l.h3 += 1
                if j < 1:
                    res_l.h1 += 1
                res_l.mr += j + 1
                res_l.mrr += 1 / (j + 1)
                break
            if len(temp_rt) == 0 or res_score[j] not in temp_rt:
                rank += 1
        time_list.append(time() - start)
        ul.print_progress(i, test_rows, "Test - %.2fs/triple" % np.mean(time_list))

    r_mrr = (res_fl.mrr + res_fr.mrr) / test_rows / 2
    r_mr = (res_fl.mr + res_fr.mr) / test_rows / 2
    r_h1 = (res_fl.h1 + res_fr.h1) / test_rows / 2
    r_h3 = (res_fl.h3 + res_fr.h3) / test_rows / 2
    r_h10 = (res_fl.h10 + res_fr.h10) / test_rows / 2
    print("\nMRR\tMR\tHit1\tHit3\tHit10")
    print("%.4f\t%.0f\t%.4f\t%.4f\t%.4f" % (r_mrr, r_mr, r_h1, r_h3, r_h10))
    return res_fl, res_fr, res_l, res_r
