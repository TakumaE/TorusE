from collections import defaultdict
import numpy as np


class KBDataset(object):
    def __init__(self, name):
        self.name = name.lower()
        self.ent_id = {}
        self.rel_id = {}
        self.nent = 0
        self.nrel = 0
        self.train = self.load_file("train")
        self.valid = self.load_file("valid")
        self.test = self.load_file("test")

        self.triples_t = defaultdict(set)
        self.triples_h = defaultdict(set)
        for triple in np.concatenate([self.train, self.valid, self.test]):
            self.triples_t[(triple[0], triple[1])].add(triple[2])
            self.triples_h[(triple[1], triple[2])].add(triple[0])

        self.rel_h = defaultdict(set)
        self.rel_t = defaultdict(set)
        self.train_triples = defaultdict(set)
        for triple in self.train:
            self.rel_h[triple[1]].add(triple[0])
            self.rel_t[triple[1]].add(triple[2])
            self.train_triples[(triple[0], triple[1])].add(triple[2])

    def __repr__(self):
        return ("%s | ent:%d | rel:%d | train:%d | valid:%d | test:%d" %
                (self.name, self.nent, self.nrel,
                 len(self.train), len(self.valid), len(self.test)))

    def load_file(self, filename):
        with open("./data/%s/%s" % (self.name, filename)) as file:
            temp = np.array(file.read().split())
            triples_ = np.zeros(temp.size, dtype=np.int32)
            for i in range(0, temp.size // 3):
                if temp[3 * i] not in self.ent_id:
                    self.ent_id[temp[3 * i]] = self.nent
                    self.nent += 1
                triples_[3 * i] = self.ent_id[temp[3 * i]]
                if temp[3 * i + 2] not in self.ent_id:
                    self.ent_id[temp[3 * i + 2]] = self.nent
                    self.nent += 1
                triples_[3 * i + 2] = self.ent_id[temp[3 * i + 2]]
                if temp[3 * i + 1] not in self.rel_id:
                    self.rel_id[temp[3 * i + 1]] = self.nrel
                    self.nrel += 1
                triples_[3 * i + 1] = self.rel_id[temp[3 * i + 1]]
        return triples_.reshape((-1, 3))
