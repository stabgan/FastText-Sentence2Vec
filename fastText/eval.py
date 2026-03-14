#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import math
import os

import numpy as np
from scipy import stats


def compat_splitting(line):
    return line.decode('utf8').split()


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.dot(v1, v2) / n1 / n2


def load_vectors(model_path):
    vectors = {}
    with open(model_path, 'rb') as fin:
        for line in fin:
            try:
                tab = compat_splitting(line)
                vec = np.array(tab[1:], dtype=float)
                word = tab[0]
                if np.linalg.norm(vec) == 0:
                    continue
                if word not in vectors:
                    vectors[word] = vec
            except ValueError:
                continue
            except UnicodeDecodeError:
                continue
    return vectors


def evaluate(vectors, data_path):
    mysim = []
    gold = []
    drop = 0.0
    nwords = 0.0

    with open(data_path, 'rb') as fin:
        for line in fin:
            tline = compat_splitting(line)
            word1 = tline[0].lower()
            word2 = tline[1].lower()
            nwords += 1.0

            if (word1 in vectors) and (word2 in vectors):
                v1 = vectors[word1]
                v2 = vectors[word2]
                d = similarity(v1, v2)
                mysim.append(d)
                gold.append(float(tline[2]))
            else:
                drop += 1.0

    if nwords == 0:
        print("No words found in data file.")
        return

    corr = stats.spearmanr(mysim, gold)
    dataset = os.path.basename(data_path)
    print(
        "{0:20s}: {1:2.0f}  (OOV: {2:2.0f}%)"
        .format(dataset, corr[0] * 100, math.ceil(drop / nwords * 100.0))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate word vectors using Spearman correlation.'
    )
    parser.add_argument(
        '--model',
        '-m',
        dest='modelPath',
        action='store',
        required=True,
        help='path to model'
    )
    parser.add_argument(
        '--data',
        '-d',
        dest='dataPath',
        action='store',
        required=True,
        help='path to data'
    )
    args = parser.parse_args()

    vectors = load_vectors(args.modelPath)
    evaluate(vectors, args.dataPath)
