# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-24 14:47:51
# Last modified: 2018-12-07 15:29:31

"""
Common help functions for evaluating.
"""
import numpy as np


def average(text: list, embed_maps: dict):
    vecs = []
    for word in text:
        if word not in embed_maps:
            print(word)
            if 'UNKNOWN' in embed_maps:
                vecs.append(embed_maps['UNKNOWN'])
        else:
            vecs.append(embed_maps[word])
    return np.average(vecs, 0)


def nearest_neighbors(labels_embeds, text_embed, func):
    distances = [func(text_embed, t) for t in labels_embeds]
    return np.argmin(distances)


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1-vec2)


def cos_distance(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return vec1.dot(vec2) / (norm1*norm2)


def load_expand_labels(path: str) -> dict:
    old_labels = dict()
    new_labels = dict()
    with open(path, encoding='utf8') as f:
        for i, line in enumerate(f):
            s = line.strip().split()
            old_labels[s[0]] = i
            new_labels[i] = s[1:]
    return old_labels, new_labels


def load_embed(path: str) -> dict:
    reval = dict()
    with open(path, encoding='utf8') as f:
        next(f)
        for line in f:
            s = line.strip().split()
            word = s[0]
            vecs = np.array([float(t) for t in s[1:]])
            vecs = vecs/np.linalg.norm(vecs, ord=2)
            reval[word] = vecs
    return reval


def load_news(path):
    texty = []
    textx = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            texty.append(s[0])
            textx.append(s[1:])
    return texty, textx


def load_google(path: str):
    reval = []
    s = 'Loading google set: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        for line in f:
            if line.startswith(':'):
                continue
            else:
                s = line.strip().split()
                reval.append(tuple(s))
    return reval


def load_MSR(path: str):
    reval = []
    s = 'Loading MSR set: {a}'.format(a=str(path))
    print(s)
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            a = s[0]
            b = s[1]
            c = s[2]
            d = s[-1]
            reval.append(tuple([a, b, c, d]))
    return reval
