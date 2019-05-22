# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-11-24 14:36:30
# Last modified: 2019-03-02 16:03:33

"""
Evaluate the news
"""

from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import numpy as np


import gel.evaluation.utils as utils


def concate_embeds(path1, path2):
    print('Loading embeddings...{a}'.format(a=path1))
    embeds1 = utils.load_embed(path1)
    print('Loading embeddings...{a}'.format(a=path2))
    embeds2 = utils.load_embed(path2)
    assert len(embeds1) == len(embeds2)
    reval = dict()
    print('Finish loading...')
    for key in embeds1.keys():
        vec = np.concatenate((embeds1[key], embeds2[key]))
        assert len(vec) == 2 * len(embeds1[key])
        reval[key] = vec
    return reval


def eval_news(embed_maps, binaries):
    path = ('../../text-classification/20news/'
            '20news-processed/expand_labels.txt')
    print('Load lable mappings...')
    old_labels_map, expand_labels_map = utils.load_expand_labels(path)

    print('Load 20news...')
    path = ('../../text-classification/20news/'
            '20news-processed/20ng-test-all-terms.txt')
    texty, textx = utils.load_news(path)

    y = [old_labels_map[label] for label in texty]

    labels_embeds = [utils.average(expand_labels_map[i], embed_maps)
                     for i in range(len(expand_labels_map))]
    # for i, vec in enumerate(labels_embeds):
    #     print(i)
    #     print(vec)
    # print('-'*50)
    x = [utils.average(text, embed_maps) for text in textx]

    # for i, vec in enumerate(labels_embeds):
    #     s = '{a}: {b}'.format(a=str(i), b=str(vec))
    #     print(s)
    precs = []
    recalls = []
    f1s = []
    accs = []
    for binary in binaries:
        v1 = expand_labels_map[binary[0]]
        v2 = expand_labels_map[binary[1]]
        newy, newx, new_labels_embeds = filtering(
                binary, y, x, labels_embeds)
        # print(new_labels_embeds)
        preds = predict(newx, new_labels_embeds)
        prec, recall, f1, acc = evaluate(preds, newy)
        precs.append(prec)
        recalls.append(recall)
        f1s.append(f1)
        accs.append(acc)
        s = ('{a} v.s. {b}: prec={c}, recall={d}, f1={e},'
             'accuracy={f}')
        s = s.format(a=str(v1), b=str(v2), c=str(prec),
                     d=str(recall), e=str(f1), f=str(acc))
        print(s)

    s = 'Average precision={a}, recall={b}, f1={c}, acc={d}'
    s = s.format(a=str(np.average(precs)),
                 b=str(np.average(recalls)),
                 c=str(np.average(f1s)),
                 d=str(np.average(accs)))
    print(s)

    # preds = predict(x, labels_embeds)
    # f1 = evaluate(preds, y)


def predict(x: list, labels_embeds: list):
    """
    Args:
        y: gold labels
        x: average embeddings of the text.
        labels_embeds: average embeddings of the label

    Returns:
        accuracy
    """
    reval = []
    assert 2 == len(labels_embeds)
    for i, text_embed in enumerate(x):
        index = utils.nearest_neighbors(
                labels_embeds, text_embed, utils.euclidean_distance)
        reval.append(index)
    return reval


def filtering(binary, y, x, labels_embeds):
    newx = []
    newy = []
    new_label_embeds = []
    for label, text in zip(y, x):
        if label in binary:
            newx.append(text)
            newy.append(binary.index(label))
    for i, vector in enumerate(labels_embeds):
        if i in binary:
            new_label_embeds.append(labels_embeds[i])
    return newy, newx, new_label_embeds


def evaluate(preds, y):
    prec = precision_score(y, preds, average='macro')
    recall = recall_score(y, preds, average='macro')
    f1 = f1_score(y, preds, average='macro')
    acc = accuracy_score(y, preds)
    return prec, recall, f1, acc


if __name__ == '__main__':
    path = '/home/flyaway/scr/word2vec/new_embeddings/ten_extrinsic_subword.txt'
    # path = ('/home/flyaway/scr/FeVER/Experiments/2706/'
    #         'embeddings/context_extrinsic.txt')
    # path1 = ('/home/flyaway/scr/FeVER/Experiments/2589/'
    #          'embeddings_70/label_extrinsic.txt')
    # path2 = ('/home/flyaway/scr/FeVER/Experiments/2589/'
    #          'embeddings_70/context_extrinsic.txt')
    # embeds = concate_embeds(path1, path2)

    embeds = utils.load_embed(path)
    binaries = [(11, 14), (2, 15), (0, 5),
                (3, 13), (6, 12), (1, 18),
                (4, 16), (7, 19), (8, 10), (9, 17)]
    eval_news(embeds, binaries)
