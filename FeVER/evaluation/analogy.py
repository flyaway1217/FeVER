# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-12-07 13:44:34
# Last modified: 2019-03-02 15:47:18

"""
Our own implementation of analogy.
"""

import os
# import multiprocessing


import numpy as np
from tqdm import tqdm
import torch

import gel.evaluation.utils as utils


def mapping(embeds):
    word2id = dict()
    id2word = dict()
    matrix = []
    for word, embed in embeds.items():
        word2id[word] = len(word2id)
        matrix.append(embed)
    matrix = np.array(matrix)
    return matrix, word2id, id2word


def analogy_test(embeds, pred_embeds, test_pairs):

    matrix, _, _ = mapping(embeds)
    pred_matrix, pred_word2id, pred_id2word = mapping(pred_embeds)
    mean = np.mean(matrix, axis=0)

    pred_matrix = torch.from_numpy(pred_matrix).cuda()
    pred_id2word = {v: k for k, v in pred_word2id.items()}

    answer = [pair[3] for pair in test_pairs]

    batch_size = 200
    scores = []
    for index in tqdm(range(0, len(test_pairs), batch_size)):
        batch = test_pairs[index:index+batch_size]
        scores += _analogy(
                embeds, pred_matrix, mean, pred_word2id,
                pred_id2word, batch)

    assert len(scores) == len(answer)
    scores = [pred == ans for pred, ans in zip(scores, answer)]
    return sum(scores) / len(scores)


def _analogy(embeds, matrix, mean, word2id, id2word, test_pairs):
    A = [embeds.get(pair[0], mean) for pair in test_pairs]
    A = np.array(A)
    A = torch.from_numpy(A).cuda()

    B = [embeds.get(pair[1], mean) for pair in test_pairs]
    B = np.array(B)
    B = torch.from_numpy(B).cuda()

    C = [embeds.get(pair[2], mean) for pair in test_pairs]
    C = np.array(C)
    C = torch.from_numpy(C).cuda()

    D = torch.matmul(matrix, (B-A+C).transpose(0, 1))

    a = []
    b = []
    for index, row in enumerate(test_pairs):
        aa = [word2id[r] for r in row[:3] if r in word2id]
        bb = [index] * len(aa)
        a += aa
        b += bb
    value = torch.DoubleTensor([float('-inf')] * len(a)).cuda()
    D[a, b] = value
    output = [id2word[i] for i in D.argmax(dim=0).cpu().numpy()]
    assert len(output) == len(test_pairs)
    return output


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


def average_embeds(path1, path2):
    print('Loading embeddings...{a}'.format(a=path1))
    embeds1 = utils.load_embed(path1)
    print('Loading embeddings...{a}'.format(a=path2))
    embeds2 = utils.load_embed(path2)
    assert len(embeds1) == len(embeds2)
    reval = dict()
    print('Finish loading...')
    for key in embeds1.keys():
        vec = (embeds1[key] + embeds2[key])
        assert len(vec) == len(embeds1[key])
        reval[key] = vec
    return reval


if __name__ == '__main__':
    google_common = '/home/flyaway/scr/web_data/analogy/EN-GOOGLE/'
    google_paths = {
            'google': 'EN-GOOGLE.txt',
            'google-1': 'EN-GOOGLE-degree1.txt',
            'google-2': 'EN-GOOGLE-degree2.txt',
            'google-3': 'EN-GOOGLE-degree3.txt'
            }
    msr_common = '/home/flyaway/scr/web_data/analogy/EN-MSR/'
    msr_paths = {
            'MSR': 'EN-MSR.txt',
            'MSR-1': 'EN-MSR-degree1.txt',
            'MSR-2': 'EN-MSR-degree2.txt',
            'MSR-3': 'EN-MSR-degree3.txt',
            }

    eval_list = [
            'google',
            # 'google-1',
            # 'google-2',
            # 'google-3',
            'MSR',
            # 'MSR-1',
            # 'MSR-2',
            # 'MSR-3'
            ]
    # ten_corpus_cbow: 0.46325
    # embed_path = ('/home/flyaway/scr/word2vec/new_embeddings'
    #               '/ten_typo_rare_subword.txt')
    # pred_embed_path = ('/home/flyaway/scr/word2vec/new_embeddings'
    #                    '/hundred_intrinsic_subword.txt')

    # embed_path = ('/home/flyaway/scr/FeVER/Experiments/'
    #               '2589/embeddings_70/context_ten_embeds.txt')
    # pred_embed_path = ('/home/flyaway/scr/FeVER/Experiments/'
    #                    '2589/embeddings_70/label_ten_embeds.txt')

    embed_path1 = ('/home/flyaway/scr/FeVER/Experiments/'
                   '2588/embeddings_70/label_ten_embeds.txt')
    embed_path2 = ('/home/flyaway/scr/FeVER/Experiments/'
                   '2588/embeddings_70/context_ten_embeds.txt')

    # pred_embed_path1 = ('/home/flyaway/scr/FeVER/Experiments/'
    #                     '2706/embeddings/label_hundred_intrinsic.txt')
    # pred_embed_path2 = ('/home/flyaway/scr/FeVER/Experiments/'
    #                     '2706/embeddings/context_hundred_intrinsic.txt')

    embeds = concate_embeds(embed_path1, embed_path2)
    # embeds = average_embeds(embed_path1, embed_path2)

    # pred_embeds = concate_embeds(pred_embed_path1, pred_embed_path2)
    # print('Loading embeddings...{a}'.format(a=embed_path))
    # embeds = utils.load_embed(embed_path)
    # print('Loading embeddings...{a}'.format(a=pred_embed_path))
    # pred_embeds = utils.load_embed(pred_embed_path)

    pred_embeds = embeds
    scores = []
    for name in eval_list:
        print('Evaluating {a}'.format(a=name))
        if 'google' in name:
            path = os.path.join(google_common,
                                google_paths[name])
            test_data = utils.load_google(path)
        else:
            path = os.path.join(msr_common,
                                msr_paths[name])
            test_data = utils.load_MSR(path)
        s = analogy_test(embeds, pred_embeds, test_data)
        print(name+': '+format(s, '0.4f'))
        scores.append(s)

    s = '\t'.join(eval_list)
    print(s)
    s = '\t'.join([format(t, '0.4f') for t in scores])
    print(s)
