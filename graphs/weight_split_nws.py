# -*- coding: utf-8 -*-
# @author: Gabriel Eilertsen
"""
The paper related to the repo
@article{EJRUY20,
         author  = "Eilertsen, Gabriel
          and J\"onsson, Daniel
           and Ropinski, Timo and Unger, Jonas and Ynnerman, Anders",
         title   = "Classifying the classifier: dissecting
                    the weight space of neural networks",
         journal = "Proceedings of the European Conference
                    on Artificial Intelligence (ECAI 2020)",
         volume  = "325",
         pages   = "1119--1126",
         year    = "2020"}
"""

import numpy as np


def unpack(x, fsize, ldepth, lwidth, bn=True):
    """

    :param x: this is the theta in the paper, i.e., the list of all trainable
              parameters as defined in Eq.(1-2) of the paper
              (concatenation of all the weights)
    :param fsize: filter size
    :param ldepth: [meta["depth_conv"][0], meta["depth_fc"][0]] a list
                   with the depth of the convolution layers and the depth of
                   the fully-connected layers of the CNN architecture
    :param lwidth: [meta["width_conv"][0], meta["width_fc"][0]] a list
                   with the width of the convolution layers and the width of
                   the fully-connected layers of the CNN architecture
    :param bn:
    :return: a tuple (sout_conv, sout_fc) of unpacked weights.
    Here, sout_conv and sout_fc are lists with the separate layer weights for
    convolutional and fully connected layers, respectively. fc holds
    layers through the first index, i.e. fc[0,:] are the weights for
    the first fully connected layer, where fc[0,0] is the weight matrix,
     fc[0,1] is the bias vector, and fc[0,2:] are weights for batch
    normalization.

    E.g.: A weight matrix list is obtained with: fc[:, 0]

    Short documentation from Vincent Thibeault and Gabriel Eilertsen.
    """
    B = 3
    ind = np.array([0, 0])

    mlt = 1
    if bn:
        mlt = 5

    sout_conv = np.empty((ldepth[0], mlt + 1), dtype=object)
    sout_fc = np.empty((ldepth[1] + 1, mlt + 1), dtype=object)

    blockc = (np.floor(ldepth[0] / B) * np.ones(B)).astype('int64')
    for c in range(B - 1, 0, -1):
        if sum(blockc) < ldepth[0]:
            blockc[c] += 1

    assert (sum(blockc) == ldepth[0])

    cc = 0
    sz = np.array([32, 32, 3], dtype='int64')
    for c in range(B):
        w = lwidth[0] * np.power(2, c)

        # Unpack convolutional filters
        ind[1] += fsize * fsize * sz[2] * w
        sout_conv[cc, 0] = np.transpose(
            np.reshape(x[ind[0]:ind[1]], [w, sz[2], fsize, fsize]),
            [2, 3, 1, 0])
        ind[0] = ind[1]

        # bias and BN weights
        for mm in range(mlt):
            ind[1] += w
            sout_conv[cc, mm + 1] = x[ind[0]:ind[1]]
            ind[0] = ind[1]

        sz[2] = w
        cc += 1

        for b in range(blockc[c] - 1):
            # Unpack convolutional filters
            ind[1] += fsize * fsize * w * w
            sout_conv[cc, 0] = np.transpose(
                np.reshape(x[ind[0]:ind[1]], [w, w, fsize, fsize]),
                [2, 3, 1, 0])
            ind[0] = ind[1]

            # bias and BN weights
            for mm in range(mlt):
                ind[1] += w
                sout_conv[cc, mm + 1] = x[ind[0]:ind[1]]
                ind[0] = ind[1]

            sz[2] = w
            cc += 1

        sz[:2] = sz[:2] / 2

    print('%d conv layers splitted (%d in total)' % (cc, ldepth[0]))

    sz = np.prod(sz)

    cc = 0
    for c in range(ldepth[1]):
        w = (lwidth[1] * np.power(2.0, -c)).astype('int64')

        # multiplicative weights
        ind[1] += sz * w
        sout_fc[cc, 0] = np.transpose(np.reshape(x[ind[0]:ind[1]], [w, sz]),
                                      [1, 0])
        ind[0] = ind[1]

        # bias and BN weights
        for mm in range(mlt):
            ind[1] += w
            sout_fc[cc, mm + 1] = x[ind[0]:ind[1]]
            ind[0] = ind[1]

        sz = w
        cc += 1

    # output layer (not counted in the specified number of layers)
    w = 20
    ind[1] += sz * w
    sout_fc[cc, 0] = np.transpose(np.reshape(x[ind[0]:ind[1]], [w, sz]),
                                  [1, 0])
    ind[0] = ind[1]

    ind[1] += w
    sout_fc[cc, 1] = x[ind[0]:ind[1]]
    ind[0] = ind[1]

    cc += 1

    print('%d FC layers splitted (%d in total)' % (cc, ldepth[1]))

    assert (ind[0] == len(x))

    return sout_conv, sout_fc


def pack(sout_conv, sout_fc):
    x = []

    for i in range(sout_conv.shape[0]):
        for j in range(sout_conv.shape[1]):
            if len(sout_conv[i, j].shape) > 1:
                w = np.reshape(np.transpose(sout_conv[i, j], [3, 2, 0, 1]),
                               [sout_conv[i, j].size])
            else:
                w = sout_conv[i, j]
            x = np.concatenate([x, w])

    for i in range(sout_fc.shape[0]):
        for j in range(sout_fc.shape[1]):
            if np.sum(sout_fc[i, j]) is not None:
                if len(sout_fc[i, j].shape) > 1:
                    w = np.reshape(np.transpose(sout_fc[i, j], [1, 0]),
                                   [sout_fc[i, j].size])
                else:
                    w = sout_fc[i, j]
                x = np.concatenate([x, w])

    return x
