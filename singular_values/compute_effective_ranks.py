# -​*- coding: utf-8 -*​-
# @author: Antoine Allard <antoineallard.info> and Vincent Thibeault

import glob
import numpy as np
import os
import pandas as pd
import tabulate
from singular_values.optimal_shrinkage import optimal_shrinkage,\
    optimal_threshold
from tqdm import tqdm
import networkx as nx
from scipy.linalg import svdvals
import time
import json
import tkinter.simpledialog
from tkinter import messagebox
import collections.abc


def to_fwf(df, fname, cols=None):
    """Custom method 'to_fwf' for Pandas

    Parameters
    ----------
    df : dataframe
    fname : str
        The path to the new file in which the hidden parameters will be written
    cols : list or array of strings, optional
        A list or an array containing the name of the columns (as strings) to
        be written in the file (if None all columns will be written).
    """

    if cols is None:
        cols = df.columns

    header = list(cols)
    header[0] = '# ' + header[0]
    content = tabulate.tabulate(df[cols].values.tolist(), header,
                                tablefmt='plain', stralign='right',
                                colalign=('left',))
    open(fname, 'w').write(content)


# Adds the custom method to Pandas.
pd.DataFrame.to_fwf = to_fwf


def computeRank(singularValues, tolerance=1e-13):
    return len(singularValues[singularValues > tolerance])


def computeERank(singularValues, tolerance=1e-13):
    """Effective rank based on the definition using spectral entropy
     (https://ieeexplore.ieee.org/document/7098875
      and doi:10.1186/1745-6150-2-2). """
    # We use the convention 0*log(0)=0 so we remove the zero singular values
    singularValues = singularValues[singularValues > tolerance]
    normalizedSingularValues = singularValues / np.sum(singularValues)
    return np.exp(-np.sum(normalizedSingularValues
                          * np.log(normalizedSingularValues)))


def findEffectiveRankElbow(singularValues):
    """Effective rank based on the elbow method."""
    # Coordinates of the diagonal line y = 1 - x  using ax + by + c = 0.
    a, b, c = 1, 1, -1

    # Define normalized axis with first SV at (0,1) and last SV at (1,0).
    x = np.linspace(0, 1, num=len(singularValues))
    y = (singularValues - np.min(singularValues)) /\
        (np.max(singularValues) - np.min(singularValues))

    # See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # Line_defined_by_an_equation
    # Distance between the diagonal line y = 1 - x, passing through the
    # largest and the smallest singular value, and the position (x_i, y_i)
    # of the i-th singular value
    distanceToDiagonal = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

    # Returns the index of the largest distance (rank must be larger than 0).
    elbowPosition = np.argmax(distanceToDiagonal) + 1  # + 1 for indices
    return elbowPosition - 1
    # - 1 to have the effective rank/nb of significant singvals


def computeEffectiveRankEnergyRatio(singularValues, threshold=0.9):
    """Effective rank based on the energy ratio."""
    normalizedCumulSquaredSingularValues = np.cumsum(np.square(singularValues))
    normalizedCumulSquaredSingularValues /= \
        normalizedCumulSquaredSingularValues[-1]
    # Below, this is the min of the argmax. See the note in the documentation
    # of np.argmax: "In case of multiple occurrences of the maximum values,
    # the indices corresponding to the first occurrence are returned."
    return np.argmax(normalizedCumulSquaredSingularValues > threshold) + 1


def computeStableRank(singularValues):
    return np.sum(singularValues*singularValues) / np.max(singularValues)**2


def computeNuclearRank(singularValues):
    return np.sum(singularValues) / np.max(singularValues)


def computeOptimalThreshold(singularValues):
    """ Optimal threshold for a given norm for a square matrix with gaussian
     noise (see Gavish, Donoho, 2014) """
    return optimal_threshold(singularValues, 1)


def computeOptimalShrinkage(singularValues, norm="frobenius", tolerance=1e-13):
    """ Optimal shrinkage for a given norm ('frobenius', 'nuclear', 'operator')
     for a square matrix with gaussian noise (see Gavish, Donoho, 2017 and
     William Leeb, 2022 for the operator norm). """
    shrinked_singvals = optimal_shrinkage(singularValues, 1, norm)
    return len(shrinked_singvals[shrinked_singvals > tolerance])


def computeEffectiveRanks(singularValues, matrixName, size):
    """
    Computes the rank and various effective ranks and add them to a dataframe.
    :param singularValues: Singular values of a matrix
    :param matrixName: (str) Name of the matrix for which we have computed the
                             singular values
    :param size: Size of the matrix
    :return Table with the name of the matrix, its size, rank, and effective
            ranks.
    """
    header = ['Name', 'Size', 'Rank', 'Optimal threshold', 'Optimal shrinkage',
              'Erank', 'Elbow', 'Energy ratio', 'Stable rank', 'Nuclear rank']
    properties = [[matrixName, size,
                  computeRank(singularValues),
                  computeOptimalThreshold(singularValues),
                  computeOptimalShrinkage(singularValues),
                  computeERank(singularValues),
                  findEffectiveRankElbow(singularValues),
                  computeEffectiveRankEnergyRatio(singularValues,
                                                  threshold=0.9),
                  computeStableRank(singularValues),
                  computeNuclearRank(singularValues)]]
    return tabulate.tabulate(properties, headers=header)


def computeEffectiveRanksRandomGraphs(generator, args, nb_graphs=1000):

    graph_str = generator.__name__
    singularValues = np.array([])
    effectiveRanks = np.zeros((8, nb_graphs))
    for i in tqdm(range(0, nb_graphs)):
        if graph_str in ["tenpy_random_matrix", "perturbed_gaussian"]:
            W = generator(*args)
        else:
            W = nx.to_numpy_array(generator(*args))
        singularValues_instance = svdvals(W)
        effectiveRanks_instance = \
            np.array([computeRank(singularValues_instance),
                      computeOptimalThreshold(singularValues_instance),
                      computeOptimalShrinkage(singularValues_instance),
                      computeERank(singularValues_instance),
                      findEffectiveRankElbow(singularValues_instance),
                      computeEffectiveRankEnergyRatio(singularValues_instance,
                                                      threshold=0.9),
                      computeStableRank(singularValues_instance),
                      computeNuclearRank(singularValues_instance)])

        singularValues = np.concatenate((singularValues,
                                         singularValues_instance))
        effectiveRanks[:, i] = effectiveRanks_instance
    if messagebox.askyesno("Python", "Would you like to save the"
                                     " parameters and the data?"):
        window = tkinter.Tk()
        window.withdraw()  # hides the window
        file = tkinter.simpledialog.askstring("File: ",
                                              "Enter your file name")
        args_list = list(args)
        for i, arg in enumerate(args_list):
            if not isinstance(arg, (collections.abc.Sequence, float, int)):
                args_list[i] = args_list[i].tolist()
        path = "C:/Users/thivi/Documents/GitHub/" \
               "low-rank-hypothesis-complex-systems/" \
               "singular_values/properties/singular_values_random_graphs/"
        order = "rank, thrank, shrank, erank, elbow, energy, srank, nrank"
        timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
        parameters_dictionary = {"graph_str": graph_str,
                                 "args_generator": tuple(args_list),
                                 "nb_graphs": nb_graphs,
                                 "order rank and effective"
                                 " ranks in effectiveRanks"
                                 " (shape (8, nb_graphs))": order
                                 }

        with open(path + f'{timestr}_{file}_concatenated_singular_values'
                         f'_{graph_str}.json', 'w') \
                as outfile:
            json.dump(singularValues.tolist(), outfile)
        with open(path + f'{timestr}_{file}_concatenated_effective_ranks'
                         f'_{graph_str}.json', 'w') \
                as outfile:
            json.dump(effectiveRanks.tolist(), outfile)
        with open(path + f'{timestr}_{file}_singular_values_histogram'
                         f'_{graph_str}_parameters_dictionary.json',
                  'w') as outfile:
            json.dump(parameters_dictionary, outfile)


def computeEffectiveRanksManyNetworks(recompute=True):
    """ Computes the rank and various effective ranks
    and add them to a dataframe for many real networks.

    recompute: (bool) If we want to recompute the effective ranks of the
                      networks from Netzschleuder that
                      are already in the dataframe effective_rank.txt, it is
                      True. Else, it is False and it helps to save computation
                      time.
    """

    graphPropFilename = 'C:/Users/thivi/Documents/GitHub/' \
                        'low-rank-hypothesis-complex-systems/' \
                        'graphs/graph_data/graph_properties.txt'
    header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
    graphPropDF = pd.read_table(graphPropFilename, names=header,
                                comment="#", delimiter=r"\s+")
    graphPropDF.set_index('name', inplace=True)

    effectiveRanksFilename = 'properties/effective_ranks.txt'
    if not os.path.isfile(effectiveRanksFilename):
        header = ['Name', 'Size', 'Rank', 'OptimalThreshold',
                  'OptimalShrinkage', 'Erank', 'Elbow',
                  'EnergyRatio', 'StableRank', 'NuclearRank']
        effectiveRanksDF = pd.DataFrame(columns=header)
        effectiveRanksList = []
    else:
        header = open(effectiveRanksFilename, 'r')\
            .readline().replace('#', ' ').split()
        effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                         comment="#", delimiter=r"\s+")
        effectiveRanksList = effectiveRanksDF.values.tolist()

    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None):
    #     print(effectiveRanksDF["Name"])
    drop_board_directors_list = ["board_directors_net1m_2002-06-01", "board_directors_net1m_2002-07-01", "board_directors_net1m_2002-08-01", "board_directors_net1m_2002-09-01", "board_directors_net1m_2002-10-01", "board_directors_net1m_2002-11-01", "board_directors_net1m_2003-02-01", "board_directors_net1m_2003-03-01", "board_directors_net1m_2003-04-01", "board_directors_net1m_2003-05-01", "board_directors_net1m_2003-06-01", "board_directors_net1m_2003-07-01", "board_directors_net1m_2003-08-01", "board_directors_net1m_2003-09-01", "board_directors_net1m_2003-10-01", "board_directors_net1m_2003-11-01", "board_directors_net1m_2004-02-01", "board_directors_net1m_2004-03-01", "board_directors_net1m_2004-04-01", "board_directors_net1m_2004-05-01", "board_directors_net1m_2004-06-01", "board_directors_net1m_2004-07-01", "board_directors_net1m_2004-08-01", "board_directors_net1m_2004-09-01", "board_directors_net1m_2004-10-01", "board_directors_net1m_2004-11-01", "board_directors_net1m_2005-02-01", "board_directors_net1m_2005-03-01", "board_directors_net1m_2005-04-01", "board_directors_net1m_2005-05-01", "board_directors_net1m_2005-06-01", "board_directors_net1m_2005-07-01", "board_directors_net1m_2005-08-01", "board_directors_net1m_2005-09-01", "board_directors_net1m_2005-10-01", "board_directors_net1m_2005-11-01", "board_directors_net1m_2006-02-01", "board_directors_net1m_2006-03-01", "board_directors_net1m_2006-04-01", "board_directors_net1m_2006-05-01", "board_directors_net1m_2006-06-01", "board_directors_net1m_2006-07-01", "board_directors_net1m_2006-08-01", "board_directors_net1m_2006-09-01", "board_directors_net1m_2006-10-01", "board_directors_net1m_2006-11-01", "board_directors_net1m_2007-02-01", "board_directors_net1m_2007-03-01", "board_directors_net1m_2007-04-01", "board_directors_net1m_2007-05-01", "board_directors_net1m_2007-06-01", "board_directors_net1m_2007-07-01", "board_directors_net1m_2007-08-01", "board_directors_net1m_2007-09-01", "board_directors_net1m_2007-10-01", "board_directors_net1m_2007-11-01", "board_directors_net1m_2008-02-01", "board_directors_net1m_2008-03-01", "board_directors_net1m_2008-04-01", "board_directors_net1m_2008-05-01", "board_directors_net1m_2008-06-01", "board_directors_net1m_2008-07-01", "board_directors_net1m_2008-08-01", "board_directors_net1m_2008-09-01", "board_directors_net1m_2008-10-01", "board_directors_net1m_2008-11-01", "board_directors_net1m_2009-02-01", "board_directors_net1m_2009-03-01", "board_directors_net1m_2009-04-01", "board_directors_net1m_2009-05-01", "board_directors_net1m_2009-06-01", "board_directors_net1m_2009-07-01", "board_directors_net1m_2009-08-01", "board_directors_net1m_2009-09-01", "board_directors_net1m_2009-10-01", "board_directors_net1m_2009-11-01", "board_directors_net1m_2010-02-01", "board_directors_net1m_2010-03-01", "board_directors_net1m_2010-04-01", "board_directors_net1m_2010-05-01", "board_directors_net1m_2010-06-01", "board_directors_net1m_2010-07-01", "board_directors_net1m_2010-08-01", "board_directors_net1m_2010-09-01", "board_directors_net1m_2010-10-01", "board_directors_net1m_2010-11-01", "board_directors_net1m_2011-02-01", "board_directors_net1m_2011-03-01", "board_directors_net1m_2011-04-01", "board_directors_net1m_2011-05-01", "board_directors_net1m_2011-06-01", "board_directors_net1m_2011-07-01", "board_directors_net1m_2011-08-01"]
    drop_wiki_list = ["edit_wikibooks_af", "edit_wikibooks_ak", "edit_wikibooks_als", "edit_wikibooks_ang", "edit_wikibooks_as", "edit_wikibooks_ast", "edit_wikibooks_bg", "edit_wikibooks_bi", "edit_wikibooks_bm", "edit_wikibooks_bn", "edit_wikibooks_bo", "edit_wikibooks_ch", "edit_wikibooks_co", "edit_wikibooks_cs", "edit_wikibooks_cv", "edit_wikibooks_el", "edit_wikibooks_eo", "edit_wikibooks_et", "edit_wikibooks_fi", "edit_wikibooks_got", "edit_wikibooks_gu"     , "edit_wikibooks_he", "edit_wikibooks_hi", "edit_wikibooks_hy", "edit_wikibooks_ia", "edit_wikibooks_id", "edit_wikibooks_is", "edit_wikibooks_kk", "edit_wikibooks_km", "edit_wikibooks_kn", "edit_wikibooks_ko", "edit_wikibooks_ks", "edit_wikibooks_ku", "edit_wikibooks_lb", "edit_wikibooks_li", "edit_wikibooks_ln", "edit_wikibooks_lt", "edit_wikibooks_lv", "edit_wikibooks_mg", "edit_wikibooks_mi"     , "edit_wikibooks_mk"     , "edit_wikibooks_ml"     , "edit_wikibooks_mn"     , "edit_wikibooks_mr"     , "edit_wikibooks_ms", "edit_wikibooks_nah"    , "edit_wikibooks_nds"    , "edit_wikibooks_ne", "edit_wikibooks_pa", "edit_wikibooks_ps", "edit_wikibooks_qu", "edit_wikibooks_rm", "edit_wikibooks_ro", "edit_wikibooks_ru", "edit_wikibooks_sa", "edit_wikibooks_se", "edit_wikibooks_simple", "edit_wikibooks_sk", "edit_wikibooks_sl", "edit_wikibooks_sq", "edit_wikibooks_su", "edit_wikibooks_sv", "edit_wikibooks_sw", "edit_wikibooks_ta", "edit_wikibooks_te", "edit_wikibooks_th", "edit_wikibooks_tk", "edit_wikibooks_tl", "edit_wikibooks_tr", "edit_wikibooks_ug", "edit_wikibooks_uk", "edit_wikibooks_uz", "edit_wikibooks_vi", "edit_wikibooks_vo", "edit_wikibooks_wa", "edit_wikibooks_yo", "edit_wikibooks_za", "edit_wikibooks_zh_min_nan", "edit_wikinews_bs", "edit_wikinews_ca", "edit_wikinews_cs", "edit_wikinews_eo", "edit_wikinews_fa", "edit_wikinews_fi"     , "edit_wikinews_he"     , "edit_wikinews_ko"     , "edit_wikinews_nl"     , "edit_wikinews_ro"     , "edit_wikinews_sd"     , "edit_wikinews_sq"     , "edit_wikinews_ta"    , "edit_wikinews_th"    , "edit_wikinews_tr"    , "edit_wikinews_uk"    , "edit_wikiquote_af", "edit_wikiquote_am"     , "edit_wikiquote_ang"    , "edit_wikiquote_ast"    , "edit_wikiquote_az"     , "edit_wikiquote_be"     , "edit_wikiquote_bg"     , "edit_wikiquote_bm"     , "edit_wikiquote_br"     , "edit_wikiquote_bs"     , "edit_wikiquote_ca"     , "edit_wikiquote_co"     , "edit_wikiquote_cs"     , "edit_wikiquote_cy"     , "edit_wikiquote_da"     , "edit_wikiquote_el"     , "edit_wikiquote_eo"     , "edit_wikiquote_eu"     , "edit_wikiquote_fi"     , "edit_wikiquote_ga"     , "edit_wikiquote_gl"     , "edit_wikiquote_gu"     , "edit_wikiquote_he"     , "edit_wikiquote_hi"     , "edit_wikiquote_hr"     , "edit_wikiquote_hu"     , "edit_wikiquote_hy"     , "edit_wikiquote_is"     , "edit_wikiquote_ja"     , "edit_wikiquote_ka"     , "edit_wikiquote_kk"     , "edit_wikiquote_kn"     , "edit_wikiquote_ko"     , "edit_wikiquote_kr"     , "edit_wikiquote_ks"     , "edit_wikiquote_ku"     , "edit_wikiquote_ky"     , "edit_wikiquote_la"     , "edit_wikiquote_lb"     , "edit_wikiquote_li"     , "edit_wikiquote_lt"     , "edit_wikiquote_mr"     , "edit_wikiquote_nl"     , "edit_wikiquote_nn"     , "edit_wikiquote_no"     , "edit_wikiquote_ro"     , "edit_wikiquote_sa"     , "edit_wikiquote_simple", "edit_wikiquote_sk"     , "edit_wikiquote_sl"     , "edit_wikiquote_sq"     , "edit_wikiquote_sr"     , "edit_wikiquote_su"     , "edit_wikiquote_sv"     , "edit_wikiquote_ta"     , "edit_wikiquote_th"     , "edit_wikiquote_tk"     , "edit_wikiquote_tr"     , "edit_wikiquote_tt"     , "edit_wikiquote_ug"     , "edit_wikiquote_uk"     , "edit_wikiquote_ur"     , "edit_wikiquote_uz"     , "edit_wikiquote_vi"     , "edit_wikiquote_vo"     , "edit_wikiquote_wo", "edit_wiktionary_aa"    , "edit_wiktionary_ab"    , "edit_wiktionary_ak"    , "edit_wiktionary_als"   , "edit_wiktionary_am"    , "edit_wiktionary_an"    , "edit_wiktionary_ang"   , "edit_wiktionary_as"    , "edit_wiktionary_av"    , "edit_wiktionary_ay"    , "edit_wiktionary_bh"    , "edit_wiktionary_bi"    , "edit_wiktionary_bm"    , "edit_wiktionary_bn"    , "edit_wiktionary_bo"    , "edit_wiktionary_bs"    , "edit_wiktionary_ch"    , "edit_wiktionary_co"    , "edit_wiktionary_cr"    , "edit_wiktionary_csb"   , "edit_wiktionary_dv"    , "edit_wiktionary_dz"    , "edit_wiktionary_fy"    , "edit_wiktionary_ga"    , "edit_wiktionary_gd"    , "edit_wiktionary_gn"    , "edit_wiktionary_gu"    , "edit_wiktionary_gv"    , "edit_wiktionary_ha"    , "edit_wiktionary_hsb"   , "edit_wiktionary_ia"    , "edit_wiktionary_ie"    , "edit_wiktionary_ik"    , "edit_wiktionary_iu"    , "edit_wiktionary_jbo"   , "edit_wiktionary_ka"    , "edit_wiktionary_kk"    , "edit_wiktionary_kl"    , "edit_wiktionary_ks"    , "edit_wiktionary_kw"    , "edit_wiktionary_lb"    , "edit_wiktionary_ln"    , "edit_wiktionary_lv"    , "edit_wiktionary_mh"    , "edit_wiktionary_mk"    , "edit_wiktionary_mo"    , "edit_wiktionary_mr"    , "edit_wiktionary_ms"    , "edit_wiktionary_mt"    , "edit_wiktionary_na"    , "edit_wiktionary_nah"   , "edit_wiktionary_ne"    , "edit_wiktionary_nn"    , "edit_wiktionary_om"    , "edit_wiktionary_pa"    , "edit_wiktionary_pnb"   , "edit_wiktionary_qu"    , "edit_wiktionary_rm"    , "edit_wiktionary_rn", "edit_wiktionary_rw"    , "edit_wiktionary_sc"    , "edit_wiktionary_sd"    , "edit_wiktionary_sg"    , "edit_wiktionary_si"    , "edit_wiktionary_sl"    , "edit_wiktionary_sm"    , "edit_wiktionary_sn"    , "edit_wiktionary_so"    , "edit_wiktionary_ss", "edit_wiktionary_su"    , "edit_wiktionary_sw"    , "edit_wiktionary_ti"    , "edit_wiktionary_tk"    , "edit_wiktionary_tn"    , "edit_wiktionary_to"    , "edit_wiktionary_tpi"   , "edit_wiktionary_ts"    , "edit_wiktionary_tt"    , "edit_wiktionary_tw"    , "edit_wiktionary_ug"    , "edit_wiktionary_ur", "edit_wiktionary_wo"    , "edit_wiktionary_xh"    , "edit_wiktionary_yi"    , "edit_wiktionary_yo"    , "edit_wiktionary_za"]
    drop_ego_social_list = ["ego_social_gplus_100521671383026672718", "ego_social_gplus_100535338638690515335", "ego_social_gplus_100637660947564674695", "ego_social_gplus_100668989009254813743", "ego_social_gplus_100715738096376666180", "ego_social_gplus_100720409235366385249", "ego_social_gplus_100962871525684315897", "ego_social_gplus_101130571432010257170", "ego_social_gplus_101133961721621664586", "ego_social_gplus_101185748996927059931", "ego_social_gplus_101263615503715477581", "ego_social_gplus_101373961279443806744", "ego_social_gplus_101499880233887429402", "ego_social_gplus_101541879642294398860", "ego_social_gplus_101560853443212199687", "ego_social_gplus_101626577406833098387", "ego_social_gplus_101848191156408080085", "ego_social_gplus_101997124338642780860", "ego_social_gplus_102170431816592344972", "ego_social_gplus_102340116189726655233", "ego_social_gplus_102615863344410467759", "ego_social_gplus_102778563580121606331", "ego_social_gplus_103236949470535942612", "ego_social_gplus_103241736833663734962", "ego_social_gplus_103251633033550231172", "ego_social_gplus_103338524411980406972", "ego_social_gplus_103503116383846951534", "ego_social_gplus_103537112468125883734", "ego_social_gplus_103752943025677384806", "ego_social_gplus_103892332449873403244", "ego_social_gplus_104105354262797387583", "ego_social_gplus_104226133029319075907", "ego_social_gplus_104290609881668164623", "ego_social_gplus_104607825525972194062", "ego_social_gplus_104672614700283598130", "ego_social_gplus_104905626100400792399", "ego_social_gplus_104917160754181459072", "ego_social_gplus_104987932455782713675", "ego_social_gplus_105565257978663183206", "ego_social_gplus_105646458226420473639", "ego_social_gplus_106186407539128840569", "ego_social_gplus_106228758905254036967", "ego_social_gplus_106328207304735502636", "ego_social_gplus_106382433884876652170", "ego_social_gplus_106417861423111072106", "ego_social_gplus_106724181552911298818", "ego_social_gplus_106837574755355833243", "ego_social_gplus_107013688749125521109", "ego_social_gplus_107040353898400532534", "ego_social_gplus_107203023379915799071", "ego_social_gplus_107223200089245371832", "ego_social_gplus_107296660002634487593", "ego_social_gplus_107362628080904735459", "ego_social_gplus_107459220492917008623", "ego_social_gplus_107489144252174167638", "ego_social_gplus_107965826228461029730", "ego_social_gplus_108156134340151350951", "ego_social_gplus_108404515213153345305", "ego_social_gplus_108541235642523883716", "ego_social_gplus_108883879052307976051", "ego_social_gplus_109130886479781915270", "ego_social_gplus_109213135085178239952", "ego_social_gplus_109327480479767108490", "ego_social_gplus_109342148209917802565", "ego_social_gplus_109596373340495798827", "ego_social_gplus_109602109099036550366", "ego_social_gplus_110232479818136355682", "ego_social_gplus_110241952466097562819", "ego_social_gplus_110538600381916983600", "ego_social_gplus_110581012109008817546", "ego_social_gplus_110614416163543421878", "ego_social_gplus_110701307803962595019", "ego_social_gplus_110739220927723360152", "ego_social_gplus_110809308822849680310", "ego_social_gplus_110971010308065250763", "ego_social_gplus_111048918866742956374", "ego_social_gplus_111058843129764709244", "ego_social_gplus_111091089527727420853", "ego_social_gplus_111213696402662884531", "ego_social_gplus_112317819390625199896", "ego_social_gplus_112463391491520264813", "ego_social_gplus_112573107772208475213", "ego_social_gplus_112724573277710080670", "ego_social_gplus_112737356589974073749", "ego_social_gplus_112787435697866537461", "ego_social_gplus_113112256846010263985", "ego_social_gplus_113122049849685469495", "ego_social_gplus_113171096418029011322", "ego_social_gplus_113356364521839061717", "ego_social_gplus_113455290791279442483", "ego_social_gplus_113597493946570654755", "ego_social_gplus_113718775944980638561", "ego_social_gplus_113799277735885972934", "ego_social_gplus_113881433443048137993", "ego_social_gplus_114054672576929802335", "ego_social_gplus_114104634069486127920", "ego_social_gplus_114122960748905067938", "ego_social_gplus_114124942936679476879", "ego_social_gplus_114147483140782280818", "ego_social_gplus_114336431216099933033", "ego_social_gplus_115121555137256496805", "ego_social_gplus_115273860520983542999", "ego_social_gplus_115360471097759949621", "ego_social_gplus_115455024457484679647", "ego_social_gplus_115516333681138986628", "ego_social_gplus_115573545440464933254", "ego_social_gplus_115576988435396060952", "ego_social_gplus_115625564993990145546", "ego_social_gplus_116059998563577101552", "ego_social_gplus_116247667398036716276", "ego_social_gplus_116315897040732668413", "ego_social_gplus_116450966137824114154", "ego_social_gplus_116807883656585676940", "ego_social_gplus_116825083494890429556", "ego_social_gplus_116899029375914044550", "ego_social_gplus_116931379084245069738", "ego_social_gplus_117412175333096244275", "ego_social_gplus_117503822947457399073", "ego_social_gplus_117668392750579292609", "ego_social_gplus_117734260411963901771", "ego_social_gplus_117798157258572080176", "ego_social_gplus_117866881767579360121", "ego_social_gplus_118107045405823607895", "ego_social_gplus_118255645714452180374", "ego_social_gplus_118379821279745746467"]

    singularValuesFiles_list = glob.glob('properties/singular_values/'
                                         '*_singular_values.txt')
    # print(singularValuesFiles_list)
    # print(len(singularValuesFiles_list))
    for networkName in drop_board_directors_list:
        singularValuesFiles_list.remove("properties/singular_values\\"
                                        + networkName + "_singular_values.txt")
    for networkName in drop_wiki_list:
        singularValuesFiles_list.remove("properties/singular_values\\"
                                        + networkName + "_singular_values.txt")
    for networkName in drop_ego_social_list:
        singularValuesFiles_list.remove("properties/singular_values\\"
                                        + networkName + "_singular_values.txt")

    # Extract effective ranks from Netzschleuder
    for networkName in tqdm(singularValuesFiles_list):
        networkName = networkName.split('_singular_values')[0].split('\\')[-1]
        if recompute:
            singularValuesFilename = 'properties/singular_values/' \
                                     + networkName + '_singular_values.txt'
            singularValues = np.loadtxt(singularValuesFilename)
            effectiveRanks = [networkName,
                              graphPropDF.loc[networkName]['nbVertices'],
                              computeRank(singularValues),
                              computeOptimalThreshold(singularValues),
                              computeOptimalShrinkage(singularValues),
                              computeERank(singularValues),
                              findEffectiveRankElbow(singularValues),
                              computeEffectiveRankEnergyRatio(singularValues,
                                                              threshold=0.9),
                              computeStableRank(singularValues),
                              computeNuclearRank(singularValues)]
            effectiveRanksList.append(effectiveRanks)
        else:
            if not effectiveRanksDF['Name'].str.contains(networkName).any():
                singularValuesFilename = 'properties/singular_values/'\
                                         + networkName + '_singular_values.txt'
                singularValues = np.loadtxt(singularValuesFilename)
                effectiveRanks = [networkName,
                                  graphPropDF.loc[networkName]['nbVertices'],
                                  computeRank(singularValues),
                                  computeOptimalThreshold(singularValues),
                                  computeOptimalShrinkage(singularValues),
                                  computeERank(singularValues),
                                  findEffectiveRankElbow(singularValues),
                                  computeEffectiveRankEnergyRatio(
                                      singularValues, threshold=0.9),
                                  computeStableRank(singularValues),
                                  computeNuclearRank(singularValues)]
                effectiveRanksList.append(effectiveRanks)

    # Extract effective ranks from other sources
    for networkName in tqdm(glob.glob('properties/*_singular_values.txt')):
        networkName = networkName.split('_singular_values')[0].split('\\')[-1]

        singularValuesFilename = 'properties/' \
                                 + networkName + '_singular_values.txt'
        singularValues = np.loadtxt(singularValuesFilename)
        """ Note that the zero singular values must be included here to compute
                the size of the networks with len(singularValues). """
        effectiveRanks = [networkName,
                          len(singularValues),
                          computeRank(singularValues),
                          computeOptimalThreshold(singularValues),
                          computeOptimalShrinkage(singularValues),
                          computeERank(singularValues),
                          findEffectiveRankElbow(singularValues),
                          computeEffectiveRankEnergyRatio(singularValues,
                                                          threshold=0.9),
                          computeStableRank(singularValues),
                          computeNuclearRank(singularValues)]
        effectiveRanksList.append(effectiveRanks)

    effectiveRanksDF = pd.DataFrame(effectiveRanksList, columns=header)
    effectiveRanksDF.sort_values('Name', inplace=True)
    effectiveRanksDF.reset_index(drop=True, inplace=True)
    effectiveRanksDF.to_fwf(effectiveRanksFilename)


if __name__ == "__main__":
    computeEffectiveRanksManyNetworks(recompute=True)
