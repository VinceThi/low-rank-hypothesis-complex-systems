# -​*- coding: utf-8 -*​-
# @author: Vincent Thibeault and Antoine Allard <antoineallard.info>
import pandas as pd
import seaborn as sns
import numpy as np
from plots.config_rcparams import *

""" Plots the effective ranks to ratios
 vs the ranks to ratios of various real networks. """

figureFilenamePDF = 'figures/pdf/fig1_effective_rank_vs_rank_real_networks.pdf'
figureFilenamePNG = 'figures/png/fig1_effective_rank_vs_rank_real_networks.png'

graphPropFilename = 'C:/Users/thivi/Documents/GitHub/' \
                    'low-rank-hypothesis-complex-systems/graphs/' \
                    'graph_data/graph_properties.txt'
header = open(graphPropFilename, 'r').readline().replace('#', ' ').split()
graphPropDF = pd.read_table(graphPropFilename, names=header,
                            comment="#", delimiter=r"\s+")
graphPropDF.set_index('name', inplace=True)

graphPropDF.drop(labels=["board_directors_net1m_2002-06-01", "board_directors_net1m_2002-07-01", "board_directors_net1m_2002-08-01", "board_directors_net1m_2002-09-01", "board_directors_net1m_2002-10-01", "board_directors_net1m_2002-11-01", "board_directors_net1m_2003-02-01", "board_directors_net1m_2003-03-01", "board_directors_net1m_2003-04-01", "board_directors_net1m_2003-05-01", "board_directors_net1m_2003-06-01", "board_directors_net1m_2003-07-01", "board_directors_net1m_2003-08-01", "board_directors_net1m_2003-09-01", "board_directors_net1m_2003-10-01", "board_directors_net1m_2003-11-01", "board_directors_net1m_2004-02-01", "board_directors_net1m_2004-03-01", "board_directors_net1m_2004-04-01", "board_directors_net1m_2004-05-01", "board_directors_net1m_2004-06-01", "board_directors_net1m_2004-07-01", "board_directors_net1m_2004-08-01", "board_directors_net1m_2004-09-01", "board_directors_net1m_2004-10-01", "board_directors_net1m_2004-11-01", "board_directors_net1m_2005-02-01", "board_directors_net1m_2005-03-01", "board_directors_net1m_2005-04-01", "board_directors_net1m_2005-05-01", "board_directors_net1m_2005-06-01", "board_directors_net1m_2005-07-01", "board_directors_net1m_2005-08-01", "board_directors_net1m_2005-09-01", "board_directors_net1m_2005-10-01", "board_directors_net1m_2005-11-01", "board_directors_net1m_2006-02-01", "board_directors_net1m_2006-03-01", "board_directors_net1m_2006-04-01", "board_directors_net1m_2006-05-01", "board_directors_net1m_2006-06-01", "board_directors_net1m_2006-07-01", "board_directors_net1m_2006-08-01", "board_directors_net1m_2006-09-01", "board_directors_net1m_2006-10-01", "board_directors_net1m_2006-11-01", "board_directors_net1m_2007-02-01", "board_directors_net1m_2007-03-01", "board_directors_net1m_2007-04-01", "board_directors_net1m_2007-05-01", "board_directors_net1m_2007-06-01", "board_directors_net1m_2007-07-01", "board_directors_net1m_2007-08-01", "board_directors_net1m_2007-09-01", "board_directors_net1m_2007-10-01", "board_directors_net1m_2007-11-01", "board_directors_net1m_2008-02-01", "board_directors_net1m_2008-03-01", "board_directors_net1m_2008-04-01", "board_directors_net1m_2008-05-01", "board_directors_net1m_2008-06-01", "board_directors_net1m_2008-07-01", "board_directors_net1m_2008-08-01", "board_directors_net1m_2008-09-01", "board_directors_net1m_2008-10-01", "board_directors_net1m_2008-11-01", "board_directors_net1m_2009-02-01", "board_directors_net1m_2009-03-01", "board_directors_net1m_2009-04-01", "board_directors_net1m_2009-05-01", "board_directors_net1m_2009-06-01", "board_directors_net1m_2009-07-01", "board_directors_net1m_2009-08-01", "board_directors_net1m_2009-09-01", "board_directors_net1m_2009-10-01", "board_directors_net1m_2009-11-01", "board_directors_net1m_2010-02-01", "board_directors_net1m_2010-03-01", "board_directors_net1m_2010-04-01", "board_directors_net1m_2010-05-01", "board_directors_net1m_2010-06-01", "board_directors_net1m_2010-07-01", "board_directors_net1m_2010-08-01", "board_directors_net1m_2010-09-01", "board_directors_net1m_2010-10-01", "board_directors_net1m_2010-11-01", "board_directors_net1m_2011-02-01", "board_directors_net1m_2011-03-01", "board_directors_net1m_2011-04-01", "board_directors_net1m_2011-05-01", "board_directors_net1m_2011-06-01", "board_directors_net1m_2011-07-01", "board_directors_net1m_2011-08-01"],
                 axis=0, inplace=True)
graphPropDF.drop(labels=["edit_wikibooks_af", "edit_wikibooks_ak", "edit_wikibooks_als", "edit_wikibooks_ang", "edit_wikibooks_as", "edit_wikibooks_ast", "edit_wikibooks_bg", "edit_wikibooks_bi", "edit_wikibooks_bm", "edit_wikibooks_bn", "edit_wikibooks_bo", "edit_wikibooks_ch", "edit_wikibooks_co", "edit_wikibooks_cs", "edit_wikibooks_cv", "edit_wikibooks_el", "edit_wikibooks_eo", "edit_wikibooks_et", "edit_wikibooks_fi", "edit_wikibooks_got", "edit_wikibooks_gu"     , "edit_wikibooks_he", "edit_wikibooks_hi", "edit_wikibooks_hy", "edit_wikibooks_ia", "edit_wikibooks_id", "edit_wikibooks_is", "edit_wikibooks_kk", "edit_wikibooks_km", "edit_wikibooks_kn", "edit_wikibooks_ko", "edit_wikibooks_ks", "edit_wikibooks_ku", "edit_wikibooks_lb", "edit_wikibooks_li", "edit_wikibooks_ln", "edit_wikibooks_lt", "edit_wikibooks_lv", "edit_wikibooks_mg", "edit_wikibooks_mi"     , "edit_wikibooks_mk"     , "edit_wikibooks_ml"     , "edit_wikibooks_mn"     , "edit_wikibooks_mr"     , "edit_wikibooks_ms", "edit_wikibooks_nah"    , "edit_wikibooks_nds"    , "edit_wikibooks_ne", "edit_wikibooks_pa", "edit_wikibooks_ps", "edit_wikibooks_qu", "edit_wikibooks_rm", "edit_wikibooks_ro", "edit_wikibooks_ru", "edit_wikibooks_sa", "edit_wikibooks_se", "edit_wikibooks_simple", "edit_wikibooks_sk", "edit_wikibooks_sl", "edit_wikibooks_sq", "edit_wikibooks_su", "edit_wikibooks_sv", "edit_wikibooks_sw", "edit_wikibooks_ta", "edit_wikibooks_te", "edit_wikibooks_th", "edit_wikibooks_tk", "edit_wikibooks_tl", "edit_wikibooks_tr", "edit_wikibooks_ug", "edit_wikibooks_uk", "edit_wikibooks_uz", "edit_wikibooks_vi", "edit_wikibooks_vo", "edit_wikibooks_wa", "edit_wikibooks_yo", "edit_wikibooks_za", "edit_wikibooks_zh_min_nan", "edit_wikinews_bs", "edit_wikinews_ca", "edit_wikinews_cs", "edit_wikinews_eo", "edit_wikinews_fa", "edit_wikinews_fi"     , "edit_wikinews_he"     , "edit_wikinews_ko"     , "edit_wikinews_nl"     , "edit_wikinews_ro"     , "edit_wikinews_sd"     , "edit_wikinews_sq"     , "edit_wikinews_ta"    , "edit_wikinews_th"    , "edit_wikinews_tr"    , "edit_wikinews_uk"    , "edit_wikiquote_af", "edit_wikiquote_am"     , "edit_wikiquote_ang"    , "edit_wikiquote_ast"    , "edit_wikiquote_az"     , "edit_wikiquote_be"     , "edit_wikiquote_bg"     , "edit_wikiquote_bm"     , "edit_wikiquote_br"     , "edit_wikiquote_bs"     , "edit_wikiquote_ca"     , "edit_wikiquote_co"     , "edit_wikiquote_cs"     , "edit_wikiquote_cy"     , "edit_wikiquote_da"     , "edit_wikiquote_el"     , "edit_wikiquote_eo"     , "edit_wikiquote_eu"     , "edit_wikiquote_fi"     , "edit_wikiquote_ga"     , "edit_wikiquote_gl"     , "edit_wikiquote_gu"     , "edit_wikiquote_he"     , "edit_wikiquote_hi"     , "edit_wikiquote_hr"     , "edit_wikiquote_hu"     , "edit_wikiquote_hy"     , "edit_wikiquote_is"     , "edit_wikiquote_ja"     , "edit_wikiquote_ka"     , "edit_wikiquote_kk"     , "edit_wikiquote_kn"     , "edit_wikiquote_ko"     , "edit_wikiquote_kr"     , "edit_wikiquote_ks"     , "edit_wikiquote_ku"     , "edit_wikiquote_ky"     , "edit_wikiquote_la"     , "edit_wikiquote_lb"     , "edit_wikiquote_li"     , "edit_wikiquote_lt"     , "edit_wikiquote_mr"     , "edit_wikiquote_nl"     , "edit_wikiquote_nn"     , "edit_wikiquote_no"     , "edit_wikiquote_ro"     , "edit_wikiquote_sa"     , "edit_wikiquote_simple", "edit_wikiquote_sk"     , "edit_wikiquote_sl"     , "edit_wikiquote_sq"     , "edit_wikiquote_sr"     , "edit_wikiquote_su"     , "edit_wikiquote_sv"     , "edit_wikiquote_ta"     , "edit_wikiquote_th"     , "edit_wikiquote_tk"     , "edit_wikiquote_tr"     , "edit_wikiquote_tt"     , "edit_wikiquote_ug"     , "edit_wikiquote_uk"     , "edit_wikiquote_ur"     , "edit_wikiquote_uz"     , "edit_wikiquote_vi"     , "edit_wikiquote_vo"     , "edit_wikiquote_wo", "edit_wiktionary_aa"    , "edit_wiktionary_ab"    , "edit_wiktionary_ak"    , "edit_wiktionary_als"   , "edit_wiktionary_am"    , "edit_wiktionary_an"    , "edit_wiktionary_ang"   , "edit_wiktionary_as"    , "edit_wiktionary_av"    , "edit_wiktionary_ay"    , "edit_wiktionary_bh"    , "edit_wiktionary_bi"    , "edit_wiktionary_bm"    , "edit_wiktionary_bn"    , "edit_wiktionary_bo"    , "edit_wiktionary_bs"    , "edit_wiktionary_ch"    , "edit_wiktionary_co"    , "edit_wiktionary_cr"    , "edit_wiktionary_csb"   , "edit_wiktionary_dv"    , "edit_wiktionary_dz"    , "edit_wiktionary_fy"    , "edit_wiktionary_ga"    , "edit_wiktionary_gd"    , "edit_wiktionary_gn"    , "edit_wiktionary_gu"    , "edit_wiktionary_gv"    , "edit_wiktionary_ha"    , "edit_wiktionary_hsb"   , "edit_wiktionary_ia"    , "edit_wiktionary_ie"    , "edit_wiktionary_ik"    , "edit_wiktionary_iu"    , "edit_wiktionary_jbo"   , "edit_wiktionary_ka"    , "edit_wiktionary_kk"    , "edit_wiktionary_kl"    , "edit_wiktionary_ks"    , "edit_wiktionary_kw"    , "edit_wiktionary_lb"    , "edit_wiktionary_ln"    , "edit_wiktionary_lv"    , "edit_wiktionary_mh"    , "edit_wiktionary_mk"    , "edit_wiktionary_mo"    , "edit_wiktionary_mr"    , "edit_wiktionary_ms"    , "edit_wiktionary_mt"    , "edit_wiktionary_na"    , "edit_wiktionary_nah"   , "edit_wiktionary_ne"    , "edit_wiktionary_nn"    , "edit_wiktionary_om"    , "edit_wiktionary_pa"    , "edit_wiktionary_pnb"   , "edit_wiktionary_qu"    , "edit_wiktionary_rm"    , "edit_wiktionary_rn", "edit_wiktionary_rw"    , "edit_wiktionary_sc"    , "edit_wiktionary_sd"    , "edit_wiktionary_sg"    , "edit_wiktionary_si"    , "edit_wiktionary_sl"    , "edit_wiktionary_sm"    , "edit_wiktionary_sn"    , "edit_wiktionary_so"    , "edit_wiktionary_ss", "edit_wiktionary_su"    , "edit_wiktionary_sw"    , "edit_wiktionary_ti"    , "edit_wiktionary_tk"    , "edit_wiktionary_tn"    , "edit_wiktionary_to"    , "edit_wiktionary_tpi"   , "edit_wiktionary_ts"    , "edit_wiktionary_tt"    , "edit_wiktionary_tw"    , "edit_wiktionary_ug"    , "edit_wiktionary_ur", "edit_wiktionary_wo"    , "edit_wiktionary_xh"    , "edit_wiktionary_yi"    , "edit_wiktionary_yo"    , "edit_wiktionary_za"],
                 axis=0, inplace=True)
graphPropDF.drop(labels=["ego_social_gplus_100521671383026672718", "ego_social_gplus_100535338638690515335", "ego_social_gplus_100637660947564674695", "ego_social_gplus_100668989009254813743", "ego_social_gplus_100715738096376666180", "ego_social_gplus_100720409235366385249", "ego_social_gplus_100962871525684315897", "ego_social_gplus_101130571432010257170", "ego_social_gplus_101133961721621664586", "ego_social_gplus_101185748996927059931", "ego_social_gplus_101263615503715477581", "ego_social_gplus_101373961279443806744", "ego_social_gplus_101499880233887429402", "ego_social_gplus_101541879642294398860", "ego_social_gplus_101560853443212199687", "ego_social_gplus_101626577406833098387", "ego_social_gplus_101848191156408080085", "ego_social_gplus_101997124338642780860", "ego_social_gplus_102170431816592344972", "ego_social_gplus_102340116189726655233", "ego_social_gplus_102615863344410467759", "ego_social_gplus_102778563580121606331", "ego_social_gplus_103236949470535942612", "ego_social_gplus_103241736833663734962", "ego_social_gplus_103251633033550231172", "ego_social_gplus_103338524411980406972", "ego_social_gplus_103503116383846951534", "ego_social_gplus_103537112468125883734", "ego_social_gplus_103752943025677384806", "ego_social_gplus_103892332449873403244", "ego_social_gplus_104105354262797387583", "ego_social_gplus_104226133029319075907", "ego_social_gplus_104290609881668164623", "ego_social_gplus_104607825525972194062", "ego_social_gplus_104672614700283598130", "ego_social_gplus_104905626100400792399", "ego_social_gplus_104917160754181459072", "ego_social_gplus_104987932455782713675", "ego_social_gplus_105565257978663183206", "ego_social_gplus_105646458226420473639", "ego_social_gplus_106186407539128840569", "ego_social_gplus_106228758905254036967", "ego_social_gplus_106328207304735502636", "ego_social_gplus_106382433884876652170", "ego_social_gplus_106417861423111072106", "ego_social_gplus_106724181552911298818", "ego_social_gplus_106837574755355833243", "ego_social_gplus_107013688749125521109", "ego_social_gplus_107040353898400532534", "ego_social_gplus_107203023379915799071", "ego_social_gplus_107223200089245371832", "ego_social_gplus_107296660002634487593", "ego_social_gplus_107362628080904735459", "ego_social_gplus_107459220492917008623", "ego_social_gplus_107489144252174167638", "ego_social_gplus_107965826228461029730", "ego_social_gplus_108156134340151350951", "ego_social_gplus_108404515213153345305", "ego_social_gplus_108541235642523883716", "ego_social_gplus_108883879052307976051", "ego_social_gplus_109130886479781915270", "ego_social_gplus_109213135085178239952", "ego_social_gplus_109327480479767108490", "ego_social_gplus_109342148209917802565", "ego_social_gplus_109596373340495798827", "ego_social_gplus_109602109099036550366", "ego_social_gplus_110232479818136355682", "ego_social_gplus_110241952466097562819", "ego_social_gplus_110538600381916983600", "ego_social_gplus_110581012109008817546", "ego_social_gplus_110614416163543421878", "ego_social_gplus_110701307803962595019", "ego_social_gplus_110739220927723360152", "ego_social_gplus_110809308822849680310", "ego_social_gplus_110971010308065250763", "ego_social_gplus_111048918866742956374", "ego_social_gplus_111058843129764709244", "ego_social_gplus_111091089527727420853", "ego_social_gplus_111213696402662884531", "ego_social_gplus_112317819390625199896", "ego_social_gplus_112463391491520264813", "ego_social_gplus_112573107772208475213", "ego_social_gplus_112724573277710080670", "ego_social_gplus_112737356589974073749", "ego_social_gplus_112787435697866537461", "ego_social_gplus_113112256846010263985", "ego_social_gplus_113122049849685469495", "ego_social_gplus_113171096418029011322", "ego_social_gplus_113356364521839061717", "ego_social_gplus_113455290791279442483", "ego_social_gplus_113597493946570654755", "ego_social_gplus_113718775944980638561", "ego_social_gplus_113799277735885972934", "ego_social_gplus_113881433443048137993", "ego_social_gplus_114054672576929802335", "ego_social_gplus_114104634069486127920", "ego_social_gplus_114122960748905067938", "ego_social_gplus_114124942936679476879", "ego_social_gplus_114147483140782280818", "ego_social_gplus_114336431216099933033", "ego_social_gplus_115121555137256496805", "ego_social_gplus_115273860520983542999", "ego_social_gplus_115360471097759949621", "ego_social_gplus_115455024457484679647", "ego_social_gplus_115516333681138986628", "ego_social_gplus_115573545440464933254", "ego_social_gplus_115576988435396060952", "ego_social_gplus_115625564993990145546", "ego_social_gplus_116059998563577101552", "ego_social_gplus_116247667398036716276", "ego_social_gplus_116315897040732668413", "ego_social_gplus_116450966137824114154", "ego_social_gplus_116807883656585676940", "ego_social_gplus_116825083494890429556", "ego_social_gplus_116899029375914044550", "ego_social_gplus_116931379084245069738", "ego_social_gplus_117412175333096244275", "ego_social_gplus_117503822947457399073", "ego_social_gplus_117668392750579292609", "ego_social_gplus_117734260411963901771", "ego_social_gplus_117798157258572080176", "ego_social_gplus_117866881767579360121", "ego_social_gplus_118107045405823607895", "ego_social_gplus_118255645714452180374", "ego_social_gplus_118379821279745746467"],
                 axis=0, inplace=True)

# Connectome tag
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Connectome", 'tags'] = "Connectome"

# Ecological tag
graphPropDF.loc[graphPropDF['tags'] ==
                "Social,Animal", 'tags'] = "Ecological"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Foodweb", 'tags'] = "Ecological"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,FoodWeb,Uncertain", 'tags'] = "Ecological"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Foodweb,Multilayer", 'tags'] = "Ecological"

# Interactome tag
# See Interactome Networks and Human Disease by Vidal et al. for more info
# on interactomes. We include the drug-drug interactions in the
# interactome category.
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Proteininteractions", 'tags'] = "Interactome"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Generegulation", 'tags'] = "Interactome"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Generegulation,Proteininteractions,Multilayer",
                'tags'] = "Interactome"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Genetic,Projection,Multilayer",
                'tags'] = "Interactome"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Metabolic", 'tags'] = "Interactome"
graphPropDF.loc[graphPropDF['tags'] ==
                "Biological,Druginteractions", 'tags'] = "Interactome"

# Economic tag
graphPropDF.loc[graphPropDF['tags'] ==
                "Economic,Trade,Multilayer", 'tags'] = "Economic"

# Communication tag
graphPropDF.loc[graphPropDF['tags'] ==
                "Social,Communication", 'tags'] = "Communication"
graphPropDF.loc[graphPropDF['tags'] == "Social,Communication,Timestamps",
                "tags"] = "Communication"

# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None):
#     print(graphPropDF['tags'])

effectiveRanksFilename = 'C:/Users/thivi/Documents/GitHub/' \
                         'low-rank-hypothesis-complex-systems/' \
                         'singular_values/properties/effective_ranks_first.txt'
header = open(effectiveRanksFilename, 'r')\
    .readline().replace('#', ' ').split()
effectiveRanksDF = pd.read_table(effectiveRanksFilename, names=header,
                                 comment="#", delimiter=r"\s+")
effectiveRanksDF.set_index('name', inplace=True)

effectiveRanksDF.drop(labels=["board_directors_net1m_2002-06-01", "board_directors_net1m_2002-07-01", "board_directors_net1m_2002-08-01", "board_directors_net1m_2002-09-01", "board_directors_net1m_2002-10-01", "board_directors_net1m_2002-11-01", "board_directors_net1m_2003-02-01", "board_directors_net1m_2003-03-01", "board_directors_net1m_2003-04-01", "board_directors_net1m_2003-05-01", "board_directors_net1m_2003-06-01", "board_directors_net1m_2003-07-01", "board_directors_net1m_2003-08-01", "board_directors_net1m_2003-09-01", "board_directors_net1m_2003-10-01", "board_directors_net1m_2003-11-01", "board_directors_net1m_2004-02-01", "board_directors_net1m_2004-03-01", "board_directors_net1m_2004-04-01", "board_directors_net1m_2004-05-01", "board_directors_net1m_2004-06-01", "board_directors_net1m_2004-07-01", "board_directors_net1m_2004-08-01", "board_directors_net1m_2004-09-01", "board_directors_net1m_2004-10-01", "board_directors_net1m_2004-11-01", "board_directors_net1m_2005-02-01", "board_directors_net1m_2005-03-01", "board_directors_net1m_2005-04-01", "board_directors_net1m_2005-05-01", "board_directors_net1m_2005-06-01", "board_directors_net1m_2005-07-01", "board_directors_net1m_2005-08-01", "board_directors_net1m_2005-09-01", "board_directors_net1m_2005-10-01", "board_directors_net1m_2005-11-01", "board_directors_net1m_2006-02-01", "board_directors_net1m_2006-03-01", "board_directors_net1m_2006-04-01", "board_directors_net1m_2006-05-01", "board_directors_net1m_2006-06-01", "board_directors_net1m_2006-07-01", "board_directors_net1m_2006-08-01", "board_directors_net1m_2006-09-01", "board_directors_net1m_2006-10-01", "board_directors_net1m_2006-11-01", "board_directors_net1m_2007-02-01", "board_directors_net1m_2007-03-01", "board_directors_net1m_2007-04-01", "board_directors_net1m_2007-05-01", "board_directors_net1m_2007-06-01", "board_directors_net1m_2007-07-01", "board_directors_net1m_2007-08-01", "board_directors_net1m_2007-09-01", "board_directors_net1m_2007-10-01", "board_directors_net1m_2007-11-01", "board_directors_net1m_2008-02-01", "board_directors_net1m_2008-03-01", "board_directors_net1m_2008-04-01", "board_directors_net1m_2008-05-01", "board_directors_net1m_2008-06-01", "board_directors_net1m_2008-07-01", "board_directors_net1m_2008-08-01", "board_directors_net1m_2008-09-01", "board_directors_net1m_2008-10-01", "board_directors_net1m_2008-11-01", "board_directors_net1m_2009-02-01", "board_directors_net1m_2009-03-01", "board_directors_net1m_2009-04-01", "board_directors_net1m_2009-05-01", "board_directors_net1m_2009-06-01", "board_directors_net1m_2009-07-01", "board_directors_net1m_2009-08-01", "board_directors_net1m_2009-09-01", "board_directors_net1m_2009-10-01", "board_directors_net1m_2009-11-01", "board_directors_net1m_2010-02-01", "board_directors_net1m_2010-03-01", "board_directors_net1m_2010-04-01", "board_directors_net1m_2010-05-01", "board_directors_net1m_2010-06-01", "board_directors_net1m_2010-07-01", "board_directors_net1m_2010-08-01", "board_directors_net1m_2010-09-01", "board_directors_net1m_2010-10-01", "board_directors_net1m_2010-11-01", "board_directors_net1m_2011-02-01", "board_directors_net1m_2011-03-01", "board_directors_net1m_2011-04-01", "board_directors_net1m_2011-05-01", "board_directors_net1m_2011-06-01", "board_directors_net1m_2011-07-01", "board_directors_net1m_2011-08-01"],
                      axis=0, inplace=True)
effectiveRanksDF.drop(labels=["edit_wikibooks_af", "edit_wikibooks_ak", "edit_wikibooks_als", "edit_wikibooks_ang", "edit_wikibooks_as", "edit_wikibooks_ast", "edit_wikibooks_bg", "edit_wikibooks_bi", "edit_wikibooks_bm", "edit_wikibooks_bn", "edit_wikibooks_bo", "edit_wikibooks_ch", "edit_wikibooks_co", "edit_wikibooks_cs", "edit_wikibooks_cv", "edit_wikibooks_el", "edit_wikibooks_eo", "edit_wikibooks_et", "edit_wikibooks_fi", "edit_wikibooks_got", "edit_wikibooks_gu"     , "edit_wikibooks_he", "edit_wikibooks_hi", "edit_wikibooks_hy", "edit_wikibooks_ia", "edit_wikibooks_id", "edit_wikibooks_is", "edit_wikibooks_kk", "edit_wikibooks_km", "edit_wikibooks_kn", "edit_wikibooks_ko", "edit_wikibooks_ks", "edit_wikibooks_ku", "edit_wikibooks_lb", "edit_wikibooks_li", "edit_wikibooks_ln", "edit_wikibooks_lt", "edit_wikibooks_lv", "edit_wikibooks_mg", "edit_wikibooks_mi"     , "edit_wikibooks_mk"     , "edit_wikibooks_ml"     , "edit_wikibooks_mn"     , "edit_wikibooks_mr"     , "edit_wikibooks_ms", "edit_wikibooks_nah"    , "edit_wikibooks_nds"    , "edit_wikibooks_ne", "edit_wikibooks_pa", "edit_wikibooks_ps", "edit_wikibooks_qu", "edit_wikibooks_rm", "edit_wikibooks_ro", "edit_wikibooks_ru", "edit_wikibooks_sa", "edit_wikibooks_se", "edit_wikibooks_simple", "edit_wikibooks_sk", "edit_wikibooks_sl", "edit_wikibooks_sq", "edit_wikibooks_su", "edit_wikibooks_sv", "edit_wikibooks_sw", "edit_wikibooks_ta", "edit_wikibooks_te", "edit_wikibooks_th", "edit_wikibooks_tk", "edit_wikibooks_tl", "edit_wikibooks_tr", "edit_wikibooks_ug", "edit_wikibooks_uk", "edit_wikibooks_uz", "edit_wikibooks_vi", "edit_wikibooks_vo", "edit_wikibooks_wa", "edit_wikibooks_yo", "edit_wikibooks_za", "edit_wikibooks_zh_min_nan", "edit_wikinews_bs", "edit_wikinews_ca", "edit_wikinews_cs", "edit_wikinews_eo", "edit_wikinews_fa", "edit_wikinews_fi"     , "edit_wikinews_he"     , "edit_wikinews_ko"     , "edit_wikinews_nl"     , "edit_wikinews_ro"     , "edit_wikinews_sd"     , "edit_wikinews_sq"     , "edit_wikinews_ta"    , "edit_wikinews_th"    , "edit_wikinews_tr"    , "edit_wikinews_uk"    , "edit_wikiquote_af", "edit_wikiquote_am"     , "edit_wikiquote_ang"    , "edit_wikiquote_ast"    , "edit_wikiquote_az"     , "edit_wikiquote_be"     , "edit_wikiquote_bg"     , "edit_wikiquote_bm"     , "edit_wikiquote_br"     , "edit_wikiquote_bs"     , "edit_wikiquote_ca"     , "edit_wikiquote_co"     , "edit_wikiquote_cs"     , "edit_wikiquote_cy"     , "edit_wikiquote_da"     , "edit_wikiquote_el"     , "edit_wikiquote_eo"     , "edit_wikiquote_eu"     , "edit_wikiquote_fi"     , "edit_wikiquote_ga"     , "edit_wikiquote_gl"     , "edit_wikiquote_gu"     , "edit_wikiquote_he"     , "edit_wikiquote_hi"     , "edit_wikiquote_hr"     , "edit_wikiquote_hu"     , "edit_wikiquote_hy"     , "edit_wikiquote_is"     , "edit_wikiquote_ja"     , "edit_wikiquote_ka"     , "edit_wikiquote_kk"     , "edit_wikiquote_kn"     , "edit_wikiquote_ko"     , "edit_wikiquote_kr"     , "edit_wikiquote_ks"     , "edit_wikiquote_ku"     , "edit_wikiquote_ky"     , "edit_wikiquote_la"     , "edit_wikiquote_lb"     , "edit_wikiquote_li"     , "edit_wikiquote_lt"     , "edit_wikiquote_mr"     , "edit_wikiquote_nl"     , "edit_wikiquote_nn"     , "edit_wikiquote_no"     , "edit_wikiquote_ro"     , "edit_wikiquote_sa"     , "edit_wikiquote_simple", "edit_wikiquote_sk"     , "edit_wikiquote_sl"     , "edit_wikiquote_sq"     , "edit_wikiquote_sr"     , "edit_wikiquote_su"     , "edit_wikiquote_sv"     , "edit_wikiquote_ta"     , "edit_wikiquote_th"     , "edit_wikiquote_tk"     , "edit_wikiquote_tr"     , "edit_wikiquote_tt"     , "edit_wikiquote_ug"     , "edit_wikiquote_uk"     , "edit_wikiquote_ur"     , "edit_wikiquote_uz"     , "edit_wikiquote_vi"     , "edit_wikiquote_vo"     , "edit_wikiquote_wo", "edit_wiktionary_aa"    , "edit_wiktionary_ab"    , "edit_wiktionary_ak"    , "edit_wiktionary_als"   , "edit_wiktionary_am"    , "edit_wiktionary_an"    , "edit_wiktionary_ang"   , "edit_wiktionary_as"    , "edit_wiktionary_av"    , "edit_wiktionary_ay"    , "edit_wiktionary_bh"    , "edit_wiktionary_bi"    , "edit_wiktionary_bm"    , "edit_wiktionary_bn"    , "edit_wiktionary_bo"    , "edit_wiktionary_bs"    , "edit_wiktionary_ch"    , "edit_wiktionary_co"    , "edit_wiktionary_cr"    , "edit_wiktionary_csb"   , "edit_wiktionary_dv"    , "edit_wiktionary_dz"    , "edit_wiktionary_fy"    , "edit_wiktionary_ga"    , "edit_wiktionary_gd"    , "edit_wiktionary_gn"    , "edit_wiktionary_gu"    , "edit_wiktionary_gv"    , "edit_wiktionary_ha"    , "edit_wiktionary_hsb"   , "edit_wiktionary_ia"    , "edit_wiktionary_ie"    , "edit_wiktionary_ik"    , "edit_wiktionary_iu"    , "edit_wiktionary_jbo"   , "edit_wiktionary_ka"    , "edit_wiktionary_kk"    , "edit_wiktionary_kl"    , "edit_wiktionary_ks"    , "edit_wiktionary_kw"    , "edit_wiktionary_lb"    , "edit_wiktionary_ln"    , "edit_wiktionary_lv"    , "edit_wiktionary_mh"    , "edit_wiktionary_mk"    , "edit_wiktionary_mo"    , "edit_wiktionary_mr"    , "edit_wiktionary_ms"    , "edit_wiktionary_mt"    , "edit_wiktionary_na"    , "edit_wiktionary_nah"   , "edit_wiktionary_ne"    , "edit_wiktionary_nn"    , "edit_wiktionary_om"    , "edit_wiktionary_pa"    , "edit_wiktionary_pnb"   , "edit_wiktionary_qu"    , "edit_wiktionary_rm"    , "edit_wiktionary_rn", "edit_wiktionary_rw"    , "edit_wiktionary_sc"    , "edit_wiktionary_sd"    , "edit_wiktionary_sg"    , "edit_wiktionary_si"    , "edit_wiktionary_sl"    , "edit_wiktionary_sm"    , "edit_wiktionary_sn"    , "edit_wiktionary_so"    , "edit_wiktionary_ss", "edit_wiktionary_su"    , "edit_wiktionary_sw"    , "edit_wiktionary_ti"    , "edit_wiktionary_tk"    , "edit_wiktionary_tn"    , "edit_wiktionary_to"    , "edit_wiktionary_tpi"   , "edit_wiktionary_ts"    , "edit_wiktionary_tt"    , "edit_wiktionary_tw"    , "edit_wiktionary_ug"    , "edit_wiktionary_ur", "edit_wiktionary_wo"    , "edit_wiktionary_xh"    , "edit_wiktionary_yi"    , "edit_wiktionary_yo"    , "edit_wiktionary_za"],
                      axis=0, inplace=True)
effectiveRanksDF.drop(labels=["ego_social_gplus_100521671383026672718", "ego_social_gplus_100535338638690515335", "ego_social_gplus_100637660947564674695", "ego_social_gplus_100668989009254813743", "ego_social_gplus_100715738096376666180", "ego_social_gplus_100720409235366385249", "ego_social_gplus_100962871525684315897", "ego_social_gplus_101130571432010257170", "ego_social_gplus_101133961721621664586", "ego_social_gplus_101185748996927059931", "ego_social_gplus_101263615503715477581", "ego_social_gplus_101373961279443806744", "ego_social_gplus_101499880233887429402", "ego_social_gplus_101541879642294398860", "ego_social_gplus_101560853443212199687", "ego_social_gplus_101626577406833098387", "ego_social_gplus_101848191156408080085", "ego_social_gplus_101997124338642780860", "ego_social_gplus_102170431816592344972", "ego_social_gplus_102340116189726655233", "ego_social_gplus_102615863344410467759", "ego_social_gplus_102778563580121606331", "ego_social_gplus_103236949470535942612", "ego_social_gplus_103241736833663734962", "ego_social_gplus_103251633033550231172", "ego_social_gplus_103338524411980406972", "ego_social_gplus_103503116383846951534", "ego_social_gplus_103537112468125883734", "ego_social_gplus_103752943025677384806", "ego_social_gplus_103892332449873403244", "ego_social_gplus_104105354262797387583", "ego_social_gplus_104226133029319075907", "ego_social_gplus_104290609881668164623", "ego_social_gplus_104607825525972194062", "ego_social_gplus_104672614700283598130", "ego_social_gplus_104905626100400792399", "ego_social_gplus_104917160754181459072", "ego_social_gplus_104987932455782713675", "ego_social_gplus_105565257978663183206", "ego_social_gplus_105646458226420473639", "ego_social_gplus_106186407539128840569", "ego_social_gplus_106228758905254036967", "ego_social_gplus_106328207304735502636", "ego_social_gplus_106382433884876652170", "ego_social_gplus_106417861423111072106", "ego_social_gplus_106724181552911298818", "ego_social_gplus_106837574755355833243", "ego_social_gplus_107013688749125521109", "ego_social_gplus_107040353898400532534", "ego_social_gplus_107203023379915799071", "ego_social_gplus_107223200089245371832", "ego_social_gplus_107296660002634487593", "ego_social_gplus_107362628080904735459", "ego_social_gplus_107459220492917008623", "ego_social_gplus_107489144252174167638", "ego_social_gplus_107965826228461029730", "ego_social_gplus_108156134340151350951", "ego_social_gplus_108404515213153345305", "ego_social_gplus_108541235642523883716", "ego_social_gplus_108883879052307976051", "ego_social_gplus_109130886479781915270", "ego_social_gplus_109213135085178239952", "ego_social_gplus_109327480479767108490", "ego_social_gplus_109342148209917802565", "ego_social_gplus_109596373340495798827", "ego_social_gplus_109602109099036550366", "ego_social_gplus_110232479818136355682", "ego_social_gplus_110241952466097562819", "ego_social_gplus_110538600381916983600", "ego_social_gplus_110581012109008817546", "ego_social_gplus_110614416163543421878", "ego_social_gplus_110701307803962595019", "ego_social_gplus_110739220927723360152", "ego_social_gplus_110809308822849680310", "ego_social_gplus_110971010308065250763", "ego_social_gplus_111048918866742956374", "ego_social_gplus_111058843129764709244", "ego_social_gplus_111091089527727420853", "ego_social_gplus_111213696402662884531", "ego_social_gplus_112317819390625199896", "ego_social_gplus_112463391491520264813", "ego_social_gplus_112573107772208475213", "ego_social_gplus_112724573277710080670", "ego_social_gplus_112737356589974073749", "ego_social_gplus_112787435697866537461", "ego_social_gplus_113112256846010263985", "ego_social_gplus_113122049849685469495", "ego_social_gplus_113171096418029011322", "ego_social_gplus_113356364521839061717", "ego_social_gplus_113455290791279442483", "ego_social_gplus_113597493946570654755", "ego_social_gplus_113718775944980638561", "ego_social_gplus_113799277735885972934", "ego_social_gplus_113881433443048137993", "ego_social_gplus_114054672576929802335", "ego_social_gplus_114104634069486127920", "ego_social_gplus_114122960748905067938", "ego_social_gplus_114124942936679476879", "ego_social_gplus_114147483140782280818", "ego_social_gplus_114336431216099933033", "ego_social_gplus_115121555137256496805", "ego_social_gplus_115273860520983542999", "ego_social_gplus_115360471097759949621", "ego_social_gplus_115455024457484679647", "ego_social_gplus_115516333681138986628", "ego_social_gplus_115573545440464933254", "ego_social_gplus_115576988435396060952", "ego_social_gplus_115625564993990145546", "ego_social_gplus_116059998563577101552", "ego_social_gplus_116247667398036716276", "ego_social_gplus_116315897040732668413", "ego_social_gplus_116450966137824114154", "ego_social_gplus_116807883656585676940", "ego_social_gplus_116825083494890429556", "ego_social_gplus_116899029375914044550", "ego_social_gplus_116931379084245069738", "ego_social_gplus_117412175333096244275", "ego_social_gplus_117503822947457399073", "ego_social_gplus_117668392750579292609", "ego_social_gplus_117734260411963901771", "ego_social_gplus_117798157258572080176", "ego_social_gplus_117866881767579360121", "ego_social_gplus_118107045405823607895", "ego_social_gplus_118255645714452180374", "ego_social_gplus_118379821279745746467"],
                      axis=0, inplace=True)

effectiveRanksDF['x'] = effectiveRanksDF['rank'] / \
    effectiveRanksDF['nbVertices']
effectiveRanksDF['y'] = effectiveRanksDF['stableRank'] / \
    effectiveRanksDF['nbVertices']

# --- Valid categories, in order of proportion
# validCategories = ['Social', 'Interactome', 'Informational',
#                    'Ecological', 'Connectome', 'Technological',
#                    'Learned', 'Economic', 'Communication',
#                    'Transportation']
# markers = ['.', 'h', 'X', '^', '*', 'd', 'H', 'p', 'D', 'v']
# colors = ["#C44E52", "#DD8452", "#55A868", "#DA8BC3", "#8172B3",
#           "#CCB974", "#937860", "#64B5CD", "#8C8C8C",  "#4C72B0"]
# facecolors = [None, None, None, None, None,
#               None, None, None, None, None]

# --- Valid categories, in increasing order of number of networks in the data
validCategories = ['Transportation', 'Communication', 'Economic', 'Learned',
                   'Technological', 'Connectome',  'Ecological',
                   'Informational', 'Interactome', 'Social']
markers = ['v', 'D', 'p', 'H', 'd', '*', '^', 'X', 'h', '.', 's',
           'P', 'o']
colors = ["#937860", "#DA8BC3", "#8172B3", "#CCB974", "#64B5CD",
          "#C44E52", "#55A868", "#8C8C8C", "#DD8452", "#4C72B0"]
facecolors = ["#C44E52", None, None, None, None,
              None, None, None, None, None]
colorMap = dict(zip(validCategories, colors))
facecolorMap = dict(zip(validCategories, facecolors))
markerMap = dict(zip(validCategories, markers))

rank = {cat: [] for cat in validCategories}
effectiveRank = {cat: [] for cat in validCategories}

for networkName in effectiveRanksDF.index:

    cat = [tag for tag in graphPropDF.loc[networkName]['tags'].split(',')
           if tag in validCategories]
    if len(cat) == 0:
        cat = 'other'
        print(networkName + ' does not have a category')
    else:
        cat = cat[0]

    rank[cat].append(effectiveRanksDF.loc[networkName]['x'])
    effectiveRank[cat].append(effectiveRanksDF.loc[networkName]['y'])

# g = sns.JointGrid(x=[0], y=[0],
#                   height=4.5, ratio=4, space=-3)  # marginal_ticks=True,

# plt.plot([0, 1], [0, 1], linestyle='--', color="#CFCFCF")

# plt.figure(figsize=(4, 4))
# plt.figure(figsize=(4.5, 4.))
plt.figure(figsize=(5, 5))

total_number_networks = 0
for i, cat in enumerate(reversed(validCategories)):

    if cat == "Connectome":
        # Connectomes from Netzschleuder
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                        facecolor='None', s=100,
                        edgecolor=colorMap[cat],  # alpha=0.7,
                        marker=markerMap[cat])  # , ax=g.ax_joint)

        # Connectomes from other sources
        other_connectomes = ["mouse_meso", "mouse_voxel", "zebrafish_meso",
                             "celegans_signed", "drosophila"]
        N_connectomes = np.array([213, 15314, 71, 297, 21733])
        rank_connectome = np.array([185, 15313, 71, 292, 21687])
        srank_connectome = np.array([3.61975, 8.04222, 1.80914,
                                     8.67117, 11.5811])

        # print(srank_connectome/N_connectomes)
        sns.scatterplot(x=rank_connectome/N_connectomes,
                        y=srank_connectome/N_connectomes, s=100,
                        facecolor='None',
                        edgecolor=colorMap[cat], marker=markerMap[cat])
        # cat = "Connectome"
        # sns.scatterplot(x=[0], y=[1.5], facecolor='None', s=100,
        #                 edgecolor=colorMap[cat],
        #                 marker=markerMap[cat], label=cat.lower())
        # # ax=g.ax_joint)
        nb_network_cat = len(effectiveRank[cat]) + len(other_connectomes)
        print(f"{cat}: {nb_network_cat}"
              f" networks")

    elif cat == "Learned":
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                        facecolor='None', s=100,
                        edgecolor=colorMap[cat],  # alpha=0.7,
                        marker=markerMap[cat])  # , ax=g.ax_joint)
        learned_networks = ["zebrafish_rnn", "mouse_rnn", "mouse_control_rnn",
                            "fully_connected_layer_cnn_00100",
                            "fully_connected_layer_cnn_00200",
                            "fully_connected_layer_cnn_00300",
                            "fully_connected_layer_cnn_00400",
                            "fully_connected_layer_cnn_00500",
                            "fully_connected_layer_cnn_00600",
                            "fully_connected_layer_cnn_00700",
                            "fully_connected_layer_cnn_00800",
                            "fully_connected_layer_cnn_00900",
                            "fully_connected_layer_cnn_01000", ]
        N_learned = np.array([4589, 178, 669, 820, 980, 900, 1028, 892,
                              1076, 1028, 628, 396, 820])
        rank_learned = np.array([4589, 178, 669, 299, 10, 128, 256, 380,
                                 308, 256, 116, 128, 308])
        srank_learned = np.array([16.4992, 1.23043, 20.1378, 1.21568,
                                  1.09635, 5.01488, 9.33475, 20.567, 30.1669,
                                  28.1594, 11.3583, 1.22835, 19.7936])
        sns.scatterplot(x=rank_learned / N_learned,
                        y=srank_learned / N_learned, s=100,
                        facecolor='None',
                        edgecolor=colorMap[cat], marker=markerMap[cat])
        nb_network_cat = len(learned_networks)
        print(f"{cat}: {nb_network_cat} networks")

    elif cat == "Economic":
        # Economic network from Netzschleuder
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                        facecolor='None', s=100,
                        edgecolor=colorMap[cat],  # alpha=0.7,
                        marker=markerMap[cat])  # , ax=g.ax_joint)

        # Economic networks from ICON
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                        facecolor='None', s=100,
                        edgecolor=colorMap[cat],  # alpha=0.7,
                        marker=markerMap[cat])  # , ax=g.ax_joint)
        economic_networks = ["AT_2008", "CY_2015", "EE_2010", "PT_2009",
                             "SI_2016", "financial_institution07-Apr-1999",
                             "households_04-Sep-1998",
                             "households_09-Jan-2002",
                             "non_financial_institution04-Jan-2001"]
        N_economic = np.array([2271, 335, 1260, 1257, 1792, 31, 111, 695, 155])
        rank_economic = np.array([842, 108, 312, 560, 790, 31, 111, 695, 155])
        srank_economic = np.array([51.6231, 14.649, 6.12699, 18.9757, 11.1584,
                                   3.18606, 1.97802, 2.26219, 2.58557])
        sns.scatterplot(x=rank_economic / N_economic,
                        y=srank_economic / N_economic, s=100,
                        facecolor='None',
                        edgecolor=colorMap[cat], marker=markerMap[cat])
        nb_network_cat = len(effectiveRank[cat]) + len(economic_networks)
        print(f"{cat}: {nb_network_cat}"
              f" networks")

    elif cat == "Interactome":
        # Interactomes from Netzschleuder
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                        facecolor='None',  # s=100,
                        edgecolor=colorMap[cat],  # alpha=0.7,
                        marker=markerMap[cat])  # , ax=g.ax_joint)

        # Interactomes from other sources
        other_interactome = ["gut"]
        N_interactome = np.array([838])
        rank_interactome = np.array([735])
        srank_interactome = np.array([2.52231])

        sns.scatterplot(x=rank_interactome / N_interactome,
                        y=srank_interactome / N_interactome, # s=50,
                        facecolor='None',
                        edgecolor=colorMap[cat], marker=markerMap[cat])

        nb_network_cat = len(effectiveRank[cat]) + len(other_interactome)
        print(f"{cat}: {nb_network_cat}"
              f" networks")

    else:
        sns.scatterplot(x=rank[cat], y=effectiveRank[cat],
                        facecolor='None',
                        edgecolor=colorMap[cat], alpha=0.7,
                        marker=markerMap[cat])  # , ax=g.ax_joint)
        nb_network_cat = len(effectiveRank[cat])
        print(f"{cat}: {nb_network_cat} networks "
              f"(all from Netzschleuder)")
    # cat = sorted(validCategories)[i]
    sns.scatterplot(x=[0], y=[1.5], facecolor='None',
                    edgecolor=colorMap[cat],
                    marker=markerMap[cat],  # .lower(),to have lower case label
                    label=cat + f" ({np.around(nb_network_cat/675*100, 1)}%)",
                    zorder=len(validCategories)-i)

    total_number_networks += len(effectiveRank[cat])

total_number_networks = total_number_networks\
                        + len(other_connectomes) + len(other_interactome)\
                        + len(learned_networks) + len(economic_networks)
print(f"Total number of networks = {total_number_networks}")

if total_number_networks != 675:
    raise ValueError("One much change the number of networks 675 which is "
                     "hard coded in other places in the script.")

# g.ax_joint.legend(loc='upper left', frameon=False, fontsize='x-small')

# sns.histplot(x=effectiveRanksDF['rank'] / effectiveRanksDF['nbVertices'],
#              stat='probability', bins=30, binrange=(0, 1),
#              color="lightsteelblue", linewidth=0.3, ax=g.ax_marg_x)
# sns.histplot(y=effectiveRanksDF['stableRank'] /
#              effectiveRanksDF['nbVertices'],
#              stat='probability', bins=4*30, binrange=(0, 1),
#              color="lightsteelblue", linewidth=0.3, ax=g.ax_marg_y)

plt.xlim(left=-0.01, right=1.05)
plt.ylim(bottom=-0.01, top=0.22)

# plt.yscale('log')
# plt.xscale('log')

# g.ax_joint.set_xlabel('Rank to dimension ratio')
# g.ax_joint.set_ylabel('Effective rank to dimension ratio')

# plt.xlabel('Rank to dimension ratio rank/N')
# plt.ylabel('Effective rank to dimension ratio srank/N')
plt.xlabel('rank/N')
plt.ylabel('srank/N')
plt.show()

plt.savefig(figureFilenamePDF)  # , bbox_inches='tight')
plt.savefig(figureFilenamePNG)  # , bbox_inches='tight')
