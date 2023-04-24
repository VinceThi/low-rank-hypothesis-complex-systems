# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from tqdm import tqdm
import glob
from plots.config_rcparams import *

path = "C:/Users/thivi/Documents/GitHub/low-rank-hypothesis-complex-systems/" \
       "singular_values/properties/"

singularValuesFiles_list = glob.glob(
    path + 'singular_values/*_singular_values.txt')

""" Get singular values """
x = np.linspace(0, 1, 1000)

drop_bd = ["board_directors_net1m_2002-06-01", "board_directors_net1m_2002-07-01", "board_directors_net1m_2002-08-01", "board_directors_net1m_2002-09-01", "board_directors_net1m_2002-10-01", "board_directors_net1m_2002-11-01", "board_directors_net1m_2003-02-01", "board_directors_net1m_2003-03-01", "board_directors_net1m_2003-04-01", "board_directors_net1m_2003-05-01", "board_directors_net1m_2003-06-01", "board_directors_net1m_2003-07-01", "board_directors_net1m_2003-08-01", "board_directors_net1m_2003-09-01", "board_directors_net1m_2003-10-01", "board_directors_net1m_2003-11-01", "board_directors_net1m_2004-02-01", "board_directors_net1m_2004-03-01", "board_directors_net1m_2004-04-01", "board_directors_net1m_2004-05-01", "board_directors_net1m_2004-06-01", "board_directors_net1m_2004-07-01", "board_directors_net1m_2004-08-01", "board_directors_net1m_2004-09-01", "board_directors_net1m_2004-10-01", "board_directors_net1m_2004-11-01", "board_directors_net1m_2005-02-01", "board_directors_net1m_2005-03-01", "board_directors_net1m_2005-04-01", "board_directors_net1m_2005-05-01", "board_directors_net1m_2005-06-01", "board_directors_net1m_2005-07-01", "board_directors_net1m_2005-08-01", "board_directors_net1m_2005-09-01", "board_directors_net1m_2005-10-01", "board_directors_net1m_2005-11-01", "board_directors_net1m_2006-02-01", "board_directors_net1m_2006-03-01", "board_directors_net1m_2006-04-01", "board_directors_net1m_2006-05-01", "board_directors_net1m_2006-06-01", "board_directors_net1m_2006-07-01", "board_directors_net1m_2006-08-01", "board_directors_net1m_2006-09-01", "board_directors_net1m_2006-10-01", "board_directors_net1m_2006-11-01", "board_directors_net1m_2007-02-01", "board_directors_net1m_2007-03-01", "board_directors_net1m_2007-04-01", "board_directors_net1m_2007-05-01", "board_directors_net1m_2007-06-01", "board_directors_net1m_2007-07-01", "board_directors_net1m_2007-08-01", "board_directors_net1m_2007-09-01", "board_directors_net1m_2007-10-01", "board_directors_net1m_2007-11-01", "board_directors_net1m_2008-02-01", "board_directors_net1m_2008-03-01", "board_directors_net1m_2008-04-01", "board_directors_net1m_2008-05-01", "board_directors_net1m_2008-06-01", "board_directors_net1m_2008-07-01", "board_directors_net1m_2008-08-01", "board_directors_net1m_2008-09-01", "board_directors_net1m_2008-10-01", "board_directors_net1m_2008-11-01", "board_directors_net1m_2009-02-01", "board_directors_net1m_2009-03-01", "board_directors_net1m_2009-04-01", "board_directors_net1m_2009-05-01", "board_directors_net1m_2009-06-01", "board_directors_net1m_2009-07-01", "board_directors_net1m_2009-08-01", "board_directors_net1m_2009-09-01", "board_directors_net1m_2009-10-01", "board_directors_net1m_2009-11-01", "board_directors_net1m_2010-02-01", "board_directors_net1m_2010-03-01", "board_directors_net1m_2010-04-01", "board_directors_net1m_2010-05-01", "board_directors_net1m_2010-06-01", "board_directors_net1m_2010-07-01", "board_directors_net1m_2010-08-01", "board_directors_net1m_2010-09-01", "board_directors_net1m_2010-10-01", "board_directors_net1m_2010-11-01", "board_directors_net1m_2011-02-01", "board_directors_net1m_2011-03-01", "board_directors_net1m_2011-04-01", "board_directors_net1m_2011-05-01", "board_directors_net1m_2011-06-01", "board_directors_net1m_2011-07-01", "board_directors_net1m_2011-08-01"]
drop_wiki = ["edit_wikibooks_af", "edit_wikibooks_ak", "edit_wikibooks_als", "edit_wikibooks_ang", "edit_wikibooks_as", "edit_wikibooks_ast", "edit_wikibooks_bg", "edit_wikibooks_bi", "edit_wikibooks_bm", "edit_wikibooks_bn", "edit_wikibooks_bo", "edit_wikibooks_ch", "edit_wikibooks_co", "edit_wikibooks_cs", "edit_wikibooks_cv", "edit_wikibooks_el", "edit_wikibooks_eo", "edit_wikibooks_et", "edit_wikibooks_fi", "edit_wikibooks_got", "edit_wikibooks_gu"     , "edit_wikibooks_he", "edit_wikibooks_hi", "edit_wikibooks_hy", "edit_wikibooks_ia", "edit_wikibooks_id", "edit_wikibooks_is", "edit_wikibooks_kk", "edit_wikibooks_km", "edit_wikibooks_kn", "edit_wikibooks_ko", "edit_wikibooks_ks", "edit_wikibooks_ku", "edit_wikibooks_lb", "edit_wikibooks_li", "edit_wikibooks_ln", "edit_wikibooks_lt", "edit_wikibooks_lv", "edit_wikibooks_mg", "edit_wikibooks_mi"     , "edit_wikibooks_mk"     , "edit_wikibooks_ml"     , "edit_wikibooks_mn"     , "edit_wikibooks_mr"     , "edit_wikibooks_ms", "edit_wikibooks_nah"    , "edit_wikibooks_nds"    , "edit_wikibooks_ne", "edit_wikibooks_pa", "edit_wikibooks_ps", "edit_wikibooks_qu", "edit_wikibooks_rm", "edit_wikibooks_ro", "edit_wikibooks_ru", "edit_wikibooks_sa", "edit_wikibooks_se", "edit_wikibooks_simple", "edit_wikibooks_sk", "edit_wikibooks_sl", "edit_wikibooks_sq", "edit_wikibooks_su", "edit_wikibooks_sv", "edit_wikibooks_sw", "edit_wikibooks_ta", "edit_wikibooks_te", "edit_wikibooks_th", "edit_wikibooks_tk", "edit_wikibooks_tl", "edit_wikibooks_tr", "edit_wikibooks_ug", "edit_wikibooks_uk", "edit_wikibooks_uz", "edit_wikibooks_vi", "edit_wikibooks_vo", "edit_wikibooks_wa", "edit_wikibooks_yo", "edit_wikibooks_za", "edit_wikibooks_zh_min_nan", "edit_wikinews_bs", "edit_wikinews_ca", "edit_wikinews_cs", "edit_wikinews_eo", "edit_wikinews_fa", "edit_wikinews_fi"     , "edit_wikinews_he"     , "edit_wikinews_ko"     , "edit_wikinews_nl"     , "edit_wikinews_ro"     , "edit_wikinews_sd"     , "edit_wikinews_sq"     , "edit_wikinews_ta"    , "edit_wikinews_th"    , "edit_wikinews_tr"    , "edit_wikinews_uk"    , "edit_wikiquote_af", "edit_wikiquote_am"     , "edit_wikiquote_ang"    , "edit_wikiquote_ast"    , "edit_wikiquote_az"     , "edit_wikiquote_be"     , "edit_wikiquote_bg"     , "edit_wikiquote_bm"     , "edit_wikiquote_br"     , "edit_wikiquote_bs"     , "edit_wikiquote_ca"     , "edit_wikiquote_co"     , "edit_wikiquote_cs"     , "edit_wikiquote_cy"     , "edit_wikiquote_da"     , "edit_wikiquote_el"     , "edit_wikiquote_eo"     , "edit_wikiquote_eu"     , "edit_wikiquote_fi"     , "edit_wikiquote_ga"     , "edit_wikiquote_gl"     , "edit_wikiquote_gu"     , "edit_wikiquote_he"     , "edit_wikiquote_hi"     , "edit_wikiquote_hr"     , "edit_wikiquote_hu"     , "edit_wikiquote_hy"     , "edit_wikiquote_is"     , "edit_wikiquote_ja"     , "edit_wikiquote_ka"     , "edit_wikiquote_kk"     , "edit_wikiquote_kn"     , "edit_wikiquote_ko"     , "edit_wikiquote_kr"     , "edit_wikiquote_ks"     , "edit_wikiquote_ku"     , "edit_wikiquote_ky"     , "edit_wikiquote_la"     , "edit_wikiquote_lb"     , "edit_wikiquote_li"     , "edit_wikiquote_lt"     , "edit_wikiquote_mr"     , "edit_wikiquote_nl"     , "edit_wikiquote_nn"     , "edit_wikiquote_no"     , "edit_wikiquote_ro"     , "edit_wikiquote_sa"     , "edit_wikiquote_simple", "edit_wikiquote_sk"     , "edit_wikiquote_sl"     , "edit_wikiquote_sq"     , "edit_wikiquote_sr"     , "edit_wikiquote_su"     , "edit_wikiquote_sv"     , "edit_wikiquote_ta"     , "edit_wikiquote_th"     , "edit_wikiquote_tk"     , "edit_wikiquote_tr"     , "edit_wikiquote_tt"     , "edit_wikiquote_ug"     , "edit_wikiquote_uk"     , "edit_wikiquote_ur"     , "edit_wikiquote_uz"     , "edit_wikiquote_vi"     , "edit_wikiquote_vo"     , "edit_wikiquote_wo", "edit_wiktionary_aa"    , "edit_wiktionary_ab"    , "edit_wiktionary_ak"    , "edit_wiktionary_als"   , "edit_wiktionary_am"    , "edit_wiktionary_an"    , "edit_wiktionary_ang"   , "edit_wiktionary_as"    , "edit_wiktionary_av"    , "edit_wiktionary_ay"    , "edit_wiktionary_bh"    , "edit_wiktionary_bi"    , "edit_wiktionary_bm"    , "edit_wiktionary_bn"    , "edit_wiktionary_bo"    , "edit_wiktionary_bs"    , "edit_wiktionary_ch"    , "edit_wiktionary_co"    , "edit_wiktionary_cr"    , "edit_wiktionary_csb"   , "edit_wiktionary_dv"    , "edit_wiktionary_dz"    , "edit_wiktionary_fy"    , "edit_wiktionary_ga"    , "edit_wiktionary_gd"    , "edit_wiktionary_gn"    , "edit_wiktionary_gu"    , "edit_wiktionary_gv"    , "edit_wiktionary_ha"    , "edit_wiktionary_hsb"   , "edit_wiktionary_ia"    , "edit_wiktionary_ie"    , "edit_wiktionary_ik"    , "edit_wiktionary_iu"    , "edit_wiktionary_jbo"   , "edit_wiktionary_ka"    , "edit_wiktionary_kk"    , "edit_wiktionary_kl"    , "edit_wiktionary_ks"    , "edit_wiktionary_kw"    , "edit_wiktionary_lb"    , "edit_wiktionary_ln"    , "edit_wiktionary_lv"    , "edit_wiktionary_mh"    , "edit_wiktionary_mk"    , "edit_wiktionary_mo"    , "edit_wiktionary_mr"    , "edit_wiktionary_ms"    , "edit_wiktionary_mt"    , "edit_wiktionary_na"    , "edit_wiktionary_nah"   , "edit_wiktionary_ne"    , "edit_wiktionary_nn"    , "edit_wiktionary_om"    , "edit_wiktionary_pa"    , "edit_wiktionary_pnb"   , "edit_wiktionary_qu"    , "edit_wiktionary_rm"    , "edit_wiktionary_rn", "edit_wiktionary_rw"    , "edit_wiktionary_sc"    , "edit_wiktionary_sd"    , "edit_wiktionary_sg"    , "edit_wiktionary_si"    , "edit_wiktionary_sl"    , "edit_wiktionary_sm"    , "edit_wiktionary_sn"    , "edit_wiktionary_so"    , "edit_wiktionary_ss", "edit_wiktionary_su"    , "edit_wiktionary_sw"    , "edit_wiktionary_ti"    , "edit_wiktionary_tk"    , "edit_wiktionary_tn"    , "edit_wiktionary_to"    , "edit_wiktionary_tpi"   , "edit_wiktionary_ts"    , "edit_wiktionary_tt"    , "edit_wiktionary_tw"    , "edit_wiktionary_ug"    , "edit_wiktionary_ur", "edit_wiktionary_wo"    , "edit_wiktionary_xh"    , "edit_wiktionary_yi"    , "edit_wiktionary_yo"    , "edit_wiktionary_za"]
drop_ego = ["ego_social_gplus_100521671383026672718", "ego_social_gplus_100535338638690515335", "ego_social_gplus_100637660947564674695", "ego_social_gplus_100668989009254813743", "ego_social_gplus_100715738096376666180", "ego_social_gplus_100720409235366385249", "ego_social_gplus_100962871525684315897", "ego_social_gplus_101130571432010257170", "ego_social_gplus_101133961721621664586", "ego_social_gplus_101185748996927059931", "ego_social_gplus_101263615503715477581", "ego_social_gplus_101373961279443806744", "ego_social_gplus_101499880233887429402", "ego_social_gplus_101541879642294398860", "ego_social_gplus_101560853443212199687", "ego_social_gplus_101626577406833098387", "ego_social_gplus_101848191156408080085", "ego_social_gplus_101997124338642780860", "ego_social_gplus_102170431816592344972", "ego_social_gplus_102340116189726655233", "ego_social_gplus_102615863344410467759", "ego_social_gplus_102778563580121606331", "ego_social_gplus_103236949470535942612", "ego_social_gplus_103241736833663734962", "ego_social_gplus_103251633033550231172", "ego_social_gplus_103338524411980406972", "ego_social_gplus_103503116383846951534", "ego_social_gplus_103537112468125883734", "ego_social_gplus_103752943025677384806", "ego_social_gplus_103892332449873403244", "ego_social_gplus_104105354262797387583", "ego_social_gplus_104226133029319075907", "ego_social_gplus_104290609881668164623", "ego_social_gplus_104607825525972194062", "ego_social_gplus_104672614700283598130", "ego_social_gplus_104905626100400792399", "ego_social_gplus_104917160754181459072", "ego_social_gplus_104987932455782713675", "ego_social_gplus_105565257978663183206", "ego_social_gplus_105646458226420473639", "ego_social_gplus_106186407539128840569", "ego_social_gplus_106228758905254036967", "ego_social_gplus_106328207304735502636", "ego_social_gplus_106382433884876652170", "ego_social_gplus_106417861423111072106", "ego_social_gplus_106724181552911298818", "ego_social_gplus_106837574755355833243", "ego_social_gplus_107013688749125521109", "ego_social_gplus_107040353898400532534", "ego_social_gplus_107203023379915799071", "ego_social_gplus_107223200089245371832", "ego_social_gplus_107296660002634487593", "ego_social_gplus_107362628080904735459", "ego_social_gplus_107459220492917008623", "ego_social_gplus_107489144252174167638", "ego_social_gplus_107965826228461029730", "ego_social_gplus_108156134340151350951", "ego_social_gplus_108404515213153345305", "ego_social_gplus_108541235642523883716", "ego_social_gplus_108883879052307976051", "ego_social_gplus_109130886479781915270", "ego_social_gplus_109213135085178239952", "ego_social_gplus_109327480479767108490", "ego_social_gplus_109342148209917802565", "ego_social_gplus_109596373340495798827", "ego_social_gplus_109602109099036550366", "ego_social_gplus_110232479818136355682", "ego_social_gplus_110241952466097562819", "ego_social_gplus_110538600381916983600", "ego_social_gplus_110581012109008817546", "ego_social_gplus_110614416163543421878", "ego_social_gplus_110701307803962595019", "ego_social_gplus_110739220927723360152", "ego_social_gplus_110809308822849680310", "ego_social_gplus_110971010308065250763", "ego_social_gplus_111048918866742956374", "ego_social_gplus_111058843129764709244", "ego_social_gplus_111091089527727420853", "ego_social_gplus_111213696402662884531", "ego_social_gplus_112317819390625199896", "ego_social_gplus_112463391491520264813", "ego_social_gplus_112573107772208475213", "ego_social_gplus_112724573277710080670", "ego_social_gplus_112737356589974073749", "ego_social_gplus_112787435697866537461", "ego_social_gplus_113112256846010263985", "ego_social_gplus_113122049849685469495", "ego_social_gplus_113171096418029011322", "ego_social_gplus_113356364521839061717", "ego_social_gplus_113455290791279442483", "ego_social_gplus_113597493946570654755", "ego_social_gplus_113718775944980638561", "ego_social_gplus_113799277735885972934", "ego_social_gplus_113881433443048137993", "ego_social_gplus_114054672576929802335", "ego_social_gplus_114104634069486127920", "ego_social_gplus_114122960748905067938", "ego_social_gplus_114124942936679476879", "ego_social_gplus_114147483140782280818", "ego_social_gplus_114336431216099933033", "ego_social_gplus_115121555137256496805", "ego_social_gplus_115273860520983542999", "ego_social_gplus_115360471097759949621", "ego_social_gplus_115455024457484679647", "ego_social_gplus_115516333681138986628", "ego_social_gplus_115573545440464933254", "ego_social_gplus_115576988435396060952", "ego_social_gplus_115625564993990145546", "ego_social_gplus_116059998563577101552", "ego_social_gplus_116247667398036716276", "ego_social_gplus_116315897040732668413", "ego_social_gplus_116450966137824114154", "ego_social_gplus_116807883656585676940", "ego_social_gplus_116825083494890429556", "ego_social_gplus_116899029375914044550", "ego_social_gplus_116931379084245069738", "ego_social_gplus_117412175333096244275", "ego_social_gplus_117503822947457399073", "ego_social_gplus_117668392750579292609", "ego_social_gplus_117734260411963901771", "ego_social_gplus_117798157258572080176", "ego_social_gplus_117866881767579360121", "ego_social_gplus_118107045405823607895", "ego_social_gplus_118255645714452180374", "ego_social_gplus_118379821279745746467"]

network_count = 0
singularValues_array = np.ones(len(x))

""" Extract singular values from Netzschleuder """
for networkName in tqdm(singularValuesFiles_list):

    networkName = networkName.split('_singular_values')[0].split('\\')[-1]
    if networkName in drop_bd:
        pass
    elif networkName in drop_wiki:
        pass
    elif networkName in drop_ego:
        pass
    else:
        singularValuesFilename = path + 'singular_values/' + networkName\
                                 + '_singular_values.txt'
        singularValues = np.loadtxt(singularValuesFilename)

        N = len(singularValues)
        indices = np.arange(0, N, 1)/(N-1)
        singularValues = singularValues/np.max(singularValues)
        singularValues = np.interp(x, indices, singularValues)
        singularValues_array = np.vstack((singularValues_array,
                                          singularValues))
        network_count += 1


""" Extract singular values from other sources """
for networkName in tqdm(glob.glob(path + '*_singular_values.txt')):
    networkName = networkName.split('_singular_values')[0].split('\\')[-1]

    singularValuesFilename = path + networkName + '_singular_values.txt'
    singularValues = np.loadtxt(singularValuesFilename)
    """ Note that the zero singular values must be included here to compute
            the size of the networks with len(singularValues). """

    N = len(singularValues)
    indices = np.arange(0, N, 1)/(N-1)
    singularValues = singularValues/np.max(singularValues)
    singularValues = np.interp(x, indices, singularValues)
    singularValues_array = np.vstack((singularValues_array,
                                      singularValues))

    network_count += 1

print(f"{network_count} networks were used.")


""" Get statistics on the singular values """
mean_singularValues = np.mean(singularValues_array, axis=0)
q5_singularValues = np.percentile(singularValues_array, q=5, axis=0)
q50_singularValues = np.percentile(singularValues_array, q=50, axis=0)
q95_singularValues = np.percentile(singularValues_array, q=95, axis=0)
q96_singularValues = np.percentile(singularValues_array, q=96, axis=0)


""" Get bounds through fits on the singular values """


def ratio_function(x, params):
    return (1 - x)**(params[0]-2)/(1 + params[1]*x)**(params[2])


def mu(params):
    return (1 - params[1]) / params[0]


def sigma(params):
    return 1 / params[0]


def objective_function(params, x, y, norm_choice):
    return norm(y - ratio_function(x, params), norm_choice)


bounds5 = ((2, 10), (0.01, 5000), (0.01, 10))
boundsavg = ((2, 10), (0.01, 1000), (0.01, 10))
bounds50 = ((2, 10), (0.01, 1000), (0.01, 10))
bounds95 = ((2, 10), (0.01, 1000), (0.01, 10))

cte95 = len(x)
args = (x[:cte95], q95_singularValues[:cte95], 2)
reg95 = minimize(objective_function, np.array([1, 2, 1]), args,
                 bounds=bounds95)
ratio95 = ratio_function(x, reg95.x)
print(f"params 95-th percentile : {reg95.x})")

cteavg = len(x)
args = (x[:cteavg], mean_singularValues[:cteavg], 2)
regavg = minimize(objective_function, np.array([1, 2, 1]), args,
                  bounds=boundsavg)
ratioavg = ratio_function(x, regavg.x)
print(f"params average : {regavg.x}")

cte50 = len(x)
args = (x[:cteavg], q50_singularValues[:cte50], 2)
reg50 = minimize(objective_function, np.array([1, 2, 1]), args,
                 bounds=boundsavg)
ratio50 = ratio_function(x, reg50.x)
print(f"params 50-th percentile : {reg50.x}")


cte5 = len(x)
args = (x[:cte5], q5_singularValues[:cte5], 2)
reg5 = minimize(objective_function, np.array([1, 2, 1]), args,
                bounds=bounds5)
ratio5 = ratio_function(x, reg5.x)
print(f"params 5-th percentile : {reg5.x}")

count_vas_below95 = 0
count_networks_below95 = 0
count_networks_below95_with_tol = 0
for vas in singularValues_array:
    count_vas_below95 += np.count_nonzero(vas <= ratio95)
    if np.all(vas <= ratio95 + 0.000001):
        count_networks_below95 += 1
    if np.count_nonzero(vas <= ratio95) > 995:
        count_networks_below95_with_tol += 1
percentage_vas_below95 = \
    int(np.round(100*count_vas_below95/(network_count * len(x))))
percentage_networks_below95 = \
    int(np.round(100*count_networks_below95/network_count))
print(100*count_vas_below95/(network_count * len(x)))
percentage_networks_below95_with_tol = \
    int(np.round(100*count_networks_below95_with_tol/network_count))
print(f"\n{percentage_vas_below95}% of singular values of all the networks"
      f" below the bound")
print(f"\n{percentage_networks_below95}% of the networks have all their"
      f" singular values below the bound")
print(f"\n{percentage_networks_below95_with_tol}% of the networks have 99.5%"
      f" of their singular values below the bound")

plt.figure(figsize=(5, 5))
ax = plt.gca()
plt.plot(x, mean_singularValues, color=deep[0], linewidth=2,
         label="Average", zorder=10)
fcolor = "#38ccf9"   # deep[9]
# plt.plot(x, ratioavg, color=fcolor, linestyle="-",
#          linewidth=1, zorder=20)
plt.plot(x, ratio50, color=fcolor, linestyle="-",
         linewidth=1, zorder=20)
# plt.plot(x, ratio5, color=fcolor, linestyle="-",
#          linewidth=1, zorder=20)
plt.plot(x, ratio95, color=fcolor, linestyle="-",
         linewidth=1, zorder=20,  # Experimental bound
         label="Envelope $\\frac{(1 - x)^{c-2}}"
               "{(1 + \\zeta x)^b}$")
# "Experimental\nupper bound"

pcolor = dark_grey
plinewidth = 0.5
ptcolor = "#fafafa"
# plt.text(x[cte95 // 10] + x[cte95 // 10] / 2, ratio95[cte95 // 10],
#          f"$a \\approx {np.round(reg95.x[0], 1)}$\n"
#          f"$b \\approx {np.round(reg95.x[1], 1)}$\n"
#          f"$c \\approx {np.round(reg95.x[2], 1)}$",
#          fontsize=8)
t95 = plt.text(0.66, 0.073,
               f"{percentage_vas_below95}%\nof the\nsingular values",
               fontsize=8)  # , ha="center"
t95.set_bbox(dict(facecolor=ptcolor, alpha=1, linewidth=0))
plt.plot(x, q5_singularValues, color=pcolor, linewidth=plinewidth,
         linestyle="--", label="Percentiles")
t5 = plt.text(0.02, 0.02, f"5%", fontsize=8)
t5.set_bbox(dict(facecolor=ptcolor, alpha=1, linewidth=0))
plt.plot(x, q95_singularValues, color=pcolor, linewidth=plinewidth,
         linestyle="--")
t95 = plt.text(x[500], q95_singularValues[500]-q95_singularValues[500]/4.5,
               f"95%", fontsize=8)
t95.set_bbox(dict(facecolor=ptcolor, alpha=1, linewidth=0))
for q in [20, 50, 80]:
    percentile = np.percentile(singularValues_array, q=q, axis=0)
    plt.plot(x, percentile, color=pcolor, linewidth=plinewidth,
             linestyle="--")
    t = plt.text(x[q*10//2], percentile[q*10//2]-percentile[q*10//2]/10,
                 f"{q}%", fontsize=8)
    t.set_bbox(dict(facecolor=ptcolor, alpha=1, linewidth=0))
# plt.fill_between(x, mean_singularValues - std_singularValues,
#                  mean_singularValues + std_singularValues, alpha=0.5)
plt.fill_between(x, q5_singularValues, q95_singularValues, color=ptcolor,
                 alpha=1)

plt.tick_params(axis='y', which='both', right=False)
plt.yscale("log")
plt.ylim([0.001, 1.1])
plt.xlabel("Rescaled singular value index")
plt.ylabel("Rescaled singular values")
plt.legend(loc=1, fontsize=10)
handles, labels = ax.get_legend_handles_labels()
order = [0, 2, 1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc=1, fontsize=10)
plt.show()
