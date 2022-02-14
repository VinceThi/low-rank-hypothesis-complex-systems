from plots.config_rcparams import *
import json


def plot_bifurcation_diagram(coupling_constants, xf, xb, Xf, Xb):
    """

    :param coupling_constants:
    :param xf: forward equilibrium points list
    :param xb: backward equilibrium points list
    :param Xf:
    :param Xb:
    :return:
    """
    plt.figure(figsize=(4, 4))
    plt.subplot(111)
    plt.plot(coupling_constants, xf,
             color=first_community_color, label="Complete")
    plt.plot(coupling_constants, xb,
             color=first_community_color)
    plt.plot(coupling_constants, Xf,
             color=second_community_color, label="Reduced")
    plt.plot(coupling_constants, Xb,
             color=second_community_color)
    plt.ylabel('Global activity equilibrium point $X^*$')
    plt.xlabel('Coupling constant')
    # plt.ylim([-0.02, 1.02])
    plt.tick_params(axis='both', which='major')
    plt.legend(loc=4, fontsize=fontsize_legend)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = 'C:/Users/thivi/Documents/GitHub/low-dimension-hypothesis/' \
            'simulations/simulations_data/microbial_data/'
    xf_path = "2022_02_07_18h52min02sec_n_50_missed_x_forward_equilibrium" \
              "_points_list_complete_microbial_gut.json"
    xb_path = "2022_02_07_18h52min02sec_n_50_missed_x_backward_equilibrium" \
              "_points_list_complete_microbial_gut.json"
    Xf_path = "2022_02_07_18h52min02sec_n_50_missed_redx_forward_equilibrium" \
              "_points_list_reduced_microbial_gut.json"
    Xb_path = "2022_02_07_18h52min02sec_n_50_missed_redx_backward_equilibrium"\
              "_points_list_reduced_microbial_gut.json"
    parameters_path = "2022_02_07_18h52min02sec_n_50_missed_microbial" \
                      "_gut_parameters_dictionary.json"

    with open(path + xf_path) as json_data:
        xf = json.load(json_data)
    with open(path + xb_path) as json_data:
        xb = json.load(json_data)
    with open(path + Xf_path) as json_data:
        Xf = json.load(json_data)
    with open(path + Xb_path) as json_data:
        Xb = json.load(json_data)
    with open(path + parameters_path) as json_data:
        parameters_dict = json.load(json_data)

    coupling_constants = parameters_dict["coupling_constants"]

    plot_bifurcation_diagram(coupling_constants, xf, xb, Xf, Xb)
