#!/usr/bin/env python3

"""
This script is used for making the plots of the AR as a function of p in the manuscript.
Note that in my code kappa is named epsilon.
Run with "$ PYTHONHASHSEED=0 python3 p_plot.py" to get consistent hash values in python 3.3.
"""

import sys

sys.path.insert(1, "../")
import numpy as np
import qem
import chainer as ch
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rc
import _HQAOA
import os

os.environ[
    "PYTHONHASHSEED"
] = "0"  # To get the same hash values every time new python process is started.
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 9})
rc("text", usetex=True)


def get_p_list(
    path, instance, d, n_iter, ep
):  # Get the list of available p values in path+instance at the given d,n_iter and ep. Sort the list.
    def filter_data(data, d, n_iter, path, ep, n):
        def conditions(line):
            cond = [
                line[0]["d"] == d,
                line[0]["n_iter"] == n_iter,
                line[0]["direction"] == path[-2:-10:-1][::-1]
                or line[0]["direction"]
                == path[-2:-9:-1][::-1],  # Check line is in the right path.
                line[0]["ep"] == ep,
                line[1]["n"] == n,
            ]
            return all(cond)

        filtered = [line for line in data if conditions(line)]
        return filtered

    data = _HQAOA.import_instance_data(path + instance)
    data = filter_data(data, d, n_iter, path, ep, n)
    p_data = list(set([line[0]["p"] for line in data]))
    p_data.sort()
    return p_data


def plot_points(unaware_path, path, ipath, d, n_iter, ep):
    pltpts = []
    E0 = _HQAOA.get_instance_E0(path + ipath)
    data = _HQAOA.import_instance_data(path + ipath)
    p_list = get_p_list(path, ipath, d, n_iter, ep)
    if p_list[0] != 0:
        p_list = [0] + p_list  # Allways add p as a plot point.

    for p in p_list:  # Per p, take the line with the smallest cost
        if (
            p == 0
        ):  # For p=0, allways look in the noise_unaware folder, because at p=0, results are the same for any ep and at p=0 I did not recompute for every ep.
            best_cost = _HQAOA.load_optimal_line(unaware_path + instance, p, 0, d)[2][
                "cost_qaoa"
            ]
        else:
            best_cost = _HQAOA.load_optimal_line(path + instance, p, ep, d)[2][
                "cost_qaoa"
            ]
        best_AR = best_cost / E0
        pltpts.append([p, best_AR])

    pltpts.sort()
    pltpts = np.array(pltpts).transpose()

    np.save(
        "plot_points/lone_p_plot_plot_points_noise_aware_{}.npy".format(
            hash((path, instance, d, n_iter, ep))
        ),
        pltpts,
    )
    return pltpts


def plot_points_noise_unaware(
    unaware_path, instance, n, d, n_iter, direc, ep
):  # Take the optimal angles at p=0, and use them to obtain an AR at other p.
    pltpts = []
    E0 = _HQAOA.get_instance_E0(unaware_path + instance)
    con = _HQAOA.get_instance_con(unaware_path + instance)
    p_list = get_p_list(
        unaware_path, instance, d, n_iter, ep
    )  # get the p for which I ran noise aware optimization.
    if p_list[0] != 0:
        p_list = [0] + p_list  # Allways add p as a plot point.

    for p in p_list:  # Use optimal angles at p=0, put them through noisy chan.
        opt_pars = _HQAOA.load_optimal_line(unaware_path + instance, 0, 0, d)[2][
            "opt_parameters"
        ]  # Load optimal parameters at p,ep=0.
        opt_pars = ch.Variable(np.array(opt_pars))
        if direc == "temporal":
            cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
                con, n, d, p, ep, opt_pars
            ).data
        elif direc == "spatial":
            cost = _HQAOA.cost_from_parameters_swap_spatial(
                con, n, d, p, ep, opt_pars
            ).data

        AR = cost / E0
        pltpts.append([p, AR])

    pltpts.sort()
    pltpts = np.array(pltpts).transpose()

    np.save(
        "plot_points/lone_p_plot_plot_points_noise_unaware_{}.npy".format(
            hash((unaware_path, instance, n, d, n_iter, direc, ep))
        ),
        pltpts,
    )
    return pltpts


def plotter(
    direc_path,
    direc,
    instance,
    unaware_path,
    n_iter,
    d,
    ep,
    label1,
    label2,
    ax,
    _color,
    man_p_list,
):
    # If you made new data points, you should remove the npy files. Only in that case the plot points are computed anew.
    fname = "plot_points/lone_p_plot_plot_points_noise_aware_{}.npy".format(
        hash((direc_path, instance, d, n_iter, ep))
    )
    if os.path.isfile(fname):
        print("Loading plot points")
        points = np.load(fname)
        if len(man_p_list) != 0:
            points = points.transpose()
            points = [point for point in points if point[0] in man_p_list]
            points = np.array(points)
            points = points.transpose()
    else:
        print("Calculating plot points, aware")
        points = plot_points(unaware_path, direc_path, instance, d, n_iter, ep)

    ax.plot(
        *points,
        "-o",
        alpha=0.75,
        color=_color,
        markerfacecolor="none",
        markersize=7,
        label=label1,
        solid_capstyle="round"
    )

    fname = "plot_points/lone_p_plot_plot_points_noise_unaware_{}.npy".format(
        hash((unaware_path, instance, n, d, n_iter, direc, ep))
    )
    if os.path.isfile(fname):
        points = np.load(fname)
        print("Loading plot points")
        if len(man_p_list) != 0:
            points = points.transpose()
            points = [point for point in points if point[0] in man_p_list]
            points = np.array(points)
            points = points.transpose()
    else:
        print("Calculating plot points, unaware")
        points = plot_points_noise_unaware(
            unaware_path, instance, n, d, n_iter, direc, ep
        )

    ax.plot(
        *points,
        "-o",
        alpha=0.75,
        color=_color,
        markeredgecolor="none",
        markersize=3.7,
        label=label2,
        solid_capstyle="round"
    )


def random_guess_AR(path, ipath):
    E0 = _HQAOA.get_instance_E0(path + ipath)
    con = _HQAOA.get_instance_con(path + ipath)
    n = 6
    d = 0
    p = 0
    ep = 0
    pars = ch.Variable(np.array([-1.0, 1.0, -1.0, 0.0, 0.0, 1.0]))
    cost = _HQAOA.cost_from_parameters_swap_temporal_quantum_classical(
        con, n, d, p, ep, pars
    ).data
    return cost / E0


# random_guess_AR('../../data/SK/SWAP-network/6/spatial/','010010100111110')

if __name__ == "__main__":
    master_path = "../data/SK/SWAP-network/6/"  # path to all instance data.
    path_temp = master_path + "temporal/"  # Path to temporal noise correlation data.
    path_spat = master_path + "spatial/"  # Path to spatial noise correlation data.
    instance = "010010100111110"  # Instance name.
    unaware_path = path_spat  # Where to look for data if p=0, for any ep.
    n_iter = (
        4  # Assert later every data point has used n_iter bassinhopping iterations.
    )
    n = 6  # Assert later data is from n=6 qubits.
    d = 3  # The number of cycles to get the data from.

    fig, ax = plt.subplots(2, figsize=(3.75, 6))
    colors = []
    for i in range(10):
        colors.append(next((ax[0]._get_lines.prop_cycler))["color"])
    ax[0].set_ylabel("AR")
    ax[0].set_xlabel("$p$")
    ax[1].set_ylabel("AR")
    ax[1].set_xlabel("$p$")
    man_p_list = np.around(np.arange(0, 0.52, 0.02), 5)

    #### Global plot ####
    ep = 1  # Correlation at which first and second plots are shown.

    direc_path = path_temp
    direc = "temporal"
    label1 = "Temp. cor., aware"
    label2 = "Temp. cor., unaware"

    plotter(
        direc_path,
        direc,
        instance,
        unaware_path,
        n_iter,
        d,
        ep,
        label1,
        label2,
        ax[0],
        colors[1],
        man_p_list,
    )

    direc_path = path_spat
    direc = "spatial"
    label1 = "Spat. cor., aware"
    label2 = "Spat. cor., unaware"

    plotter(
        direc_path,
        direc,
        instance,
        unaware_path,
        n_iter,
        d,
        ep,
        label1,
        label2,
        ax[0],
        colors[2],
        man_p_list,
    )

    ep = 0  # Correlation at which third plot is shown.

    direc_path = path_spat
    direc = "spatial"
    label1 = "Uncor., aware"
    label2 = "Uncor., unaware"

    plotter(
        direc_path,
        direc,
        instance,
        unaware_path,
        n_iter,
        d,
        ep,
        label1,
        label2,
        ax[0],
        colors[0],
        man_p_list,
    )

    # ax.legend(loc='upper left', bbox_to_anchor=(0.0,0.81))
    ax[0].legend()

    ######### Zoomed in plot ##########

    man_p_list = np.around(np.arange(0, 0.01002, 0.0004), 5)

    ep = 1  # Correlation at which first and second plots are shown.

    direc_path = path_temp
    direc = "temporal"
    label1 = "Temp. cor., aware"
    label2 = "Temp. cor., unaware"

    plotter(
        direc_path,
        direc,
        instance,
        unaware_path,
        n_iter,
        d,
        ep,
        label1,
        label2,
        ax[1],
        colors[1],
        man_p_list,
    )

    direc_path = path_spat
    direc = "spatial"
    label1 = "Spat. cor., aware"
    label2 = "Spat. cor., unaware"

    plotter(
        direc_path,
        direc,
        instance,
        unaware_path,
        n_iter,
        d,
        ep,
        label1,
        label2,
        ax[1],
        colors[2],
        man_p_list,
    )

    ep = 0  # Correlation at which third plot is shown.

    direc_path = path_spat
    direc = "spatial"
    label1 = "Uncor., aware"
    label2 = "Uncor., unaware"

    plotter(
        direc_path,
        direc,
        instance,
        unaware_path,
        n_iter,
        d,
        ep,
        label1,
        label2,
        ax[1],
        colors[0],
        man_p_list,
    )

    #### Crosses ####
    # Add extrapolated AR datapoints found with derivatives.py for p=0.001
    # For p=0.001
    # temp_AR=[0.001,0.866452441411907] # [p,AR]
    # spat_AR=[0.001,0.8506262123039643]
    # uncor_AR=[0.001,0.8267500487250729]

    # Same for p=0.01
    # temp2_AR=[0.01,0.8175971945193308] # [p,AR]
    # spat2_AR=[0.01,0.6593349034399043]
    # uncor2_AR=[0.01,0.4205732676509891]

    # pts=np.array([temp_AR,spat_AR,uncor_AR,temp2_AR,spat2_AR,uncor2_AR]).transpose()
    # ax[1].scatter(*pts,marker="x",color='black',s=20,zorder=1000,alpha=0.75,capstyle='round')

    #### Dashed lines ####
    E0 = _HQAOA.get_instance_E0(unaware_path + instance)
    ARp0 = -6.103165615244241 / E0
    # Data from derivatives.py
    p_list_lin = [0, 0.01]
    (dash,) = ax[1].plot(
        p_list_lin,
        [ARp0, ARp0 + 0.01 * (-5.428360765841799)],
        linestyle="dashed",
        color="grey",
        markeredgecolor="black",
        zorder=-1,
        label="Linearized AR.",
        dash_capstyle="round",
    )  # Temp. cor.
    ax[1].plot(
        p_list_lin,
        [ARp0, ARp0 + 0.01 * (-21.254589873784447)],
        linestyle="dashed",
        color="grey",
        zorder=-1,
        dash_capstyle="round",
    )  # Spat. cor.
    ax[1].plot(
        p_list_lin,
        [ARp0, ARp0 + 0.01 * (-45.13075345267596)],
        linestyle="dashed",
        color="grey",
        zorder=-1,
        dash_capstyle="round",
    )  # Uncor.

    # Vertical lines
    ax[1].axvline(0.001, color="lightgray", zorder=-2)
    ax[1].axvline(0.01, color="lightgray", zorder=-2)

    # Add legend only containing dashed lines in lower plot
    ax[1].legend(handles=[dash])

    #### Save Fig ####
    fig.savefig("plots/p_plot.pdf", bbox_inches="tight", pad_inches=0.005)
