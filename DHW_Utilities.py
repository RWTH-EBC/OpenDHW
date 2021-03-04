# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def convert_dhw_load_to_storage_load(dhw_demand, dir_output, s_step=600, V_stor=300, dT_stor=55, dT_threshhold=10,
                                     Qcon_flow_max=5000, plot_cum_demand=False, with_losses=True,
                                     start_plot='2019-08-08-18', end_plot='2019-08-09-09',
                                     save_fig=True):
    """
    Converts the input DHW-Profile without a DHW-Storage to a DHW-Profile with a DHW-Storage.
    The output profile looks as if the HP would not supply the DHW-load directly but would rather re-heat
    the DHW-Storage, which has dropped below a certain dT Threshold.
    The advantage is, that no storage model has to be part of a dynamic simulation,
    although the heatpump still acts as if a storage is supplied. Based on DIN EN 12831-3.

    :param dhw_demand:      List, stores the DHW-demand profile in [W] per Timestep
    :param dir_output:      Directory where to save the plot
    :param s_step:          Seconds within a timestep. Usual Values are 3600 (1h timesteps) or 600 (10min timesteps)
    :param V_stor:          Storage Volume in Liters
    :param dT_stor:         max dT in Storage
    :param dT_threshhold:   max dT Drop before Storage needs to be re-heated
    :param Qcon_flow_max:   Heat Flow Rate at the Heatpump when refilling the Storage in [W]
    :param plot_cum_demand: Plot the cumulative "Summenliniendiagram" as described in DIN DIN EN 12831-3
    :param with_losses:     Boolean if the storage should have losses
    :param start_plot:      Pandas Datetime where the Plot should start, e.g. '2019-08-02'
    :param end_plot:        Pandas Datetime where the plot should end, e.g. '2019-08-03'
    :param save_fig:        decide to save the fig as a pdf and png in dir_output
    :return: storage_load:  DHW-profile that re-heats a storage.
    """

    # convert the DHW demand from Watt to Joule by multiplying by the timestep width
    dhw_demand = [dem_step * s_step for dem_step in dhw_demand]

    # --------- Storage Data ---------------
    # Todo: think about how Parameters should be for Schichtspeicher
    rho = 1  # Liters to Kilograms
    m_w = V_stor * rho  # Mass Water in Storage
    c_p = 4180  # Heat Capacity Water in [J/kgK]
    Q_full = m_w * c_p * dT_stor
    dQ_threshhold = m_w * c_p * dT_threshhold
    Q_dh_timestep = Qcon_flow_max * s_step  # energy added in 1 timestep
    # Todo: implement a ramp-up period?

    # ---------- write storage load time series, with Losses --------
    Q_storr_curr = Q_full  # tracks the Storage Filling
    storage_load = []  # new time series
    storage_level = []
    loss_load = []
    fill_storage = False

    for t_step, dem_step in enumerate(dhw_demand, start=0):
        storage_level.append(Q_storr_curr)
        if with_losses:
            Q_loss = (Q_storr_curr * 0.001 * s_step) / 3600  # 0,1% Loss per Hour
        else:
            Q_loss = 0
        loss_load.append(Q_loss)

        # for initial condition, when storage_load is still empty
        if len(storage_load) == 0:
            Q_storr_curr = Q_storr_curr - dem_step - Q_loss
        else:
            Q_storr_curr = Q_storr_curr - dem_step - Q_loss + storage_load[t_step - 1]

        if Q_storr_curr >= Q_full:  # storage full, dont fill it!
            fill_storage = False
            storage_load.append(0)
            continue

        # storage above dT Threshhold, but not full. depending if is charging or discharging, storage_load is appended
        elif Q_storr_curr > Q_full - dQ_threshhold:
            if fill_storage:
                storage_load.append(Q_dh_timestep)
            else:
                storage_load.append(0)
                continue

        else:  # storage below dT Threshhold, fill it!
            fill_storage = True
            storage_load.append(Q_dh_timestep)

    # print out total demands and Difference between them
    print("Total DHW Demand is {:.2f} kWh".format(sum(dhw_demand) / (3600 * 1000)))
    print("Total Storage Demand is {:.2f} kWh".format(sum(storage_load) / (3600 * 1000)))
    diff = sum(dhw_demand) + sum(loss_load) - sum(storage_load)
    print("Difference between dhw demand and storage load ist {:.2f} kWh".format(diff / (3600 * 1000)))
    if diff < 0:
        print("More heat than dhw demand is added to the storage in loss-less mode!")

    # Count number of clusters of non-zero values ("peaks"). One Peak is comprised by 2 HP mode switches.
    dhw_peaks = int(np.diff(np.concatenate([[0], dhw_demand, [0]]) == 0).sum() / 2)
    stor_peaks = int(np.diff(np.concatenate([[0], storage_load, [0]]) == 0).sum() / 2)
    print("The Storage reduced the number of DHW heating periods from {} to {}, which is equal to "
          "{:.2f} and {:.2f} per day, respectively.".format(dhw_peaks, stor_peaks, dhw_peaks / 365, stor_peaks / 365))

    # draw cumulative demand (german: "Summenlinien")
    dhw_demand_sumline = []
    acc_dem = 0  # accumulated demand
    for dem_step in dhw_demand:
        acc_dem += dem_step
        dhw_demand_sumline.append(acc_dem)

    storage_load_sumline = []
    acc_load = 0  # accumulated load
    for i, stor_step in enumerate(storage_load):
        acc_load += stor_step - loss_load[i]
        storage_load_sumline.append(acc_load)
    storage_load_sumline = [Q + Q_full for Q in storage_load_sumline]

    # Todo: Fill storage so that at the end of the year its full again
    fill_storage = False
    if fill_storage:
        last_zero_index = None
        for idx, item in enumerate(reversed(storage_load), start=0):
            if item == 0:
                last_zero_index = idx
        storage_load[last_zero_index] += diff

    # Plot the cumulative demand
    if plot_cum_demand:

        # use RWTH Colors
        rwth_blue = "#00549F"
        rwth_blue_50 = "#8EBAE5"
        rwth_green = "#57AB27"
        rwth_green_50 = "#B8D698"
        rwth_orange = "#F6A800"
        rwth_orange_50 = "#FDD48F"
        rwth_red = "#CC071E"
        rwth_red_50 = "#E69679"
        rwth_yellow = "#FFED00"
        rwth_yellow_50 = "#FFF59B"
        rwth_colors_all = [rwth_blue, rwth_green, rwth_orange, rwth_red, rwth_yellow, rwth_blue_50, rwth_green_50,
                           rwth_orange_50, rwth_red_50, rwth_yellow_50]
        sns.set_palette(sns.color_palette(rwth_colors_all))  # does nothing? specify colors with palette=[c1, c2..]

        if platform.system() == 'Darwin':
            dir_home = "/Users/jonasgrossmann"
        elif platform.system() == 'Windows':
            dir_home = "D:/mma-jgr"
        else:
            raise Exception("Unkown Operating System")
        plt.style.use(dir_home + "/git_repos/matplolib-style/ebc.paper.mplstyle")
        sns.set()
        sns.set_style("white")
        sns.set_context("paper")

        # set date range to simplify plot slicing
        date_range = pd.date_range(start='2019-01-01', end='2020-01-01', freq=str(s_step) + 'S')
        date_range = date_range[:-1]

        # convert Joule values to kWh or kW
        dhw_demand_sumline_kWh = [dem_step / (3600 * 1000) for dem_step in dhw_demand_sumline]
        storage_load_sumline_kWh = [stor_step / (3600 * 1000) for stor_step in storage_load_sumline]
        dhw_demand_kW = [dem_step / (s_step * 1000) for dem_step in dhw_demand]
        storage_load_kW = [stor_step / (s_step * 1000) for stor_step in storage_load]
        losses_W = [loss_step / s_step for loss_step in loss_load]

        # make dataframe for plotting with seaborn
        dhw_demand_sumline_df = pd.DataFrame({'sum DHW Demand': dhw_demand_sumline_kWh,
                                              'sum Storage Load': storage_load_sumline_kWh,
                                              'DHW Demand': dhw_demand_kW,
                                              'Storage Load': storage_load_kW,
                                              'Losses': losses_W},
                                             index=date_range)

        # decide how to resample data based on plot interval
        start_plot = '2019-08-01'
        end_plot = '2019-08-14'
        timedelta = pd.Timedelta(pd.Timestamp(end_plot) - pd.Timestamp(start_plot))

        if timedelta.days < 3:
            resample_delta = "600S"  # 10min
        elif timedelta.days < 14:  # 2 Weeks
            resample_delta = "1800S"  # 30min
        elif timedelta.days < 62:  # 2 months
            resample_delta = "H"  # hourly
        else:
            resample_delta = "D"

        # make figures with 3 different y-axes
        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1_data = dhw_demand_sumline_df[['sum DHW Demand', 'sum Storage Load']][start_plot:end_plot]
        ax1 = sns.lineplot(data=ax1_data.resample(resample_delta).mean(), dashes=[(6, 2), (6, 2)], linewidth=1.2,
                           palette=[rwth_blue, rwth_orange])
        ax1.grid(False)

        ax2 = ax1.twinx()
        ax2_data = dhw_demand_sumline_df[['DHW Demand', 'Storage Load']][start_plot:end_plot]
        ax2 = sns.lineplot(data=ax2_data.resample(resample_delta).mean(), dashes=False, linewidth=0.7,
                           palette=[rwth_blue, rwth_orange])

        ax3 = ax1.twinx()
        ax3_data = dhw_demand_sumline_df[['Losses']][start_plot:end_plot]
        ax3 = sns.lineplot(data=ax3_data.resample(resample_delta).mean(), dashes=False, linewidth=0.5,
                           palette=[rwth_red])
        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(ymin, ymax * 1.5)
        ax3.spines["right"].set_position(("axes", 1.15))

        # make one legend for the figure
        ax1.legend_.remove()
        ax2.legend_.remove()
        ax3.legend_.remove()
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.9), frameon=False,
                   # prop={'size': 6}
                   )

        ax1.set_ylabel('cumulative Demand and Supply in kWh')
        ax2.set_ylabel('current Demand and Supply in kW')
        ax3.set_ylabel('Losses in W')
        ax2.grid(False)
        ax3.grid(False)

        # set the x axis ticks
        # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f', ]
        formatter.zero_formats = [''] + formatter.formats[:-1]
        formatter.zero_formats[3] = '%d-%b'
        formatter.offset_formats = ['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y', '%d %b %Y %H:%M', ]
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        plt.title('Demand ({} Peaks) and Supply ({} Peaks)'.format(dhw_peaks, stor_peaks))
        plt.show()

        if save_fig:
            fig.savefig(os.path.join(dir_output + "/SummenlineinDiagramm.pdf"))
            fig.savefig(os.path.join(dir_output + "/SummenlineinDiagramm.png"), dpi=600)

    # Output Unit of storage load should be equal to Input Unit of DHW demand
    storage_load = [stor_step / s_step for stor_step in storage_load]

    return storage_load