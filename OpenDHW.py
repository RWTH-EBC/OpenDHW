# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xlrd
import math
import statistics
import random
import scipy
from scipy.stats import beta
import matplotlib.dates as mdates

# RWTH colours
rwth_blue = "#00549F"
rwth_red = "#CC071E"


def compare_generators(first_method, first_series_LperH, second_method,
                       second_series_LperH, s_step, start_plot='2019-03-01',
                       end_plot='2019-03-08', plot_date_slice=True,
                       plot_distribution=True, save_fig=False):
    """
    Compares two methods of computing the water flow time series by means of
    a subplot.
    :param first_method:        string: name of first method (f.e. DHWcalc)
    :param first_series_LperH:  list:   first water flow time series
    :param second_method:       string: name of second method (f.e. OpenDHW)
    :param second_series_LperH: list:   second water flow time series
    :param s_step:              int:    seconds within a timestep
    :param start_plot:          string: date, f.e. 2019-03-01
    :param end_plot:            string: date, f.e. 2019-03-08
    :param save_fig:            bool:   decide to save the plot
    :return:
    """

    # compute Stats for first series
    first_series_LperSec = [x / 3600 for x in first_series_LperH]

    first_series_non_zeros = len([x for x in first_series_LperH if x != 0])

    first_series_yearly_water_demand = round(sum(first_series_LperSec) *
                                             s_step, 1)  # in L
    first_series_av_daily_water = round(first_series_yearly_water_demand /
                                        365, 1)
    first_series_av_daily_water_lst = [first_series_av_daily_water for i in
                                       first_series_LperH]  # L/day
    first_series_max_water_flow = round(max(first_series_LperH), 1)  # in L/h

    # compute Stats for second series
    second_series_LperSec = [x / 3600 for x in second_series_LperH]

    second_series_non_zeros = len([x for x in second_series_LperH if x != 0])

    second_series_yearly_water_demand = round(sum(second_series_LperSec) *
                                              s_step, 1)  # in L
    second_series_av_daily_water = round(second_series_yearly_water_demand /
                                         365, 1)
    second_series_av_daily_water_lst = [second_series_av_daily_water for i in
                                        second_series_LperH]  # L/day
    second_series_max_water_flow = round(max(second_series_LperH), 1)  # in L/h

    # RWTH colours
    rwth_blue = "#00549F"
    rwth_red = "#CC071E"

    # sns.set_style("white")
    sns.set_context("paper")

    if plot_date_slice:

        # set date range to simplify plot slicing
        date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                                   freq=str(s_step) + 'S')
        date_range = date_range[:-1]

        # make dataframe for plotting with seaborn
        first_plot_df = pd.DataFrame({'Waterflow {} [L/h]'.format(first_method):
                                          first_series_LperH,
                                      'Yearly av. Demand [{} L/day]'.format(
                                          first_series_av_daily_water):
                                          first_series_av_daily_water_lst},
                                     index=date_range)

        second_plot_df = pd.DataFrame(
            {'Waterflow {} [L/h]'.format(second_method):
                 second_series_LperH,
             'Yearly av. Demand [{} L/day]'.format(
                 second_series_av_daily_water):
                 second_series_av_daily_water_lst},
            index=date_range)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout()

        ax1 = sns.lineplot(ax=ax1, data=first_plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        ax1.set_title(
            'Water time-series from {}, timestep = {} s\n Yearly Demand ='
            '{} L, Peak = {} L/h, No. Drawoffs = {}'.format(
                first_method, s_step, first_series_yearly_water_demand,
                first_series_max_water_flow, first_series_non_zeros))

        ax1.legend(loc="upper left")

        ax2 = sns.lineplot(ax=ax2, data=second_plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        ax2.set_title(
            'Water time-series from {}, timestep = {} s\n Yearly Water '
            '{} L, Peak = {} L/h, No. Drawoffs = {}'.format(
                second_method, s_step, second_series_yearly_water_demand,
                second_series_max_water_flow, second_series_non_zeros))

        ax2.legend(loc="upper left")

        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_Comparision.pdf")

    if plot_distribution:
        # get non-zero values of the proule
        drawoffs_first_series = [i for i in first_series_LperH if i > 0]
        drawoffs_second_series = [i for i in second_series_LperH if i > 0]

        # compute some stats
        drawoffs_mean_no1 = round(statistics.mean(drawoffs_first_series), 2)
        drawoffs_stdev_no1 = round(statistics.stdev(drawoffs_first_series), 2)

        drawoffs_mean_no2 = round(statistics.mean(drawoffs_second_series), 2)
        drawoffs_stdev_no2 = round(statistics.stdev(drawoffs_second_series), 2)

        # compute Jensen Shannon Distance
        distance = jensen_shannon_distance(q=first_series_LperH,
                                           p=second_series_LperH)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout()

        # plot the distribution
        # https://seaborn.pydata.org/generated/seaborn.displot.html
        ax1 = sns.histplot(ax=ax1, data=drawoffs_first_series, kde=True)
        ax2 = sns.histplot(ax=ax2, data=drawoffs_second_series, kde=True)

        ax1.set_title('Jensen Shannon Distance = {} \n Water time-series from'
                      '{}, timestep = {} s, Yearly Demand = {} L, \n No. '
                      'Drawoffs = {}, Mean = {} L/h, Standard Deviation = {} '
                      'L/h'.format(distance,
                                   first_method, s_step,
                                   first_series_yearly_water_demand,
                                   first_series_non_zeros, drawoffs_mean_no1,
                                   drawoffs_stdev_no1))

        ax1.set_ylabel('Count in a Year')

        ax2.set_title('Water time-series from {}, timestep = {} s Yearly '
                      'Demand = {} L, \n No. Drawoffs = {}, Mean = {} L/h, '
                      'Standard Deviation = {} L/h'.format(
            second_method, s_step, second_series_yearly_water_demand,
            second_series_non_zeros, drawoffs_mean_no2, drawoffs_stdev_no2))

        ax2.set_ylabel('Count in a Year')
        ax2.set_xlabel('Flowrate [L/h]')

        plt.show()

    return


def import_from_dhwcalc(s_step, categories, daily_demand):
    """
    DHWcalc yields Volume Flow TimeSeries (in Liters per hour).

    :param  s_step:     int:    resolution of output file in seconds
    :param  categories  int:    either '1' or '4', see DHWcalc settings

    :return dhw_demand: list:   each timestep contains the Energyflow in [W]
    """

    dhw_file = "{}L_{}min_{}cat_step_functions.txt".format(daily_demand,
                                                           int(s_step / 60),
                                                           categories)

    dhw_profile = Path.cwd().parent / "DHWcalc_Files" / dhw_file

    assert dhw_profile.exists(), 'No DHWcalc File for the selected parameters.'

    # Flowrate in Liter per Hour in each Step
    water_LperH = [int(word.strip('\n')) for word in
                   open(dhw_profile).readlines()]  # L/h each step

    return water_LperH


def draw_histplot(dhw_profile_LperH, s_step):
    """
    Takes a DHW profile and plots a histogram with some stats in the title

    :param dhw_profile_LperH:   list:   the profile
    :param s_step:              int:    seconds within a timestep
    :return:
    """
    water_LperSec = [x / 3600 for x in dhw_profile_LperH]
    yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L

    # get non-zero values of the proule
    drawoffs = [i for i in dhw_profile_LperH if i > 0]

    # compute some stats
    drawoffs_mean = round(statistics.mean(drawoffs), 2)
    drawoffs_stdev = round(statistics.stdev(drawoffs), 2)

    # plot the distribution
    # https://seaborn.pydata.org/generated/seaborn.displot.html
    ax = sns.histplot(drawoffs, kde=True, palette=[rwth_blue, rwth_red])

    # ax2 = ax.twinx()
    # sns.kdeplot(ax=ax2, data=drawoffs, alpha=.25, bw_adjust=0.05)

    ax.set_title('Timestep = {} s, Yearly Demand = {} L, \n No. Drawoffs = {}, '
                 'Mean = {} L/h, Standard Deviation = {} L/h'.format(
        s_step, yearly_water_demand, len(drawoffs), drawoffs_mean,
        drawoffs_stdev), fontdict={'fontsize': 10})

    plt.show()


def shift_weekend_weekday(p_weekday, p_weekend, factor=1.2):
    """
    Shifts the probabilities between the weekday list and the weekend list by a
    defined factor. If the factor is bigger than 1, the probability on the
    weekend is increased. If its smaller than 1, the probability on the
    weekend is decreased.

    :param p_weekday:   list:   probabilites for 1 day of the week [0...1]
    :param p_weekend:   list:   probabilitiers for 1 day of the weekend [0...1]
    :param factor:      float:  factor to shift the probabiliters between
                                weekdays and weekenddays
    :return:
    """

    av_p_wd = statistics.mean(p_weekday)
    av_p_we = statistics.mean(p_weekend)

    av_p_week = av_p_wd * 5 / 7 + av_p_we * 2 / 7

    p_wd_factor = 1 / (5 / 7 + factor * 2 / 7)
    p_we_factor = 1 / (1 / factor * 5 / 7 + 2 / 7)

    assert p_wd_factor * 5 / 7 + p_we_factor * 2 / 7 == 1

    p_wd_weighted = [p * p_we_factor for p in p_weekday]
    p_we_weighted = [p * p_we_factor for p in p_weekend]

    av_p_wd_weighted = statistics.mean(p_wd_weighted)
    av_p_we_weighted = statistics.mean(p_we_weighted)

    av_p_week_weighted = av_p_wd_weighted * 5 / 7 + av_p_we_weighted * 2 / 7

    return p_wd_weighted, p_we_weighted, av_p_week_weighted


def generate_average_daily_profile(mode, l_day, sigma_day, av_p_day,
                                   s_step, plot_profile=False):
    """
    Generates an average profile for daily water drawoffs. The total amount
    of water in the average profile has to be higher than the demanded water
    per day, as the average profile is multiplied by the average probability
    each day. two modes are given to generate the average profile.

    :param mode:            string: type of probability distribution
    :param l_day:           float:  mean value of resulting profile
    :param sigma_day:       float:  standard deviation of resulting profile
    :param av_p_day:        float:  average probability of
    :param s_step:          int:    seconds within a time step
    :param plot_profile:    bool:   decide to plot the profile

    :return: average_profile:   list:   average water drawoff profile in L/h
                                        per timestep
    """

    timesteps_day = int(24 * 3600 / s_step)

    l_av_profile = l_day / av_p_day
    sigma_av_profile = sigma_day / av_p_day

    LperH_step_av_profile = l_av_profile / 24
    sigma_step_av_profile = sigma_av_profile / 24

    if mode == 'gauss':

        # problem: generates negative values.

        average_profile = [random.gauss(LperH_step_av_profile,
                                        sigma=sigma_step_av_profile) for i in
                           range(timesteps_day)]

        if min(average_profile) < 0:
            raise Exception("negative values in average profiles detected. "
                            "Choose a different mean or standard deviation, "
                            "or choose a differnt mode to create the average "
                            "profile.")

    elif mode == 'gauss_abs':

        # If we take the absolute of the gauss distribution, we have no more
        # negative values, but the mean and standard deviation changes,
        # and more than 200 L/d are being consumed.

        average_profile = [random.gauss(LperH_step_av_profile,
                                        sigma=sigma_step_av_profile) for i in
                           range(timesteps_day)]

        average_profile_abs = [abs(entry) for entry in average_profile]

        if statistics.mean(average_profile) != statistics.mean(
                average_profile_abs):
            scale = statistics.mean(average_profile) / statistics.mean(
                average_profile_abs)

            average_profile = [i * scale for i in average_profile_abs]

    elif mode == 'lognormal':

        # problem: understand the settings of the lognormal function.
        # https://en.wikipedia.org/wiki/Log-normal_distribution

        m = LperH_step_av_profile
        sigma = sigma_step_av_profile / 40

        v = sigma ** 2
        norm_mu = np.log(m ** 2 / np.sqrt(v + m ** 2))
        norm_sigma = np.sqrt((v / m ** 2) + 1)

        average_profile = np.random.lognormal(norm_mu, norm_sigma,
                                              timesteps_day)

    else:
        raise Exception("Unkown Mode for average daily water profile "
                        "geneartion")

    if plot_profile:
        mean = [statistics.mean(average_profile) for i in average_profile]
        plt.plot(average_profile)
        plt.plot(mean)
        plt.show()

    return average_profile


def generate_dhw_profile_average_profile(s_step, weekend_weekday_factor=1.2,
                                         initial_day=0):

    p_we = generate_daily_probability_step_function(
        mode='weekend',
        s_step=s_step
    )

    p_wd = generate_daily_probability_step_function(
        mode='weekday',
        s_step=s_step
    )

    p_wd_weighted, p_we_weighted, av_p_week_weighted = shift_weekend_weekday(
        p_weekday=p_wd,
        p_weekend=p_we,
        factor=weekend_weekday_factor
    )

    average_profile = generate_average_daily_profile(
        mode='gauss_abs',
        l_day=200,
        sigma_day=70,
        av_p_day=av_p_week_weighted,
        s_step=s_step,
    )

    p_final = generate_yearly_probabilities(
        initial_day=initial_day,
        p_weekend=p_we_weighted,
        p_weekday=p_wd_weighted,
        s_step=s_step
    )

    p_final = normalize_list_to_max(lst=p_final)

    water_LperH = distribute_average_profile(
        average_profile=average_profile,
        p_final=p_final,
        s_step=s_step
    )

    return water_LperH


def generate_dhw_profile(s_step, weekend_weekday_factor=1.2,
                         drawoff_method='gauss_combined',
                         mean_vol_per_drawoff=8, mean_drawoff_vol_per_day=200,
                         initial_day=0):
    """
    Generates a DHW profile. The generation is split up in different
    functions and generally follows the methodology described in the DHWcalc
    paper from Uni Kassel.

    1)  Probabilities for weekdays and weekend-days are loaded (p_day).
    2)  Probability of weekend-days is increased relative to weekdays (shift).
    3)  Based on an initial day, the yearly probability distribution (p_final)
        is generated. The seasonal influence is modelled by a sine-function.
    4)  p_final is normalized and integrated. The sum over the year is thus
        equal to 1 (p_norm_integral).
    5)  Drawoffs are generated based on a Beta distribution. Additionally,
        a list of random values bwtween 0 and 1 is generated.
    6)  The drawoffs are distributed based on the random values into the
        p_norm_integral space. The final yearly drawoff profile is returned.
    7)  Optionally, the profile can be plotted and the associated heat
        computed.

    :param s_step:
    :param weekend_weekday_factor:
    :param drawoff_method:
    :param mean_vol_per_drawoff:
    :param mean_drawoff_vol_per_day:
    :param initial_day:
    :return:
    """

    p_we = generate_daily_probability_step_function(
        mode='weekend',
        s_step=s_step
    )

    p_wd = generate_daily_probability_step_function(
        mode='weekday',
        s_step=s_step
    )

    p_wd_weighted, p_we_weighted, av_p_week_weighted = shift_weekend_weekday(
        p_weekday=p_wd,
        p_weekend=p_we,
        factor=weekend_weekday_factor
    )

    p_final = generate_yearly_probabilities(
        initial_day=initial_day,
        p_weekend=p_we_weighted,
        p_weekday=p_wd_weighted,
        s_step=s_step
    )

    p_norm_integral = normalize_and_sum_list(lst=p_final)

    drawoffs, p_drawoffs = generate_drawoffs(
        method=drawoff_method,
        mean_vol_per_drawoff=mean_vol_per_drawoff,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        s_step=s_step,
        p_norm_integral=p_norm_integral
    )

    water_LperH = distribute_drawoffs(
        drawoffs=drawoffs,
        p_drawoffs=p_drawoffs,
        p_norm_integral=p_norm_integral,
        s_step=s_step
    )

    return water_LperH


def distribute_average_profile(average_profile, s_step, p_final):
    """
    distribute the average profile.

    :param average_profile:
    :param s_step:
    :param p_final:
    :return:
    """

    average_profile = average_profile * 365

    timesteps_day = int(24 * 3600 / s_step)

    water_LperH = []

    for step in range(365 * timesteps_day):

        if random.random() < p_final[step]:

            water_t = random.gauss(average_profile[step], sigma=114.33)
            water_LperH.append(abs(water_t))
        else:
            water_LperH.append(0)

    return water_LperH


def generate_yearly_probabilities(initial_day, p_weekend, p_weekday, s_step):
    """
    Takes the probabilities of a weekday and a weekendday and generates a
    list of yearly probabilities by adding a seasonal probability factor.
    The seasonal factor is a sine-function, like in DHWcalc.

    :param initial_day: int:    0: Mon, 1: Tue, 2: Wed, 3: Thur, 4: Fri,
                                5 : Sat, 6 : Sun
    :param p_weekend:   list:   probabilities of a weekend day
    :param p_weekday:   list:   probabilities of a weekday
    :param s_step:      int:    seconds within a timestep

    :return: p_final:   list:   probabilities of a full year
    """

    p_final = []
    timesteps_day = int(24 * 3600 / s_step)

    for day in range(365):

        # Is the current day on a weekend?
        if (day + initial_day) % 7 >= 5:
            p_day = p_weekend
        else:
            p_day = p_weekday

        # Compute seasonal factor
        arg = math.pi * (2 / 365 * day - 1 / 4)
        probability_season = 1 + 0.1 * np.cos(arg)

        for step in range(timesteps_day):
            probability = p_day[step] * probability_season
            p_final.append(probability)

    return p_final


def distribute_drawoffs(drawoffs, p_drawoffs, p_norm_integral, s_step):
    """
    Takes a small list (p_drawoffs) and sorts it into a bigger list (
    p_norm_integral). Both lists are being sorted. Then, the big list is
    iterated over, and whenever a value of the small list is smaller than a
    value of the big list, the index of the big list is saved and a drawoff
    event from the drawoffs list occurs.

    :param drawoffs:        list:   drawoff events in L/h
    :param p_drawoffs:      list:   drawoff event probabilities [0...1]
    :param p_norm_integral: list:   normalized sum of yearly water use
                                    probabilities [0...1]
    :param s_step:          int:    seconds within a timestep

    :return: water_LperH:   list:   resutling water drawoff profile
    """

    p_drawoffs.sort()
    p_norm_integral.sort()

    drawoff_count = 0

    # for return statement
    water_LperH = [0] * int(365 * 24 * 3600 / s_step)

    for step, p_current_sum in enumerate(p_norm_integral):

        if p_drawoffs[drawoff_count] < p_current_sum:
            water_LperH[step] = drawoffs[drawoff_count]
            drawoff_count += 1

            if drawoff_count >= len(drawoffs):
                break

    return water_LperH


def generate_drawoffs(s_step, p_norm_integral, mean_vol_per_drawoff=8,
                      mean_drawoff_vol_per_day=200, method='gauss_combined'):
    """
    Generates two lists. First, the "drawoffs" list, with the darwoff events as
    flowrate entries in Liter/hour.  Second, the "p_drawoffs" list, which has
    the same length as the "drawoffs" lists but contains random values,
    between the minimum and the maximum of "p_norm_integral". These are
    usually values between 0 and 1, following the convention of DHWcalc.

    The drawoffs are generated based on some key parameters, like the mean
    water volume consumed per drawoff and the mean water volume consumed per
    day.

    Then, the drawoffs are generated following either a Gauss Distribution (
    as describesd in the DHWcalc paper) or a beta distribution. The
    advantage of the Beta Distribution is the ability to directly set values
    for the minimum and the maximum value. More information on the Beta
    Distribution can be found at https://en.wikipedia.org/wiki/Beta_distribution

    :param s_step:                      int     seconds within a timestep
    :param p_norm_integral:             list    min and max values taken
    :param mean_vol_per_drawoff:        int     mean volume per drawpff
    :param mean_drawoff_vol_per_day:    int     mean volume drawn off per day
    :param method:                      string  "gauss" or "beta"
    :return:    drawoffs:               list    drawoff events in [L/h]
                p_drawoffs:             list    probabilities 0...1
    """
    # dhw calc has more settings here, see Fig 5 in paper "Draw off features".

    # Todo: "A flow rate step size of 6 l/h is defined by the program."
    # Todo: checken warum total drawoffs in DHWcalc anders ist

    av_drawoff_flow_rate = mean_vol_per_drawoff * 3600 / s_step  # in L/h

    sdt_dev_drawoff_flow_rate = 120  # in L/h

    mean_no_drawoffs_per_day = mean_drawoff_vol_per_day / mean_vol_per_drawoff

    total_drawoffs = int(mean_no_drawoffs_per_day * 365)

    max_drawoff_flow_rate = 1200  # in L/h
    min_drawoff_flow_rate = 6  # in L/h

    if method == 'beta':
        # https://en.wikipedia.org/wiki/Beta_distribution
        # https://stats.stackexchange.com/questions/317729/is-the-gaussian-distribution-a-specific-case-of-the-beta-distribution
        # https://stackoverflow.com/a/62364837
        # https://www.vosesoftware.com/riskwiki/NormalapproximationtotheBetadistribution.php
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

        # scale the mean down to a value between 0 and 1 based on:
        # https://en.wikipedia.org/wiki/Beta_distribution#Four_parameters
        mu_dist = (av_drawoff_flow_rate - min_drawoff_flow_rate) / (
                max_drawoff_flow_rate - min_drawoff_flow_rate)

        # scale down the sdt_dev by the same factor
        sig_dist = sdt_dev_drawoff_flow_rate * mu_dist / av_drawoff_flow_rate

        # parameterize a and b shape parameters based on mean and std_dev:
        # https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        v = (mu_dist * (1 - mu_dist) / sig_dist ** 2) - 1

        a = mu_dist * v
        b = (1 - mu_dist) * v

        dist = beta(a, b)

        mean, var = beta.stats(a, b, moments='mv')

        drawoffs = min_drawoff_flow_rate + dist.rvs(size=total_drawoffs) * (
                max_drawoff_flow_rate - min_drawoff_flow_rate)

        drawoffs_mean = statistics.mean(drawoffs)
        drawoffs_stdev = statistics.stdev(drawoffs)
        drawoffs_min = round(min(drawoffs), 2)
        drawoffs_max = round(max(drawoffs), 2)

        error_mean = abs(av_drawoff_flow_rate - drawoffs_mean) / max(
            av_drawoff_flow_rate, drawoffs_mean)

        error_stdev = abs(sdt_dev_drawoff_flow_rate - drawoffs_stdev) / max(
            sdt_dev_drawoff_flow_rate, drawoffs_stdev)

        error_max = (max_drawoff_flow_rate - drawoffs_max) / \
                    max_drawoff_flow_rate
        error_max = round(error_max, 3)

        error_min = (drawoffs_min - min_drawoff_flow_rate) / \
                    min_drawoff_flow_rate
        error_min = round(error_min, 3)

        if error_mean > 0.01:
            raise Exception("The beta distribution changes the Mean Value of "
                            "the drawoffs by more than 1%. Please Re-Run the "
                            "script or change the accuracy requirements for "
                            "the given sample size.")

        if error_stdev > 0.02:
            raise Exception("The beta distribution changes the Standard "
                            "Deviation of the drawoffs by more than 2%. "
                            "Please Re-Run the script or change the accuracy "
                            "requirements for the given sample size.")

        print("Max-Value Drawoffs = {} L/h, off by {} % from the Set "
              "Max-Value =  {} L/h".format(drawoffs_max, error_max * 100,
                                           max_drawoff_flow_rate))

        print("Min-Value Drawoffs = {} L/h, off by {} % from the Set "
              "Min-Value = {} L/h".format(drawoffs_min, error_min * 100,
                                          min_drawoff_flow_rate))

    elif method == 'gauss_simple':
        # outdated, not reccomended to use

        mu = av_drawoff_flow_rate  # mean
        sig = av_drawoff_flow_rate / 4  # standard deviation
        drawoffs = []  # in [L/h]
        mu_initial = 0

        # drawoff flow rate has to be positive. try 4 times
        for try_i in range(4):

            drawoffs = [random.gauss(mu, sig) for i in range(total_drawoffs)]
            mu_initial = statistics.mean(drawoffs)

            if min(drawoffs) >= 0:
                break

        # if still negative values after 4 tries, make 0's from negatives
        if min(drawoffs) <= 0:
            drawoffs_new = []
            for event in drawoffs:
                if event >= 0:
                    drawoffs_new.append(event)
                if event < 0:
                    drawoffs_new.append(0)
            drawoffs = drawoffs_new

            mu_zeros = statistics.mean(drawoffs_new)

            if mu_zeros / mu_initial > 1.01:
                raise Exception("changing the negative values in the drawoffs "
                                "list to zeros changes the Mean Value by more "
                                "than 1%. Please choose a different standard "
                                "deviation.")

    elif method == 'gauss_combined':
        # as close as it gets to the DHWcalc Algorithm

        mu = av_drawoff_flow_rate  # in L/h
        sig = sdt_dev_drawoff_flow_rate  # in L/h

        drawoffs = [random.gauss(mu, sig) for i in range(total_drawoffs)]

        low_lim = mu - 2 * sig
        up_lim = mu + 2 * sig

        # cut gauss distribution
        drawoffs = [i for i in drawoffs if low_lim < i < up_lim]

        curr_no_drawoffs = len(drawoffs)
        no_drawoffs_left = total_drawoffs - curr_no_drawoffs

        noise = [random.randint(low_lim, max_drawoff_flow_rate) for i in
                 range(no_drawoffs_left)]

        drawoffs.extend(noise)

        # DHWcalc has a set flow rate step rather than a continuous
        # distribution. Thus, we round the drawoff distribution according to
        # this step width.
        flow_rate_step = 6  # L/h
        drawoffs = [flow_rate_step * round(i / flow_rate_step) for i in
                    drawoffs]

        # sns.displot(drawoffs, kde=True)
        # plt.show()

    else:
        raise Exception("Unkown method to generate drawoffs. choose Gauss or "
                        "Beta Distribution.")

    min_rand = min(p_norm_integral)
    max_rand = max(p_norm_integral)

    p_drawoffs = [random.uniform(min_rand, max_rand) for i in drawoffs]

    return drawoffs, p_drawoffs


def normalize_and_sum_list(lst):
    """
    takes a list and normalizes it based on the sum of all list elements.
    then generates a new list based on the current sum of each list entry.

    :param lst:                 list:   input list
    :return: lst_norm_integral: list    output list
    """

    sum_lst = sum(lst)
    lst_norm = [float(i) / sum_lst for i in lst]

    current_sum = 0
    lst_norm_integral = []

    for entry in lst_norm:
        current_sum += entry
        lst_norm_integral.append(current_sum)

    return lst_norm_integral


def normalize_list_to_max(lst):
    """
    takes a list and normalizes it based on the max of all list elements.

    :param lst:                 list:   input list
    :return: lst_norm_integral: list    output list
    """

    max_lst = max(lst)
    lst_norm = [float(element) / max_lst for element in lst]

    return lst_norm


def generate_daily_probability_step_function(mode, s_step, plot_p_day=False):
    """
    Generates probabilities for a day with 6 periods. Corresponds to the mode
    "step function for weekdays and weekends" in DHWcalc and uses the same
    standard values. Each Day starts at 0:00. Steps in hours. Sum of steps
    has to be 24. Sum of probabilites has to be 1.

    :param mode:        string: decide to compute for a weekday of a weekend day
    :param s_step:      int:    seconds within a timestep
    :param plot_p_day:  Bool:   decide to plot the probability distribution
    :return: p_day      list:   the probability distribution for one day.
    """

    if mode == 'weekday':

        steps_and_ps = [(6.5, 0.01), (1, 0.5), (4.5, 0.06), (1, 0.16),
                        (5, 0.06), (4, 0.2), (2, 0.01)]

    elif mode == 'weekend':

        steps_and_ps = [(7, 0.02), (2, 0.475), (6, 0.071), (2, 0.237),
                        (3, 0.036), (3, 0.143), (1, 0.018)]

    else:
        raise Exception('Unkown Mode. Please Choose "Weekday" or "Weekend".')

    steps = [tup[0] for tup in steps_and_ps]
    ps = [tup[1] for tup in steps_and_ps]

    assert sum(steps) == 24
    assert sum(ps) == 1

    p_day = []

    for tup in steps_and_ps:
        p_lst = [tup[1] for i in range(int(tup[0] * 3600 / s_step))]
        p_day.extend(p_lst)

    # check if length of daily intervals fits into the stepwidth. if s_step
    # f.e is 3600s (1h), one daily intervall cant be 4.5 hours.
    assert len(p_day) == 24 * 3600 / s_step

    if plot_p_day:
        plt.plot(p_day)
        plt.show()

    return p_day


def plot_average_profiles_pycity(save_fig=False):
    profiles_path = Path.cwd() / 'dhw_stochastical.xlsx'
    profiles = {"we": {}, "wd": {}}
    book = xlrd.open_workbook(profiles_path)

    s_step = 600

    # Iterate over all sheets. wd = weekday, we = weekend. mw = ist the
    # average profile in [L/h] in 10min steps. occupancy is between 1-6 (we1 -
    # we6).
    for sheetname in book.sheet_names():
        sheet = book.sheet_by_name(sheetname)

        # Read values
        values = [sheet.cell_value(i, 0) for i in range(1440)]

        # Store values in dictionary
        if sheetname in ("wd_mw", "we_mw"):
            profiles[sheetname] = values  # minute-wise average profile L/h

    water_LperH_we = profiles["we_mw"]
    water_LperH_wd = profiles["wd_mw"]

    water_L_we = [i * s_step / 3600 for i in water_LperH_we]
    water_L_wd = [i * s_step / 3600 for i in water_LperH_wd]

    daily_water_we = round(sum(water_L_we), 1)
    daily_water_wd = round(sum(water_L_wd), 1)

    av_wd_lst = [statistics.mean(water_LperH_we) for i in range(1440)]
    av_we_lst = [statistics.mean(water_LperH_wd) for i in range(1440)]

    fig, ax = plt.subplots()
    ax.plot(water_LperH_we, linewidth=0.7, label="Weekend")
    ax.plot(water_LperH_wd, linewidth=0.7, label="Weekday")
    ax.plot(av_wd_lst, linewidth=0.7, label="Average Weekday")
    ax.plot(av_we_lst, linewidth=0.7, label="Average Weekday")
    plt.ylabel('Water [L/h]')
    plt.xlabel('Minutes in a day')
    plt.title('Average profiles from PyCity. \n'
              'Daily Sum Weekday: {} L, Daily Sum Weekend: {} L'.format(
        daily_water_wd, daily_water_we))

    plt.legend(loc='upper left')
    plt.show()

    if save_fig:
        dir_output = Path.cwd() / "plots"
        dir_output.mkdir(exist_ok=True)
        fig.savefig(dir_output / "Average_Profiles_PyCity.pdf")


def compute_heat(s_step, water_LperH, temp_dT=35, print_stats=True):
    """
    Takes a timeseries of waterflows per timestep in [L/h]. Computes a
    DHW Demand series in [kWh]. Computes additional stats an optionally
    prints them out. Optionally plots the timesieries with additional stats.

    :param s_step:      int:    seconds within a timestep. F.e. 60, 600, 3600
    :param water_LperH: list:   list that holds the waterflow values for each
                                timestep in Liters per Hour.
    :param temp_dT:     int:    temperature difference between freshwater and
                                average DHW outlet temperature. F.e. 35

    :return:    dhw:    list:   list of the heat demand for DHW for each
                                timestep in kWh.
    """

    water_LperSec = [x / 3600 for x in water_LperH]  # L/s each step

    rho = 980 / 1000  # kg/L for Water (at 60°C? at 10°C its = 1)
    cp = 4180  # J/kgK
    dhw = [i * rho * cp * temp_dT for i in water_LperSec]  # in W

    if print_stats:
        # compute Sums and Maxima for Water and Heat
        yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
        max_water_flow = round(max(water_LperH), 1)  # in L/h

        yearly_dhw_demand = round(sum(dhw) * s_step / (3600 * 1000), 1)  # kWh
        max_dhw_heat_flow = round(max(dhw) / 1000, 1)  # in kW

        print("Yearly drinking water demand from DHWcalc Alias is {:.2f} L"
              " with a maximum of {:.2f} L/h".format(yearly_water_demand,
                                                     max_water_flow))

        print("Yearly DHW energy demand from DHWcalc Alias is {:.2f} kWh"
              " with a maximum of {:.2f} kW".format(yearly_dhw_demand,
                                                    max_dhw_heat_flow))

    return dhw  # in kWh


def draw_lineplot(method, s_step, series, series_type='water',
                  start_plot='2019-02-01', end_plot='2019-02-05',
                  save_fig=False):
    """
    Takes a timeseries of waterflows per timestep in [L/h]. Computes a
    DHW Demand series in [kWh]. Computes additional stats an optionally
    prints them out. Optionally plots the timesieries with additional stats.

    :param method:      str:    Name of the DHW Method, f.e. DHWcalc, PyCity.
                                Just for naming the plot.
    :param s_step:      int:    seconds within a timestep. F.e. 60, 600, 3600
    :param series:      list:   list that holds the timeseries.
    :param start_plot:  str:    start date of the plot. F.e. 2019-01-01
    :param end_plot:    str:    end date of the plot. F.e. 2019-02-01
    :param save_fig:    bool:   decide to save plots as pdf

    :return:    fig:    fig:    figure of the plot
                dhw:    list:   list of the heat demand for DHW for each
                                timestep in kWh.
    """

    # RWTH colours
    rwth_blue = "#00549F"
    rwth_red = "#CC071E"

    # sns.set_style("white")
    sns.set_context("paper")

    # set date range to simplify plot slicing
    date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                               freq=str(s_step) + 'S')
    date_range = date_range[:-1]

    fig, ax1 = plt.subplots()
    fig.tight_layout()

    if series_type == 'water':
        water_LperH = series
        water_LperSec = [x / 3600 for x in water_LperH]  # L/s each step

        # compute Sums and Maxima for Water and Heat
        yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
        av_daily_water = round(yearly_water_demand / 365, 1)
        av_daily_water_lst = [av_daily_water for i in water_LperH]  # L/day
        max_water_flow = round(max(water_LperH), 1)  # in L/h

        # make dataframe for plotting with seaborn
        plot_df = pd.DataFrame({'Waterflow [L/h]': water_LperH,
                                'Yearly av. Demand [{} L/day]'.format(
                                    av_daily_water): av_daily_water_lst},
                               index=date_range)

        ax1 = sns.lineplot(ax=ax1, data=plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        ax1.legend(loc="upper left")

        plt.title('Water Time-series from {}, timestep = {} s\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h'.format(
            method, s_step, yearly_water_demand, max_water_flow))

    if series_type == 'heat':
        heat = series  # in Watt

        # make dataframe for plotting with seaborn
        plot_df = pd.DataFrame({'Heat [W]': heat},
                               index=date_range)

        ax1 = sns.lineplot(ax=ax1, data=plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_red])

        ax1.legend(loc="upper left")

        plt.title('Heat Time-series from {}, timestep = {} s\n'
                  'with a Peak of {} L/h'.format(method, s_step, max(heat)))

    # set the x axis ticks
    # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f', ]
    formatter.zero_formats = [''] + formatter.formats[:-1]
    formatter.zero_formats[3] = '%d-%b'
    formatter.offset_formats = ['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y',
                                '%d %b %Y %H:%M', ]
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    plt.show()

    if save_fig:
        dir_output = Path.cwd() / "plots"
        dir_output.mkdir(exist_ok=True)
        fig.savefig(dir_output / "Demand_{}_sliced.pdf".format(method))

    return fig


def generate_multiple_profiles(s_step, runs=5):
    """
    Calls the 'generate_dhw_profile' function multiple times and saves the
    resulting Time-Series in a pandas dataframe. Also saves all drawoff events (
    non-zero entries of the dhw_profiles) in another Dataframe. Both
    Dataframes are then returned.

    :param s_step:  int:    seconds within a timestep
    :param runs:    int:    number of dhw profile that are generated
    :return: dhw_demands_df, drawoffs_df
    """
    date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                               freq=str(s_step) + 'S')
    date_range = date_range[:-1]

    # make dataframes for plotting with seaborn
    dhw_demands_df = pd.DataFrame(columns=range(runs), index=date_range)
    drawoffs_df = pd.DataFrame(columns=range(runs))

    for run in range(runs):
        profile = generate_dhw_profile(s_step=s_step,
                                       weekend_weekday_factor=1.2,
                                       drawoff_method='gauss_combined',
                                       mean_vol_per_drawoff=8,
                                       mean_drawoff_vol_per_day=200,
                                       initial_day=0)

        dhw_demands_df[run] = profile

        # drawoffs events are values in the profile larger than 0
        drawoffs = [event for event in profile if event > 0]
        drawoffs_df[run] = drawoffs

    dhw_demands_df['Average'] = dhw_demands_df.mean(axis=1)  # average profile

    return dhw_demands_df, drawoffs_df


def plot_multiple_runs(dhw_demands_df, drawoffs_df, plot_demands_overlay=False,
                       start_plot='2019-02-01', end_plot='2019-02-02',
                       plot_hist=True, plot_kde=True):
    """
    Plots the dataframes generated by the 'generate_multiple_profiles'
    function. The Dataframes hold multiple dhw demand profiles.

    :param dhw_demands_df:
    :param drawoffs_df:
    :param plot_demands_overlay:
    :param start_plot:
    :param end_plot:
    :param plot_hist:
    :param plot_kde:
    :return:
    """

    if plot_demands_overlay:
        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1 = sns.lineplot(ax=ax1, data=dhw_demands_df[start_plot:end_plot],
                           linewidth=0.5, legend=False)

        # set beautiful x axis ticks for datetime
        # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        plt.show()

    if plot_hist:
        ax = sns.histplot(data=drawoffs_df, kde=False, element="step",
                          fill=False, stat='count', line_kws={'alpha': 0.8,
                                                              'linewidth': 0.9})
        plt.show()

    if plot_kde:
        ax = sns.kdeplot(data=drawoffs_df, bw_adjust=0.1, alpha=0.5,
                         fill=False, linewidth=0.5, legend=False)
        plt.show()


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance between two probability
    distributions. 0 indicates that the two distributions are the same,
    and 1 would indicate that they are nowhere similar.

    From https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return round(distance, 4)
