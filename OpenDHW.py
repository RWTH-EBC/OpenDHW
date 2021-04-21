# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
import statistics
import random
import scipy
from scipy.stats import beta
import matplotlib.dates as mdates

"""
This is the script that stores all function of the DHWcalc package.
It is not meant to be executed on its own, but rather a toolbox for building
small scripts. Examples are given in OpenDHW/Examples.

OpenDHW is mostly built on Pandas, a good introduction is given here:
https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

OpenDHW_Utilities stores a few other functions that do not generate DHW 
Timeseries directly, like the StorageLoad Function.
"""

# RWTH colours
rwth_blue = "#00549F"
rwth_red = "#CC071E"
# sns.set_style("white")
sns.set_context("paper")

# --- Constants ---
rho = 980 / 1000  # kg/L for Water (at 60°C? at 10°C its = 1)
cp = 4180  # J/kgK


def import_from_dhwcalc(s_step, daylight_saving, categories,
                        mean_drawoff_vol_per_day=200, max_flowrate=1200):
    """
    DHWcalc yields Volume Flow TimeSeries (in Liters per hour).

    :param  s_step:         int:    resolution of output file in seconds
    :param  categories:     int:    either '1' or '4', see DHWcalc settings
    :param  mean_drawoff_vol_per_day:   int:    daily water demand in Liters
    :param  daylight_saving:    Bool:   decide to apply daylight saving or not
    :param  max_flowrate:   int:    maximum water flowrate in L/h

    :return dhw_demand: list:   each timestep contains the Energyflow in [W]
    """

    if daylight_saving:
        ds_string = 'ds'
    else:
        ds_string = 'nods'

    dhw_file = "{vol}L_{s_step}min_{cats}cat_sf_{ds}_max{max_flow}.txt".format(
        vol=mean_drawoff_vol_per_day,
        s_step=int(s_step / 60),
        cats=categories,
        ds=ds_string,
        max_flow=max_flowrate,
    )

    dhw_profile = Path.cwd().parent / "DHWcalc_Files" / dhw_file

    assert dhw_profile.exists(), 'No DHWcalc File for the selected parameters.'

    # Flowrate in Liter per Hour in each Step
    water_LperH = [int(word.strip('\n')) for word in
                   open(dhw_profile).readlines()]  # L/h each step

    date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                               freq=str(s_step) + 'S')
    date_range = date_range[:-1]

    # make dataframe
    timeseries_df = pd.DataFrame(water_LperH, index=date_range, columns=[
        'Water_LperH'])

    timeseries_df['Water_L'] = timeseries_df['Water_LperH'] / 3600 * s_step
    timeseries_df['method'] = 'DHWcalc'
    timeseries_df['mean_drawoff_vol_per_day'] = mean_drawoff_vol_per_day
    timeseries_df['sdtdev_drawoff_vol_per_day'] = mean_drawoff_vol_per_day / 4
    timeseries_df['categories'] = categories
    timeseries_df['initial_day'] = 0
    timeseries_df['weekend_weekday_factor'] = 1.2

    mean_vol_per_drawoff = 8  # constant DHWcalc 1 category
    timeseries_df['mean_vol_per_drawoff'] = mean_vol_per_drawoff

    av_drawoff_flow_rate = mean_vol_per_drawoff * 3600 / s_step  # in L/h
    timeseries_df['mean_drawoff_flow_rate_LperH'] = av_drawoff_flow_rate

    sdt_dev_drawoff_flow_rate = av_drawoff_flow_rate / 4  # in L/h
    timeseries_df['sdtdev_drawoff_flow_rate_LperH'] = sdt_dev_drawoff_flow_rate

    mean_no_drawoffs_per_day = mean_drawoff_vol_per_day / mean_vol_per_drawoff
    timeseries_df['mean_no_drawoffs_per_day'] = mean_no_drawoffs_per_day

    return timeseries_df


def generate_dhw_profile(s_step, categories, weekend_weekday_factor=1.2,
                         mean_drawoff_vol_per_day=200, initial_day=0):
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

    :param s_step:                      int:    timestep width in seconds. f.e.
                                                60. Seems like the Distribution
                                                is more similar to DHWcalc if
                                                always a stepwidth of 60S is
                                                chosen and later resampled with
                                                'resample_water_series'.
    :param categories:                  int:    1 or 4 (see DHWcalc)
    :param weekend_weekday_factor:      int:    taken from DHWcalc
    :param mean_drawoff_vol_per_day:    int:    function of number of people in
                                                the house of floor area.
    :param initial_day:                 int:    0:Mon - 1:Tues ... 6:Sun
    :return: timeseries_df              df:     dataframe with all timeseries
    """

    if categories == 4:

        cats_df = get_data_drawoff_categories(
            s_step=s_step, mean_drawoff_vol_per_day=mean_drawoff_vol_per_day)

        # deterministic
        timeseries_df = generate_yearly_probability_profile(
            s_step=s_step,
            weekend_weekday_factor=1.2,
            initial_day=0
        )

        for i in range(len(cats_df)):
            timeseries_df = generate_and_distribute_drawoffs_cats(
                timeseries_df=timeseries_df,
                cats_series=cats_df.iloc[i],
            )

        col_names = list(timeseries_df.columns)
        cols_LperH = [name for name in col_names if 'Water_LperH' in name]
        water_LperH_df = timeseries_df[cols_LperH]
        timeseries_df['Water_LperH'] = water_LperH_df.sum(axis=1)

        # todo: find better way of superpositioning subject to max
        global_max_flow_rate = cats_df['max_flow_rate_per_drawoff_LperH'][0]
        timeseries_df.loc[timeseries_df['Water_LperH'] > global_max_flow_rate,
                          'Water_LperH'] = global_max_flow_rate

        timeseries_df['Water_L'] = timeseries_df['Water_LperH'] / 3600 * s_step

        timeseries_df['method'] = 'OpenDHW'
        timeseries_df['categories'] = len(cats_df.index)
        timeseries_df['drawoff_method'] = 'gauss_categories'
        timeseries_df['initial_day'] = initial_day
        timeseries_df['weekend_weekday_factor'] = weekend_weekday_factor
        timeseries_df['mean_drawoff_vol_per_day'] = mean_drawoff_vol_per_day

    elif categories == 1:

        drawoff_method = 'gauss_combined'
        mean_vol_per_drawoff = 8

        # deterministic
        timeseries_df = generate_yearly_probability_profile(
            s_step=s_step,
            weekend_weekday_factor=1.2,
            initial_day=0
        )

        timeseries_df, drawoffs = generate_drawoffs(
            timeseries_df=timeseries_df,
            method=drawoff_method,
            mean_vol_per_drawoff=mean_vol_per_drawoff,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

        timeseries_df = distribute_drawoffs(
            timeseries_df=timeseries_df,
            drawoffs=drawoffs,
        )

        timeseries_df['method'] = 'OpenDHW'
        timeseries_df['categories'] = 1
        timeseries_df['drawoff_method'] = drawoff_method
        timeseries_df['mean_drawoff_vol_per_day'] = mean_drawoff_vol_per_day
        timeseries_df[
            'sdtdev_drawoff_vol_per_day'] = mean_drawoff_vol_per_day / 4
        timeseries_df['initial_day'] = initial_day
        timeseries_df['weekend_weekday_factor'] = weekend_weekday_factor
        timeseries_df['mean_vol_per_drawoff'] = mean_vol_per_drawoff

    else:
        raise Exception("Unknown number of categories.")

    return timeseries_df


def get_data_drawoff_categories(s_step, mean_drawoff_vol_per_day):
    """
    get the data for the drawoofs when chooseing 4 categories.
    """

    cats_data_60 = {'mean_flow_rate_per_drawoff_LperH': [60, 360, 840, 480],
                    'drawoff_duration_min': [1, 1, 10, 5],
                    'portion': [0.14, 0.36, 0.1, 0.4],
                    'stddev_flow_rate_per_drawoff_LperH': [120, 120, 12, 24],
                    }

    cats_df = pd.DataFrame(data=cats_data_60)

    # if DHWcalc uses 4 categories with a timestep other than 60s,
    # the drawoffs data has to be altered.
    if s_step != 60:
        cats_df['drawoff_duration_min_old'] = cats_df['drawoff_duration_min']

        cats_df['drawoff_duration_min'] = int(s_step / 60)

        cats_df['conversion_factor'] = cats_df['drawoff_duration_min'] / \
                                       cats_df['drawoff_duration_min_old']

        cats_df['mean_flow_rate_per_drawoff_LperH'] \
            = cats_df['mean_flow_rate_per_drawoff_LperH'] / cats_df[
            'conversion_factor']
        cats_df['stddev_flow_rate_per_drawoff_LperH'] = \
            cats_df['stddev_flow_rate_per_drawoff_LperH'] / cats_df[
                'conversion_factor']

    # add more data to the category dataframe.
    cats_df['mean_vol_per_drawoff'] = \
        cats_df['mean_flow_rate_per_drawoff_LperH'] \
        / 60 * cats_df['drawoff_duration_min']

    cats_df['mean_vol_per_day'] = mean_drawoff_vol_per_day * cats_df[
        'portion']

    cats_df['mean_vol_per_year'] = cats_df['mean_vol_per_day'] * 365

    cats_df['mean_no_drawoffs_per_day'] = \
        cats_df['mean_vol_per_day'] / cats_df['mean_vol_per_drawoff']

    cats_df['mean_no_drawoffs_per_year'] = \
        cats_df['mean_no_drawoffs_per_day'] * 365

    # add max flow rate: Max(1200, highest category mean flow rate)
    cats_df['max_flow_rate_per_drawoff_LperH'] = max(cats_df[
        'mean_flow_rate_per_drawoff_LperH'].max(), 1200)


    return cats_df


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

    # todo: add profiles for non-residential buildings

    if mode == 'weekday':

        steps_and_ps = [(6.5, 0.01), (1, 0.5), (4.5, 0.06), (1, 0.16),
                        (5, 0.06), (4, 0.2), (2, 0.01)]

    elif mode == 'weekend':

        steps_and_ps = [(7, 0.02), (2, 0.475), (6, 0.071), (2, 0.237),
                        (3, 0.036), (3, 0.143), (1, 0.018)]

    else:
        raise Exception('Unknown Mode. Please Choose "Weekday" or "Weekend".')

    steps = [tup[0] for tup in steps_and_ps]
    ps = [tup[1] for tup in steps_and_ps]

    assert sum(steps) == 24
    assert sum(ps) == 1

    p_day = []

    for tup in steps_and_ps:
        p_lst = [tup[1] for _ in range(int(tup[0] * 3600 / s_step))]
        p_day.extend(p_lst)

    # check if length of daily intervals fits into the stepwidth. if s_step
    # f.e is 3600s (1h), one daily intervall cant be 4.5 hours.
    assert len(p_day) == 24 * 3600 / s_step

    if plot_p_day:
        plt.plot(p_day)
        plt.show()

    return p_day


def generate_yearly_probability_profile(s_step, weekend_weekday_factor=1.2,
                                        initial_day=0):
    """
    generate a summed yearly probabilty profile. The whole function is
    deterministc. The same inputs always produce the same outputs.

    1)  Probabilities for weekdays and weekend-days are loaded (p_we, p_wd).
    2)  Probability of weekend-days is increased relative to weekdays (shift).
    3)  Based on an initial day, the yearly probability distribution (p_final)
        is generated. The seasonal influence is modelled by a sine-function.
    4)  p_final is normalized and integrated. The sum over the year is thus
        equal to 1 (p_norm_integral).

    """

    # load daily probabilities (deterministic)
    p_we = generate_daily_probability_step_function(
        mode='weekend',
        s_step=s_step
    )

    p_wd = generate_daily_probability_step_function(
        mode='weekday',
        s_step=s_step
    )

    # shift towards weekend (deterministic)
    p_wd_weighted, p_we_weighted, av_p_week_weighted = shift_weekend_weekday(
        p_weekday=p_wd,
        p_weekend=p_we,
        factor=weekend_weekday_factor
    )

    # yearly curve (deterministic)
    p_final = generate_yearly_probabilities(
        initial_day=initial_day,
        p_weekend=p_we_weighted,
        p_weekday=p_wd_weighted,
        s_step=s_step
    )

    p_norm_integral = normalize_and_sum_list(lst=p_final)

    # make timeseries dataframe
    date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                               freq=str(s_step) + 'S')
    date_range = date_range[:-1]

    timeseries_df = pd.DataFrame(index=date_range,
                                 data={'p_norm_integral': p_norm_integral})

    return timeseries_df


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

    # av_p_wd = statistics.mean(p_weekday)
    # av_p_we = statistics.mean(p_weekend)
    # av_p_week = av_p_wd * 5 / 7 + av_p_we * 2 / 7

    p_wd_factor = 1 / (5 / 7 + factor * 2 / 7)
    p_we_factor = 1 / (1 / factor * 5 / 7 + 2 / 7)

    assert p_wd_factor * 5 / 7 + p_we_factor * 2 / 7 == 1

    p_wd_weighted = [p * p_we_factor for p in p_weekday]
    p_we_weighted = [p * p_we_factor for p in p_weekend]

    av_p_wd_weighted = statistics.mean(p_wd_weighted)
    av_p_we_weighted = statistics.mean(p_we_weighted)

    av_p_week_weighted = av_p_wd_weighted * 5 / 7 + av_p_we_weighted * 2 / 7

    return p_wd_weighted, p_we_weighted, av_p_week_weighted


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


def generate_and_distribute_drawoffs_cats(timeseries_df, cats_series):
    """
    when choosing "4 categories"

    :param      timeseries_df:          df:     holds the timeseries
    :param      cats_series:            series: constants for a category
    """

    # --- compute how many timesteps the drawoff occupies ---
    s_step = int(timeseries_df.index.freqstr[:-1])
    drawoff_duration = cats_series['drawoff_duration_min'] * 60
    drawoff_steps = int(drawoff_duration / s_step)

    # --- generate drawoffs until V_max is reached
    V_curr = 0
    V_max = cats_series['mean_vol_per_year']
    drawoffs = []  # L/h

    while V_curr <= V_max:
        # generate single drawoff and append to V_curr
        drawoff = generate_single_drawoff_inside_boundaries(cats_series)  # L/h
        drawoffs.append(drawoff)

        drawoff_L = drawoff / 3600 * s_step * drawoff_steps
        V_curr += drawoff_L

    # --- make the p_drawoffs list ---
    p_norm_integral = list(timeseries_df['p_norm_integral'])
    min_rand = min(p_norm_integral)
    max_rand = max(p_norm_integral)
    p_drawoffs = [random.uniform(min_rand, max_rand) for _ in range(
        len(drawoffs))]

    # --- sort both lists for the distribution algorithm ---
    p_drawoffs.sort()
    p_norm_integral.sort()

    # --- distribute drawoffs
    drawoff_count = 0

    # for return statement
    water_LperH = [0] * int(365 * 24 * 3600 / s_step)

    for time_step, p_current_sum in enumerate(p_norm_integral):

        if water_LperH[time_step] != 0:
            continue

        if p_drawoffs[drawoff_count] < p_current_sum:

            drawoff = drawoffs[drawoff_count]
            drawoffs_time_step_delta = [drawoff] * drawoff_steps

            for i in range(drawoff_steps):
                water_LperH[time_step + i] = drawoffs_time_step_delta[i]

            drawoff_count += 1

            if drawoff_count >= len(drawoffs):
                break

    cat_id = int(cats_series['mean_flow_rate_per_drawoff_LperH'])
    timeseries_df['Water_LperH_cat{}'.format(cat_id)] = water_LperH

    timeseries_df['Water_L_cat{}'.format(cat_id)] = \
        timeseries_df['Water_LperH_cat{}'.format(cat_id)] * s_step / 3600

    return timeseries_df


def generate_single_drawoff_inside_boundaries(cats_series):
    """
    from the data of one category, generate a drawoff.
    """

    # --- get mean and stddev from series ---
    mu = cats_series['mean_flow_rate_per_drawoff_LperH']  # in L/h
    sig = cats_series['stddev_flow_rate_per_drawoff_LperH']  # in L/h

    drawoff = random.gauss(mu, sig)

    # todo: max flow rates are sometime exceeded after the superposition of
    #  multiple drawoff categories!
    max_drawoff_flow_rate = cats_series['max_flow_rate_per_drawoff_LperH']
    min_drawoff_flow_rate = 1  # in L/h

    low_lim = max(float(mu - 2 * sig), min_drawoff_flow_rate)
    up_lim = min(float(mu + 2 * sig), max_drawoff_flow_rate)

    while drawoff < low_lim or drawoff > up_lim:
        drawoff = random.gauss(mu, sig)

    return drawoff  # in L/h


def generate_drawoffs(timeseries_df, mean_vol_per_drawoff=8,
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

    Then, the drawoff list are generated following either a Gauss
    Distribution (as describesd in the DHWcalc paper) or a beta distribution.

    :param timeseries_df:               df:     holds the timeseries
    :param mean_vol_per_drawoff:        int     mean volume per drawpff
    :param mean_drawoff_vol_per_day:    int     mean volume drawn off per day
    :param method:                      string  "gauss" or "beta"
    :return:    drawoffs:               list    drawoff events in [L/h]
                p_drawoffs:             list    probabilities 0...1
    """

    s_step = int(timeseries_df.index.freqstr[:-1])

    av_drawoff_flow_rate = mean_vol_per_drawoff * 3600 / s_step  # in L/h

    sdt_dev_drawoff_flow_rate = av_drawoff_flow_rate / 4  # in L/h

    mean_no_drawoffs_per_day = mean_drawoff_vol_per_day / mean_vol_per_drawoff

    total_drawoffs = int(mean_no_drawoffs_per_day * 365)

    timeseries_df['mean_drawoff_flow_rate_LperH'] = av_drawoff_flow_rate
    timeseries_df['sdtdev_drawoff_flow_rate_LperH'] = sdt_dev_drawoff_flow_rate
    timeseries_df['mean_no_drawoffs_per_day'] = mean_no_drawoffs_per_day

    # todo: dont hardcode this!
    if s_step <= 60:
        max_drawoff_flow_rate = 1200  # in L/h
    elif s_step <= 360:
        max_drawoff_flow_rate = 660  # in L/h
    elif s_step <= 600:
        max_drawoff_flow_rate = 414  # in L/h
    elif s_step <= 900:
        max_drawoff_flow_rate = 327  # in L/h
    else:
        max_drawoff_flow_rate = 250  # in L/h

    min_drawoff_flow_rate = 6  # in L/h

    if method == 'gauss_combined':
        # as close as it gets to the DHWcalc Algorithm

        mu = av_drawoff_flow_rate  # in L/h
        sig = sdt_dev_drawoff_flow_rate  # in L/h

        drawoffs = [random.gauss(mu, sig) for _ in range(total_drawoffs)]

        low_lim = int(mu - 2 * sig)
        up_lim = int(mu + 2 * sig)

        # cut gauss distribution, lowers standard-deviation. keeps Mean.
        drawoffs_reduced = [i for i in drawoffs if low_lim < i < up_lim]

        cut = [i for i in drawoffs if i <= low_lim or up_lim <= i]

        drawoffs = drawoffs_reduced

        mean_flow_rate_noise = ((max_drawoff_flow_rate - up_lim) / 2) + up_lim

        # after we cut the distribution, we have to distribute the remaining
        # drawoffs. Multiple Options possible.
        water_left = sum(cut) / 3600 * s_step  # in L

        hours_left = water_left / mean_flow_rate_noise

        no_drawoffs_left3 = int(hours_left * 3600 / s_step)

        no_drawoffs_left2 = int(water_left / mean_vol_per_drawoff)

        curr_no_drawoffs = len(drawoffs)
        no_drawoffs_left = total_drawoffs - curr_no_drawoffs

        noise = [random.randint(up_lim, max_drawoff_flow_rate) for _ in
                 range(no_drawoffs_left3)]

        drawoffs.extend(noise)

        # the underlying noise should be evenly distributed
        random.shuffle(drawoffs)

        # DHWcalc has a set flow rate step rather than a continuous
        # distribution. Thus, we round the drawoff distribution according to
        # this step width.
        # todo: looks like this a function of s_step!
        if s_step == 60:
            flow_rate_step = 6  # L/h
        else:
            flow_rate_step = 1
        drawoffs = [flow_rate_step * round(i / flow_rate_step) for i in
                    drawoffs]

    elif method == 'beta':
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

            drawoffs = [random.gauss(mu, sig) for _ in range(total_drawoffs)]
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

    else:
        raise Exception("Unkown method to generate drawoffs. choose Gauss or "
                        "Beta Distribution.")

    return timeseries_df, drawoffs


def distribute_drawoffs(timeseries_df, drawoffs):
    """
    Takes a small list (p_drawoffs) and sorts it into a bigger list (
    p_norm_integral). Both lists are being sorted. Then, the big list is
    iterated over, and whenever a value of the small list is smaller than a
    value of the big list, the index of the big list is saved and a drawoff
    event from the drawoffs list occurs.

    :param timeseries_df:   df:     holds the timeseries
    :param drawoffs:        list:   drawoff events in L/h
                                    probabilities [0...1]

    :return: water_LperH:   list:   resutling water drawoff profile
    """

    # if taken from the 'get_drawoffs' method, drawoffs have to be shuffled
    random.shuffle(drawoffs)

    s_step = int(timeseries_df.index.freqstr[:-1])
    p_norm_integral = list(timeseries_df['p_norm_integral'])

    min_rand = min(p_norm_integral)
    max_rand = max(p_norm_integral)
    p_drawoffs = [random.uniform(min_rand, max_rand) for _ in drawoffs]

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

    timeseries_df['Water_LperH'] = water_LperH
    timeseries_df['Water_LperSec'] = timeseries_df['Water_LperH'] / 3600
    timeseries_df['Water_L'] = timeseries_df['Water_LperSec'] * s_step

    return timeseries_df


def compute_heat(timeseries_df, temp_dT=35):
    """
    Takes a timeseries of waterflows per timestep in [L/h]. Computes a
    DHW Demand series in [kWh].

    :param timeseries_df:       Pandas Dataframe with all the timeseries
    :param temp_dT:     int:    temperature difference between freshwater and
                                average DHW outlet temperature. F.e. 35°C

    :return: timeseries_df:     Dataframe with added 'Heat' Column
    """

    timeseries_df['Heat_W'] = \
        timeseries_df['Water_LperH'] / 3600 * rho * cp * temp_dT

    s_step = int(timeseries_df.index.freqstr[:-1])
    timeseries_df['Heat_J'] = timeseries_df['Heat_W'] * s_step
    timeseries_df['Heat_kW'] = timeseries_df['Heat_W'] / 1000
    timeseries_df['Heat_kWh'] = timeseries_df['Heat_J'] / (3600 * 1000)

    return timeseries_df


def draw_lineplot(timeseries_df, plot_var='water', start_plot='2019-02-01',
                  end_plot='2019-02-05', save_fig=False):
    """
    Takes a timeseries of waterflows per timestep in [L/h]. Computes a
    DHW Demand series in [kWh]. Computes additional stats an optionally
    prints them out. Optionally plots the timesieries with additional stats.


    :param timeseries_df:       Pandas Dataframe that holds the timeseries.
    :param plot_var:    str:    choose to plot Water or Heat
    :param start_plot:  str:    start date of the plot. F.e. 2019-01-01
    :param end_plot:    str:    end date of the plot. F.e. 2019-02-01
    :param save_fig:    bool:   decide to save plots as pdf

    :return:    fig:    fig:    figure of the plot
                dhw:    list:   list of the heat demand for DHW for each
                                timestep in kWh.
    """

    # sns.set_style("white")
    sns.set_context("paper")

    fig, ax1 = plt.subplots()
    fig.tight_layout()

    # compute some stats for figure title
    yearly_water_demand = round(timeseries_df['Water_L'].sum(), 1)  # in L
    max_water_flow = round(timeseries_df['Water_LperH'].max(), 1)  # in L/h
    s_step = timeseries_df.index.freqstr
    method = timeseries_df['method'][0]

    if plot_var == 'water':
        # make dataframe for plotting with seaborn
        plot_df = timeseries_df[['Water_LperH', 'mean_drawoff_vol_per_day']]

        ax1 = sns.lineplot(ax=ax1, data=plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        ax1.legend(loc="upper left")

        title_str = make_title_str(timeseries_df=timeseries_df)
        ax1.set_title(title_str)

    if plot_var == 'heat':
        # make dataframe for plotting with seaborn
        plot_df = timeseries_df[['Heat_W']]

        ax1 = sns.lineplot(ax=ax1, data=plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_red])

        ax1.legend(loc="upper left")

        plt.title('Heat Time-series from {}, timestep = {}\n'
                  'with a Peak of {:.2f} L/h'.format(method, s_step,
                                                     max_water_flow))

    # set the x axis ticks
    # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    plt.show()

    if save_fig:
        dir_output = Path.cwd() / "plots"
        dir_output.mkdir(exist_ok=True)
        fig.savefig(dir_output / "Demand_{}_sliced.pdf".format(method))

    return fig


def draw_histplot(timeseries_df, extra_kde=False):
    """
    Takes a DHW profile and plots a histogram with some stats in the title
    :param timeseries_df:   Dataframe that holds the water timeseries
    :param extra_kde:       plot a detailed kde plot behind the main histogram.
    """

    # get non-zero values of the profile
    drawoffs_df = get_drawoffs(timeseries_df=timeseries_df,
                               col_part='Water_LperH')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # https://seaborn.pydata.org/generated/seaborn.histplot.html
    sns.histplot(data=drawoffs_df, ax=ax2, stat='count', kde=True,
                 kde_kws={'bw_adjust': 1})

    if extra_kde:
        # https://seaborn.pydata.org/generated/seaborn.kdeplot.html
        sns.kdeplot(data=drawoffs_df, ax=ax1, alpha=.05, bw_adjust=0.05,
                    legend=False, color='r')

    # title
    title_str = make_title_str(timeseries_df=timeseries_df)
    ax1.set_title(title_str)

    plt.show()


def draw_detailed_histplot(timeseries_df):
    """
    https://towardsdatascience.com/advanced-histogram-using-python-bceae288e715
    plot to further analyse timeseries with 1 drawoff category.

    counts  = numpy.ndarray of count of data ponts for each bin/column in the histogram
    bins    = numpy.ndarray of bin edge/range values
    patches = a list of Patch objects.
            each Patch object contains a Rectangle object.
            e.g. Rectangle(xy=(-2.51953, 0), width=0.501013, height=3, angle=0)
    """
    cats = timeseries_df['categories'][0]

    if cats == 1:

        # create bin values
        mean = timeseries_df['mean_drawoff_flow_rate_LperH'][0]
        sdtdev = timeseries_df['sdtdev_drawoff_flow_rate_LperH'][0]
        non_zero_min = timeseries_df[timeseries_df['Water_LperH'] > 0][
            'Water_LperH'].min()  # smallest entry that is not 0.

        bin_values = [non_zero_min,
                      mean - 2 * sdtdev,
                      mean - sdtdev,
                      mean,
                      mean + sdtdev,
                      mean + 2 * sdtdev,
                      timeseries_df['Water_LperH'].max()]
        bin_values = list(set(bin_values))  # remove double entries
        bin_values.sort()  # bins have to be sorted

        # get non-zero values of the profile
        drawoffs = timeseries_df[timeseries_df['Water_LperH'] != 0][
            'Water_LperH']

        # Plot the Histogram from the random data
        fig, (ax) = plt.subplots()

        counts, bins, patches = ax.hist(drawoffs, bins=bin_values,
                                        edgecolor='black')

        # Set the ticks to be at the edges of the bins.
        ax.set_xticks(bins.round(2))

        # Set the graph title and axes titles
        plt.ylabel('Count')
        plt.xlabel('Flowrate L/h')

        # Calculate bar centre to display the count of data points and %
        bin_x_centers = 0.5 * np.diff(bins) + bins[:-1]
        bin_y_centers = ax.get_yticks()[1] * 0.25

        # Display the the count of data points and % for each bar in histogram
        for i in range(len(bins) - 1):
            bin_label = "{0:,}".format(counts[i]) + "  ({0:.2f}%)".format(
                (counts[i] / counts.sum()) * 100)
            plt.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=90,
                     rotation_mode='anchor')

        # Display the graph
        plt.show()

    else:
        print('detailed histplot is only meant to analyse timeseries with one'
              'drawoff category.')


def add_additional_runs(timeseries_df, total_runs=5, save_to_csv=True):
    added_runs = total_runs - 1

    s_step = int(timeseries_df.index.freqstr[:-1])
    mean_vol_per_drawoff = timeseries_df['mean_vol_per_drawoff'][0]
    mean_drawoff_vol_per_day = timeseries_df['mean_drawoff_vol_per_day'][0]
    weekend_weekday_factor = timeseries_df['weekend_weekday_factor'][0]
    initial_day = timeseries_df['initial_day'][0]
    method = timeseries_df['method'][0]
    categories = timeseries_df['categories'][0]

    if method == 'OpenDHW':

        drawoff_method = timeseries_df['drawoff_method'][0]

        for run in range(added_runs):
            extra_timeseries_df = generate_dhw_profile(
                s_step=s_step,
                categories=categories,
                weekend_weekday_factor=weekend_weekday_factor,
                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
                initial_day=initial_day
            )

            additional_profile = extra_timeseries_df['Water_LperH']
            timeseries_df['Water_LperH_' + str(run)] = additional_profile

    elif method == 'DHWcalc':

        raise Exception('adding multiple plots for DWHcalc is not so useful, '
                        'as DHWcalc does not work with a random seed!')

        # categories = timeseries_df['categories'][0]
        #
        # for run in range(total_runs):
        #     extra_timeseries_df = import_from_dhwcalc(
        #         s_step=s_step,
        #         categories=categories,
        #         mean_drawoff_vol_per_day=mean_drawoff_vol_per_day
        #     )
        #
        #     additional_profile = extra_timeseries_df['Water_LperH']
        #     timeseries_df['Water_LperH_' + str(run)] = additional_profile

    if save_to_csv:
        # set a name for the file
        save_name = "{}_{}runs_{}L_{}min_{}LperDrawoff.csv".format(
            method, total_runs, mean_drawoff_vol_per_day, int(s_step / 60),
            mean_vol_per_drawoff)

        # make a directory. if it already exists, no problemooo, just use it
        save_dir = Path.cwd().parent / "Saved_Timeseries"
        save_dir.mkdir(exist_ok=True)

        # save the dataframe in the folder as a csv with the chosen name
        timeseries_df.to_csv(save_dir / save_name)

    return timeseries_df


def get_drawoffs(timeseries_df, col_part='Water_LperH'):
    """
    get sorted drawoff events from a timeseries Dataframe.
    """

    if col_part != 'all':
        # only get specific columns
        col_names = list(timeseries_df.columns)
        cols_LperH = [name for name in col_names if col_part in name]
        water_LperH_df = timeseries_df[cols_LperH]
    else:
        # the timeseries_df has already been cleaned from unwanted columns
        water_LperH_df = timeseries_df
        cols_LperH = list(timeseries_df.columns)

    #  generate drawoff Dataframe. initially, it has the same length as the
    #  timeseries_df. Index Column is not the DatetimeIndex anymore, as values
    #  in a single row do not correspond to a single Date!
    drawoffs_df = pd.DataFrame(columns=cols_LperH,
                               index=range(len(timeseries_df)))

    for col_name in cols_LperH:
        #  From each column, get only values > 0.
        drawoffs_series = water_LperH_df[water_LperH_df[col_name] != 0][
            col_name]
        drawoffs_lst = list(drawoffs_series)

        #  fill zero-values with NaN's
        empty_cells_len = len(timeseries_df) - len(drawoffs_lst)
        empty_cells_lst = [np.nan] * empty_cells_len
        drawoffs_lst.extend(empty_cells_lst)
        drawoffs_lst.sort()

        drawoffs_df[col_name] = drawoffs_lst

    # Drop rows that have only NaN's as values
    drawoffs_df = drawoffs_df.dropna(how='all')

    return drawoffs_df


def plot_multiple_runs(timeseries_df, plot_demands_overlay=False,
                       start_plot='2019-02-01', end_plot='2019-02-02',
                       plot_hist=True, plot_kde=True):
    """
    this function should only be used when the 'add_additional_runs' function
    has been used before.
    """

    drawoffs_df = get_drawoffs(timeseries_df=timeseries_df)
    cols_bool_str = timeseries_df.columns.str.contains('Water_LperH')
    water_LperH_df = timeseries_df.loc[:, cols_bool_str]

    if plot_demands_overlay:
        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1 = sns.lineplot(ax=ax1, data=water_LperH_df[start_plot:end_plot],
                           linewidth=0.5, legend=False)

        # set beautiful x axis ticks for datetime
        # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        plt.show()

    if plot_hist:
        sns.histplot(data=drawoffs_df, kde=False, element="step", fill=False,
                     stat='count', line_kws={'alpha': 0.8, 'linewidth': 0.9})
        plt.show()

    if plot_kde:
        sns.kdeplot(data=drawoffs_df, bw_adjust=0.1, alpha=0.5, fill=False,
                    linewidth=0.5, legend=False)
        plt.show()


def plot_multiple_timeseries(timeseries_lst, col_part='Water_LperH',
                             plot_demands_overlay=True,
                             start_plot='2019-02-01', end_plot='2019-02-02',
                             plot_hist=True, plot_kde=True):
    """
    only use with DHWcalc Timeseries tested.
    """

    plot_index = timeseries_lst[0].index
    plot_df = pd.DataFrame(index=plot_index)

    for i, df in enumerate(timeseries_lst):
        # get colum names
        cols_LperH = [name for name in list(df.columns) if col_part in name]

        # the timeseries_df should only have 1 column that matches the
        # desired string. more are not implemented yet
        assert len(cols_LperH) <= 1

        # fill the plot dataframe with the matching column
        plot_df[i] = df[cols_LperH]

    drawoffs_df = get_drawoffs(timeseries_df=plot_df, col_part='all')

    if plot_demands_overlay:
        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1 = sns.lineplot(ax=ax1, data=plot_df[start_plot:end_plot],
                           linewidth=0.5, legend=True)

        # set beautiful x axis ticks for datetime
        # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        plt.show()

    if plot_hist:
        sns.histplot(data=drawoffs_df, kde=True, element="step", fill=False,
                     stat='count', line_kws={'alpha': 0.8, 'linewidth': 0.9})
        plt.show()

    if plot_kde:
        sns.kdeplot(data=drawoffs_df, bw_adjust=0.1, alpha=0.5, fill=False,
                    linewidth=0.5, legend=True)
        plt.show()


def compare_generators(timeseries_df_1, timeseries_df_2,
                       start_plot='2019-03-01', end_plot='2019-03-08',
                       plot_date_slice=True, plot_distribution=True,
                       plot_detailed_distribution=True, save_fig=False):
    """
    Compares two methods of computing the water flow time series by means of
    a subplot.
    :param timeseries_df_1:     list:   first water flow time series
    :param timeseries_df_2:     list:   second water flow time series
    :param start_plot:          string: date, f.e. 2019-03-01
    :param end_plot:            string: date, f.e. 2019-03-08
    :param plot_date_slice
    :param plot_distribution
    :param plot_detailed_distribution
    :param save_fig:            bool:   decide to save the plot
    :return:
    """

    # detailed distribution is designed to compare timeseries with one
    # drawoff category.
    cats_1 = timeseries_df_1['categories'][0]
    cats_2 = timeseries_df_2['categories'][0]
    if cats_1 or cats_2 == 1:
        plot_detailed_distribution = False

    # compute Stats for the title
    drawoffs_1 = timeseries_df_1[timeseries_df_1['Water_LperH'] != 0][
        'Water_LperH']

    drawoffs_2 = timeseries_df_2[timeseries_df_2['Water_LperH'] != 0][
        'Water_LperH']

    if plot_date_slice:

        # make dataframe for plotting with seaborn
        plot_df_1 = timeseries_df_1[['Water_LperH', 'mean_drawoff_vol_per_day']]
        plot_df_2 = timeseries_df_2[['Water_LperH', 'mean_drawoff_vol_per_day']]

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout()

        # First Subplot
        ax1 = sns.lineplot(ax=ax1, data=plot_df_1[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        title_str_1 = make_title_str(timeseries_df=timeseries_df_1)
        ax1.set_title(title_str_1)

        ax1.legend(loc="upper left")

        # Second Subplot
        ax2 = sns.lineplot(ax=ax2, data=plot_df_2[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        title_str_2 = make_title_str(timeseries_df=timeseries_df_2)
        ax2.set_title(title_str_2)

        ax2.legend(loc="upper left")

        # --- set both aes to the same y limit ---
        ymin1, ymax1 = ax1.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()

        ymax_set = max(ymax1, ymax2)

        ax1.set_ylim(ymin1, ymax_set)
        ax2.set_ylim(ymin2, ymax_set)

        # --- beautiful x-ticks ---
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)

        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_Comparision.pdf")

    if plot_distribution:
        # compute Jensen Shannon Distance
        distance = jensen_shannon_distance(q=timeseries_df_1['Water_LperH'],
                                           p=timeseries_df_2['Water_LperH'])

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout()

        # plot the distribution
        # https://seaborn.pydata.org/generated/seaborn.displot.html
        ax1 = sns.histplot(ax=ax1, data=drawoffs_1, kde=True)
        ax2 = sns.histplot(ax=ax2, data=drawoffs_2, kde=True)

        # --- Set titles and Labels ---
        title_str_1 = make_title_str(timeseries_df=timeseries_df_1)
        title_str_1 = 'Jensen Shannon Distance = {:.4f} \n'.format(distance) \
                      + title_str_1
        ax1.set_title(title_str_1)

        ax1.set_ylabel('Count in a Year')

        title_str_2 = make_title_str(timeseries_df=timeseries_df_2)
        ax2.set_title(title_str_2)

        ax2.set_ylabel('Count in a Year')
        ax2.set_xlabel('Flowrate [L/h]')

        # --- set both axes to the same y limit ---
        ymin1, ymax1 = ax1.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()

        ymax_set = max(ymax1, ymax2)

        ax1.set_ylim(ymin1, ymax_set)
        ax2.set_ylim(ymin2, ymax_set)

        # --- set both axes to the same x limit ---
        xmin1, xmax1 = ax1.get_xlim()
        xmin2, xmax2 = ax2.get_xlim()

        xmax_set = max(xmax1, xmax2)

        ax1.set_xlim(xmin1, xmax_set)
        ax2.set_xlim(xmin2, xmax_set)

        plt.show()

    if plot_detailed_distribution:

        # https://towardsdatascience.com/advanced-histogram-using-python-bceae288e715

        # compute Jensen Shannon Distance
        distance = jensen_shannon_distance(q=timeseries_df_1['Water_LperH'],
                                           p=timeseries_df_2['Water_LperH'])

        fig, axes = plt.subplots(2, 1)
        ax1 = axes[0]
        ax2 = axes[1]
        fig.tight_layout()

        drawoffs_lst = [drawoffs_1, drawoffs_2]

        # create bin values
        mean1 = timeseries_df_1['mean_drawoff_flow_rate_LperH'][0]
        sdtdev1 = timeseries_df_1['sdtdev_drawoff_flow_rate_LperH'][0]
        non_zero_min1 = timeseries_df_1[timeseries_df_1['Water_LperH'] > 0][
            'Water_LperH'].min()  # smallest entry that is not 0.

        bin_values1 = [non_zero_min1,
                       mean1 - 2 * sdtdev1,
                       mean1 - sdtdev1,
                       mean1,
                       mean1 + sdtdev1,
                       mean1 + 2 * sdtdev1,
                       timeseries_df_1['Water_LperH'].max()]
        bin_values1 = list(set(bin_values1))  # remove double entries
        bin_values1.sort()  # bins have to be sorted

        mean2 = timeseries_df_2['mean_drawoff_flow_rate_LperH'][0]
        sdtdev2 = timeseries_df_2['sdtdev_drawoff_flow_rate_LperH'][0]
        non_zero_min2 = timeseries_df_2[timeseries_df_2['Water_LperH'] > 0][
            'Water_LperH'].min()  # smallest entry that is not 0.

        bin_values2 = [non_zero_min2,
                       mean2 - 2 * sdtdev2,
                       mean2 - sdtdev2,
                       mean2,
                       mean2 + sdtdev2,
                       mean2 + 2 * sdtdev2,
                       timeseries_df_2['Water_LperH'].max()]
        bin_values2 = list(set(bin_values2))  # remove double entries
        bin_values2.sort()  # bins have to be sorted

        bin_values_lst = [bin_values1, bin_values2]

        for sub_i, drawoffs_i in enumerate(drawoffs_lst):

            ax = axes[sub_i]

            counts, bins, patches = ax.hist(
                drawoffs_i, bins=bin_values_lst[sub_i], edgecolor='black')

            # Set the ticks to be at the edges of the bins.
            ax.set_xticks(bins.round(2))

            # Calculate bar centre to display the count of data points and %
            bin_x_centers = 0.1 * np.diff(bins) + bins[:-1]
            bin_y_centers = ax.get_yticks()[1] * 0.25

            # Display the the count of data points and % for each bar in hist
            for i in range(len(bins) - 1):
                bin_label = str(int(counts[i])) + "\n{0:.2f}%".format(
                    (counts[i] / counts.sum()) * 100)
                ax.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=0)

        title_str_1 = make_title_str(timeseries_df=timeseries_df_1)
        title_str_1 = 'Jensen Shannon Distance = {:.4f} \n'.format(distance) \
                      + title_str_1
        ax1.set_title(title_str_1)

        ax1.set_ylabel('Count in a Year')

        title_str_2 = make_title_str(timeseries_df=timeseries_df_2)
        ax2.set_title(title_str_2)

        ax2.set_ylabel('Count in a Year')
        ax2.set_xlabel('Flowrate [L/h]')

        # --- set both aes to the same y limit ---
        ymin1, ymax1 = ax1.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()

        ymax_set = max(ymax1, ymax2)

        ax1.set_ylim(ymin1, ymax_set)
        ax2.set_ylim(ymin2, ymax_set)

        plt.show()

    return


def plot_three_histplots(timeseries_df_1, timeseries_df_2, timeseries_df_3):
    """
    Compares two methods of computing the water flow time series by means of
    a subplot.
    :param timeseries_df_1:     list:   first water flow time series
    :param timeseries_df_2:     list:   second water flow time series
    """

    # detailed distribution is designed to compare timeseries with one
    # drawoff category.
    cats_1 = timeseries_df_1['categories'][0]
    cats_2 = timeseries_df_2['categories'][0]
    if cats_1 or cats_2 == 1:
        plot_detailed_distribution = False

    # compute Stats for the title
    drawoffs_1 = timeseries_df_1[timeseries_df_1['Water_LperH'] != 0][
        'Water_LperH']

    drawoffs_2 = timeseries_df_2[timeseries_df_2['Water_LperH'] != 0][
        'Water_LperH']

    drawoffs_3 = timeseries_df_3[timeseries_df_3['Water_LperH'] != 0][
        'Water_LperH']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.tight_layout()

    # plot the distribution
    # https://seaborn.pydata.org/generated/seaborn.displot.html
    ax1 = sns.histplot(ax=ax1, data=drawoffs_1, kde=True)
    ax2 = sns.histplot(ax=ax2, data=drawoffs_2, kde=True)
    ax3 = sns.histplot(ax=ax3, data=drawoffs_3, kde=True)

    # --- Set titles and Labels ---
    title_str_1 = make_title_str(timeseries_df=timeseries_df_1)
    ax1.set_title(title_str_1)
    ax1.set_ylabel('Count in a Year')

    title_str_2 = make_title_str(timeseries_df=timeseries_df_2)
    ax2.set_title(title_str_2)
    ax2.set_ylabel('Count in a Year')

    title_str_3 = make_title_str(timeseries_df=timeseries_df_3)
    ax3.set_title(title_str_3)
    ax3.set_ylabel('Count in a Year')
    ax3.set_xlabel('Flowrate [L/h]')



    # --- set both axes to the same y limit ---
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ymin3, ymax3 = ax3.get_ylim()

    ymax_set = max(ymax1, ymax2, ymax3)

    ax1.set_ylim(ymin1, ymax_set)
    ax2.set_ylim(ymin2, ymax_set)
    ax3.set_ylim(ymin3, ymax_set)

    # --- set both axes to the same x limit ---
    xmin1, xmax1 = ax1.get_xlim()
    xmin2, xmax2 = ax2.get_xlim()
    xmin3, xmax3 = ax3.get_xlim()

    xmax_set = max(xmax1, xmax2, xmax3)

    ax1.set_xlim(xmin1, xmax_set)
    ax2.set_xlim(xmin2, xmax_set)
    ax3.set_xlim(xmin3, xmax_set)

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


def get_s_step(timeseries_df):
    """
    get the seconds within a timestep from a pandas dataframe. When loading
    Dataframes from a csv, the index loses its 'freq' attribute. This is thus
    just a workaround when loading Timeseries from csv.
    """

    try:
        s_step = int(timeseries_df.index.freqstr[:-1])
        # todo: why doesnt this work for Dataframes loaded from a csv?

    except TypeError:

        steps = len(timeseries_df)
        secs_in_year = 8760 * 60 * 60
        s_step = secs_in_year / steps

        # check if s_step has no decimal points (should not be 60.01 f.e.)
        assert s_step % 1 == 0
        s_step = int(s_step)

    return s_step


def make_title_str(timeseries_df):
    """
    creates a title string based on the timerseries dataframe. The title
    string can then be used for a variety of plots.
    """

    # compute additional stats for title
    s_step = timeseries_df.index.freqstr
    yearly_water_demand = timeseries_df['Water_L'].sum()  # in L
    drawoffs = timeseries_df[timeseries_df['Water_LperH'] != 0]['Water_LperH']
    max_water_flow = timeseries_df['Water_LperH'].max()
    method = timeseries_df['method'][0]
    cats = timeseries_df['categories'][0]

    if cats == 1:
        method = "{} ({} cat)".format(method, cats)

        title_str = '{}, ∆t = {}, Yearly Demand = {:.1f} L \n' \
                    'No. Drawoffs = {}, Peak = {:.1f} L/h, ' \
                    'Mean = {:.1f} L/h, SdtDev = {:.1f} L/h'.format(
            method, s_step, yearly_water_demand, len(drawoffs), max_water_flow,
            drawoffs.mean(), drawoffs.std())

    else:  # f.e. four categories

        if 'OpenDHW' in method:

            method = "{} ({} cats)".format(method, cats)

            col_names = list(timeseries_df.columns)
            cols_LperH = [name for name in col_names if 'Water_L_' in name]
            water_LperH_df = timeseries_df[cols_LperH]

            cats_str = ''
            for col in cols_LperH:
                cat_sum = water_LperH_df[col].sum()
                cats_str += '{:.0f} L, '.format(cat_sum)
            cats_str = cats_str[:-2]

            title_str = '{}, ∆t = {}, No. Drawoffs = {}, Peak = {:.1f} L/h ' \
                        '\n Yearly Demand = {:.0f} L (= {})'.format(
                method, s_step, len(drawoffs), max_water_flow,
                yearly_water_demand, cats_str)

        elif 'DHWcalc' in method:

            method = "{} ({} cats)".format(method, cats)

            title_str = '{}, ∆t = {}, No. Drawoffs = {}, Peak = {:.1f} L/h ' \
                        '\n Yearly Demand = {:.0f} L'.format(
                method, s_step, len(drawoffs), max_water_flow,
                yearly_water_demand)

        else:
            raise Exception("Unkown method, try 'OpenDHW' or 'DHWcalc'.")

    return title_str


def resample_water_series(timeseries_df, s_step_output):
    """
    before resampling a dataframe, we have to choose which data has to be
    resampled in what way. some columns list constants, some list intensive
    properties (like L/h, kW) and some list extensive properties (Like
    Liters/kWh).
    Constants should stay the same, intensive properties should be averaged
    and extensive properties should be summed up.
    """

    s_step_old = get_s_step(timeseries_df)

    rule = str(s_step_output) + 'S'
    conversion_factor = s_step_output / s_step_old

    if conversion_factor != 1:
        # seperate constants from varaibles
        cols_consts = list(timeseries_df.columns[timeseries_df.nunique() <= 1])
        cols_vars = list(timeseries_df.columns[timeseries_df.nunique() > 1])

        # seperate flows (intensive) from sums (extensive)
        cols_flows = [i for i in cols_vars if 'Lper' in i]
        cols_sums = [i for i in cols_vars if i not in cols_flows]

        # seperate which constants change with the timestep (intensive properties)
        cols_const_flows = [i for i in cols_consts if 'Lper' in i]
        cols_consts = [i for i in cols_consts if i not in cols_const_flows]

        # make new sub-dataframes
        timeseries_df_sum = timeseries_df[cols_sums]
        timeseries_df_flows = timeseries_df[cols_flows]
        timeseries_df_consts = timeseries_df[cols_consts]
        timeseries_df_const_flows = timeseries_df[cols_const_flows]

        # resample them according to their physical properties
        timeseries_df_sum_re = timeseries_df_sum.resample(rule=rule).sum()
        timeseries_df_flows_re = timeseries_df_flows.resample(rule=rule).mean()

        # cut the dataframe with the constant variables and update the index
        resampled_index = timeseries_df_sum_re.index
        timeseries_df_consts_cut = timeseries_df_consts[0:len(resampled_index)]
        timeseries_df_consts_re \
            = timeseries_df_consts_cut.set_index(resampled_index)

        # update the constants that change with the timestep (intensive
        # properties) with the conversion factor
        timeseries_df_consts_flows_cut \
            = timeseries_df_const_flows[0:len(resampled_index)]
        timeseries_df_consts_flows_re \
            = timeseries_df_consts_flows_cut.set_index(resampled_index)
        timeseries_df_consts_flows_re \
            = timeseries_df_consts_flows_re / conversion_factor

        timeseries_df_re = pd.concat(
            [timeseries_df_flows_re, timeseries_df_sum_re,
             timeseries_df_consts_re, timeseries_df_consts_flows_re],
            axis=1)

        # add 'resampled' tag to method column
        timeseries_df_re['method'] = timeseries_df['method'][0] + ' (resampled)'

    else:
        timeseries_df_re = timeseries_df

    return timeseries_df_re
