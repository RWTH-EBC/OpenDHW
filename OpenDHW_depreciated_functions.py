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

    :param s_step:                      int:    timestep width in seconds. f.e.
                                                60. Seems like the Distribution
                                                is more similar to DHWcalc if
                                                always a stepwidth of 60S is
                                                chosen and later resampled with
                                                'resample_water_series'.
    :param weekend_weekday_factor:      int:    taken from DHWcalc
    :param drawoff_method:              str:    how drawoffs are computed
    :param mean_vol_per_drawoff:        int:    taken from DHWcalc
    :param mean_drawoff_vol_per_day:    int:    function of number of people in
                                                the house of floor area.
    :param initial_day:                 int:    0:Mon - 1:Tues ... 6:Sun
    :return: timeseries_df              df:     dataframe with all timeseries
    """

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
    timeseries_df['sdtdev_drawoff_vol_per_day'] = mean_drawoff_vol_per_day / 4
    timeseries_df['initial_day'] = initial_day
    timeseries_df['weekend_weekday_factor'] = weekend_weekday_factor
    timeseries_df['mean_vol_per_drawoff'] = mean_vol_per_drawoff

    return timeseries_df


def generate_dhw_profile_cats(s_step, weekend_weekday_factor=1.2,
                              initial_day=0, mean_drawoff_vol_per_day=200):
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

    4 categories standard values from DHWcalc:

                                        cat 1   cat2    cat3    cat4    sum
    mean flow rate per drawoff [L/h]    60      360     840     480
    drawoff duration [min]              1       1       10      5
    portion [%]                         14%     36%     10%     40%     100%
    sdt-dev [L/h]                       120     120     12      24

    --- when assuming 200L are drawn off per day:
    mean vol per drawoff [L]            1       6       140     40
    mean no. drawoffs per day [-]       28      12      0.142   2
    mean no. drawoffs per year [-]      10220   4380    51.8    730     15848
    mean vol per day [L]                28      72      20      80      200
    mean vol per year [L]               10220   26280   7300    29200   73000

    :param s_step:
    :param weekend_weekday_factor:
    :param initial_day:
    :return:
    """
    cats_data = {'mean_flow_rate_per_drawoff_LperH': [60, 360, 840, 480],
                 'drawoff_duration_min': [1, 1, 10, 5],
                 'portion': [0.14, 0.36, 0.1, 0.4],
                 'stddev_flow_rate_per_drawoff_LperH': [120, 120, 12, 24],
                 }

    cats_data_single = {'mean_flow_rate_per_drawoff_LperH': [480],
                        'drawoff_duration_min': [1],
                        'portion': [1],
                        'stddev_flow_rate_per_drawoff_LperH': [120],
                        }

    cats_df = pd.DataFrame(data=cats_data)

    # add more data to the category dataframe.
    cats_df['mean_vol_per_drawoff'] = \
        cats_df['mean_flow_rate_per_drawoff_LperH'] \
        / 60 * cats_df['drawoff_duration_min']

    cats_df['mean_vol_per_day'] = mean_drawoff_vol_per_day * cats_df['portion']

    cats_df['mean_vol_per_year'] = cats_df['mean_vol_per_day'] * 365

    cats_df['mean_no_drawoffs_per_day'] = \
        cats_df['mean_vol_per_day'] / cats_df['mean_vol_per_drawoff']

    cats_df['mean_no_drawoffs_per_year'] = \
        cats_df['mean_no_drawoffs_per_day'] * 365

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

    timeseries_df['Water_L'] = timeseries_df['Water_LperH'] / 3600 * s_step

    timeseries_df['method'] = 'OpenDHW'
    timeseries_df['categories'] = len(cats_df.index)
    timeseries_df['drawoff_method'] = 'gauss_categories'
    timeseries_df['initial_day'] = initial_day
    timeseries_df['weekend_weekday_factor'] = weekend_weekday_factor
    timeseries_df['mean_drawoff_vol_per_day'] = mean_drawoff_vol_per_day

    return timeseries_df


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

