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
import matplotlib.dates as mdates

import pycity_base.classes.demand.domestic_hot_water as dhw
import pycity_base.classes.timer as time
import pycity_base.classes.weather as weath
import pycity_base.classes.prices as price
import pycity_base.classes.environment as env
import pycity_base.classes.demand.occupancy as occ


def main():

    pass


def generate_dhw_profile_pycity_alias(s_step=60, initial_day=0,
                                      current_occupancy=5, temp_dT=35,
                                      print_stats=True, plot_demand=True,
                                      start_plot='2019-08-01',
                                      end_plot='2019-08-03', save_fig=False):
    """
    :param: s_step: int:        seconds within a time step. Should be
                                60s for pycity.
    :param: initial_day:        0: Mon, 1: Tue, 2: Wed, 3: Thur,
                                4: Fri, 5 : Sat, 6 : Sun
    :param: current_occuapncy:  number of people in the house. In PyCity,
                                this is a list and the occupancy changes during
                                the year. Between 0 and 5. Values have to be
                                integers. occupancy of 6 people seems to be
                                wrongly implemented in PyCity, as the sum of
                                the probabilities increases with occupancy (
                                1-5) but then decreases for the 6th person.
    :param: temp_dT: int/float: How much does the tap water has to be heated up?
    :return: water: List:       Tap water volume flow in liters per hour.
             heat : List:       Resulting minute-wise sampled heat demand in
                                Watt. The heat capacity of water is assumed
                                to be 4180 J/(kg.K) and the density is
                                assumed to be 980 kg/m3.
    """

    # get dhw stochastical file should be in the same dir as this script.

    profiles_path = Path.cwd() / 'dhw_stochastical.xlsx'
    profiles = {"we": {}, "wd": {}}
    book = xlrd.open_workbook(profiles_path)

    # Iterate over all sheets. wd = weekday, we = weekend. mw = ist the
    # average profile. occupancy is between 1-6 (we1 - we6).
    for sheetname in book.sheet_names():
        sheet = book.sheet_by_name(sheetname)

        # Read values
        values = [sheet.cell_value(i, 0) for i in range(1440)]

        # Store values in dictionary
        if sheetname in ("wd_mw", "we_mw"):
            profiles[sheetname] = values  # minute-wise average profile L/h
        elif sheetname[1] == "e":
            profiles["we"][int(sheetname[2])] = values  # probabilities 0 - 1
        else:
            profiles["wd"][int(sheetname[2])] = values  # probabilities 0 - 1

    # https://en.wikipedia.org/wiki/Geometric_distribution
    # occupancy is random, not a function of daytime! -> reasonable?
    timesteps_year = int(365 * 24 * 3600 / s_step)
    occupancy = np.random.geometric(p=0.8, size=timesteps_year) - 1  # [0, 2..]
    occupancy = np.minimum(5, occupancy)

    # time series for return statement
    water = []  # in L/h

    number_days = 365

    for day in range(number_days):

        # Is the current day on a weekend?
        if (day + initial_day) % 7 >= 5:
            probability_profiles = profiles["we"]
            average_profile = profiles["we_mw"]
        else:
            probability_profiles = profiles["wd"]
            average_profile = profiles["wd_mw"]

        water_daily = []

        # Compute seasonal factor
        arg = math.pi * (2 / 365 * day - 1 / 4)
        probability_season = 1 + 0.1 * np.cos(arg)

        timesteps_day = int(24 * 3600 / s_step)
        for t in range(timesteps_day):  # Iterate over all time-steps in a day

            first_timestep_day = day * timesteps_day
            last_timestep_day = (day + 1) * timesteps_day
            daily_occupancy = occupancy[first_timestep_day:last_timestep_day]
            current_occupancy = daily_occupancy[t]

            if current_occupancy > 0:
                probability_profile = probability_profiles[current_occupancy][t]
            else:
                probability_profile = 0

            # Compute probability for tap water demand at time t
            probability = probability_profile * probability_season

            # Check if tap water demand occurs. The higher the probability,
            # the more likely the if statement is true.
            if random.random() < probability:
                # Compute amount of tap water consumption. Start with seed?
                # This consumption has to be positive!
                water_t = random.gauss(average_profile[t], sigma=114.33)
                water_daily.append(abs(water_t))
            else:
                water_daily.append(0)

        # Include current_water and current_heat in water and heat
        water.extend(water_daily)

    water_LperH = water

    # Plot
    dhw_demand = compute_stats_and_plot_demand(
        method='DHWcalc_Alias',
        s_step=s_step,
        water_LperH=water_LperH,
        start_plot=start_plot,
        end_plot=end_plot,
        temp_dT=temp_dT,
        print_stats=print_stats,
        plot_demand=plot_demand,
        save_fig=save_fig
    )


def generate_dhw_profile_pycity(s_step=60, temp_dT=35, print_stats=True,
                                plot_demand=True, start_plot='2019-08-01',
                                end_plot='2019-08-03', save_fig=False):
    """
    from https://github.com/RWTH-EBC/pyCity
    :return:
    """
    #  Generate environment with timer, weather, and prices objects
    timer = time.Timer(time_discretization=s_step,  # in seconds
                       timesteps_total=int(365 * 24 * 3600 / s_step)
                       )

    weather = weath.Weather(timer=timer)
    prices = price.Prices()
    environment = env.Environment(timer=timer, weather=weather, prices=prices)

    #  Generate occupancy object with stochastic user profile
    occupancy = occ.Occupancy(environment=environment, number_occupants=5)

    dhw_obj = dhw.DomesticHotWater(
        environment=environment,
        t_flow=10 + temp_dT,  # DHW output temperature in degree Celsius
        method=2,  # Stochastic dhw profile, Method 1 not working
        supply_temperature=10,  # DHW inlet flow temperature in degree C.
        occupancy=occupancy.occupancy)  # Occupancy profile (600 sec resolution)
    dhw_demand = dhw_obj.loadcurve  # ndarray with 8760 timesteps in Watt

    # constants of pyCity:
    cp = 4180
    rho = 980 / 1000
    temp_diff = 35

    water_LperSec = [i / (rho * cp * temp_diff) for i in dhw_demand]
    water_LperH = [x * 3600 for x in water_LperSec]

    # Plot
    dhw_demand2 = compute_stats_and_plot_demand(
        method='DHWcalc_Alias',
        s_step=s_step,
        water_LperH=water_LperH,
        start_plot=start_plot,
        end_plot=end_plot,
        temp_dT=temp_dT,
        print_stats=print_stats,
        plot_demand=plot_demand,
        save_fig=save_fig
    )

    return dhw_demand, water_LperH


if __name__ == '__main__':
    main()
