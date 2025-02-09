# -*- coding: utf-8 -*-
import OpenDHW
from pathlib import Path

"""
This Example generates multiple TimeSeries at once and saves them as a CSV. 
First, a single DHW profile is generated. Then, additional profiles with the 
same settings as the original profile are generated and appended to the main 
dataframe.
"""

# --- Parameters ---
s_step = 600
runs = 5

# --- constants ---
mean_drawoff_vol_per_day = 200
categories = 4
dir_output = Path.cwd().parent / "Saved_Timeseries"


def main():

    # generate timeseries
    timeseries_df = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        categories=categories,
        holidays=[1, 93, 96, 121, 134, 145, 155, 275, 305, 358, 359, 360, 365], # Julian day number of the holidays in NRW in 2015
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
    )

    timeseries_df = OpenDHW.add_additional_runs(
        timeseries_df=timeseries_df,
        total_runs=runs,
        holidays=[1, 93, 96, 121, 134, 145, 155, 275, 305, 358, 359, 360, 365], # Julian day number of the holidays in NRW in 2015
        dir_output=dir_output)


if __name__ == '__main__':
    main()
