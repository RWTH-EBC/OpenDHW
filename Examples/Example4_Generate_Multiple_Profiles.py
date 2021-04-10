# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example generates multiple TimeSeries at once and saves them as a CSV. 
First, a single DHW profile is generated. Then, additional profiles with the 
same settings as the original profile are generated and appended to the main 
dataframe.
"""

# ------- Parameter Section ---------
s_step = 600


def main():

    # generate timeseries
    timeseries_df = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        weekend_weekday_factor=1.2,
        mean_vol_per_drawoff=8,
        mean_drawoff_vol_per_day=200,
        initial_day=0,
    )

    timeseries_df = OpenDHW.add_additional_runs(
        timeseries_df=timeseries_df, total_runs=5, save_to_csv=True)


if __name__ == '__main__':
    main()
