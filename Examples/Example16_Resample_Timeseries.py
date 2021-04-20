# -*- coding: utf-8 -*-
import OpenDHW

"""
Resampling a base timestep of 60S yields different distributions compared to 
generating a distribution with different initial timesteps. (why?)
"""

# --- Parameters ---
s_steps = [360, 600, 900]

people = 5
start_plot = '2019-03-31'
end_plot = '2019-04-01'

# --- Constants ---
s_step_base = 60
mean_drawoff_vol_per_day_and_person = 40
mean_drawoff_vol_per_day = mean_drawoff_vol_per_day_and_person * people


def main():

    # generate time-series with OpenDHW
    timeseries_df = OpenDHW.generate_dhw_profile(
        s_step=60,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
    )

    OpenDHW.draw_histplot(timeseries_df=timeseries_df)

    for s_step in s_steps:

        timeseries_df = OpenDHW.resample_water_series(
            timeseries_df=timeseries_df, s_step_output=s_step)

        OpenDHW.draw_histplot(timeseries_df=timeseries_df)


if __name__ == '__main__':
    main()
