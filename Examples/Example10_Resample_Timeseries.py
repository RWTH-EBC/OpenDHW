# -*- coding: utf-8 -*-
import OpenDHW

"""
Resampling a base timestep of 60S yields different distributions compared to 
generating a distribution with different initial timesteps. (why?)
"""

# --- Parameters ---
s_steps = [360, 600, 900]
start_plot = '2019-03-04'
end_plot = '2019-03-08'

# --- Constants ---
mean_drawoff_vol_per_day = 200
categories = 1


def main():

    # generate base time-series with OpenDHW
    timeseries_df_base = OpenDHW.generate_dhw_profile(
        s_step=60,
        categories=categories,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
    )

    for s_step in s_steps:

        # either take the base timeseries and resample it
        timeseries_df_resampled = OpenDHW.resample_water_series(
            timeseries_df=timeseries_df_base, s_step_output=s_step)

        # or make a new timeseries with a different s_step
        timeseries_df = OpenDHW.generate_dhw_profile(
            s_step=s_step,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

        # import from DHWcalc
        timeseries_df_dhwcalc = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False)

        # compare the resampled series with the non-resampled one.
        OpenDHW.compare_generators(
            timeseries_df_1=timeseries_df_resampled,
            timeseries_df_2=timeseries_df,
            start_plot=start_plot,
            end_plot=end_plot,
        )


if __name__ == '__main__':
    main()
