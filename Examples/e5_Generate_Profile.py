# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example generates an OpenDHW Timeseries.

Like DHWcalc, OpenDHW need a parametrisation. The usual parametrisation for 
households is 40 L/(person*day). Thus, a 5 person single family house has a mean
drawoff volume of 200 L/day. More information can be found in 
'OpenDHW/Resources_Water_Demand_Parametrisation'.

A multi family house with 10 flats, and 5 people per flat thus has a mean 
drawoff rate of 2000 L/day.

For non-residental buildings, the daily probability functions should be 
altered, as there are no typical shower or cooking periods in the morning
and the evening.

As an alternative for timesteps different from 60seconds, a second method has 
been implemented in OpenDHW: the 'resample_water_series' function.

The idea is to always generate a timeseries with s_step=60s and then resample it
afterwards to the desired output stepwidth. The disadvantage is a higher 
computing time.
"""


# --- Parameters ---
s_step = 60
resample_method = False
mean_drawoff_vol_per_day = 200
categories = 1
start_plot = '2019-03-31-06'
end_plot = '2019-03-31-09'
temp_dT = 35    # K

# --- Constants ---


def main():

    if not resample_method:
        # generate time-series with OpenDHW
        timeseries_df = OpenDHW.generate_dhw_profile(
            s_step=s_step,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

    else:

        # generate time-series with OpenDHW
        timeseries_df = OpenDHW.generate_dhw_profile(
            s_step=60,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

        # resample to the desired stepwidth
        timeseries_df = OpenDHW.resample_water_series(
            timeseries_df=timeseries_df, s_step_output=s_step)

    # Compute Heat from Water TimeSeries
    timeseries_df = OpenDHW.compute_heat(
        timeseries_df=timeseries_df,
        temp_dT=temp_dT
    )

    # example timeseries which could be fed into Dynamic Simulations
    water_series = timeseries_df['Water_L']
    heat_series = timeseries_df['Heat_kWh']

    # inspect the drawoff events
    drawoffs_df = timeseries_df[timeseries_df['Water_LperH'] != 0]

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot(timeseries_df=timeseries_df, extra_kde=False,
                          save_fig=True)

    # Generate Lineplot from the loaded timeseries
    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot, save_fig=True)

    # Generate detailed Histogram from the loaded timeseries
    OpenDHW.draw_detailed_histplot(timeseries_df=timeseries_df)


if __name__ == '__main__':
    main()
