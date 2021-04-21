# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example computes two TimeSeries with OpenDHW and load one Timeseries 
from DHWcalc. Then each OpenDHW TimeSeries is compared with the DHWcalc 
Timeseries.
"""

# --- Parameters ---
s_steps = [60, 360, 600, 900]
categories = 4
resample_method = True
start_plot = '2019-03-04'
end_plot = '2019-03-08'

# --- Constants ---
mean_drawoff_vol_per_day = 200


def main():

    for s_step in s_steps:

        # Load time-series from DHWcalc
        dhwcalc_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False
        )

        if not resample_method:
            # generate time-series with OpenDHW
            open_dhw_df = OpenDHW.generate_dhw_profile(
                s_step=s_step,
                categories=categories,
                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            )

        else:

            # generate time-series with OpenDHW
            open_dhw_df = OpenDHW.generate_dhw_profile(
                s_step=60,
                categories=categories,
                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            )

            # resample to the desired stepwidth
            open_dhw_df = OpenDHW.resample_water_series(
                timeseries_df=open_dhw_df, s_step_output=s_step)

        # compare time-series
        OpenDHW.compare_generators(
            timeseries_df_1=dhwcalc_df,
            timeseries_df_2=open_dhw_df,
            start_plot=start_plot,
            end_plot=end_plot,
            plot_date_slice=False
        )


if __name__ == '__main__':
    main()
