# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example computes two TimeSeries with OpenDHW and load one Timeseries 
from DHWcalc. Then each OpenDHW TimeSeries is compared with the DHWcalc 
Timeseries.
"""

# --- Parameters ---
s_steps = [60, 360, 600, 900]
start_plot = '2019-03-04'
end_plot = '2019-03-08'

# --- Constants ---
mean_drawoff_vol_per_day = 200


def main():

    for s_step in s_steps:

        # Load time-series from DHWcalc
        dhwcalc_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=1,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False
        )

        # generate time-series with OpenDHW
        open_dhw_df = OpenDHW.generate_dhw_profile(
            s_step=s_step,
            mean_vol_per_drawoff=mean_drawoff_vol_per_day,
        )

        # compare time-series
        OpenDHW.compare_generators(
            timeseries_df_1=dhwcalc_df,
            timeseries_df_2=open_dhw_df,
            start_plot=start_plot,
            end_plot=end_plot,
        )


if __name__ == '__main__':
    main()
