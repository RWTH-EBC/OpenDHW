# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example computes two TimeSeries with OpenDHW and load one Timeseries 
from DHWcalc. Then each OpenDHW TimeSeries is compared with the DHWcalc 
Timeseries.
"""

# --- Parameter Section ---
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    # Load time-series from DHWcalc
    dhwcalc_60_df = OpenDHW.import_from_dhwcalc(
        s_step=60, categories=1, mean_drawoff_vol_per_day=200,
        daylight_saving=False)

    dhwcalc_600_df = OpenDHW.import_from_dhwcalc(
        s_step=600, categories=1, mean_drawoff_vol_per_day=200,
        daylight_saving=False)

    # generate time-series with OpenDHW
    open_dhw_60_df = OpenDHW.generate_dhw_profile(
        s_step=60, mean_drawoff_vol_per_day=200)

    open_dhw_600_df = OpenDHW.generate_dhw_profile(
        s_step=600, mean_drawoff_vol_per_day=200)

    # compare  time-series from DWHcalc and OpenDHW
    OpenDHW.compare_generators(
        timeseries_df_1=dhwcalc_60_df,
        timeseries_df_2=open_dhw_60_df,
        start_plot=start_plot,
        end_plot=end_plot,
    )

    OpenDHW.compare_generators(
        timeseries_df_1=dhwcalc_600_df,
        timeseries_df_2=open_dhw_600_df,
        start_plot=start_plot,
        end_plot=end_plot,
    )


if __name__ == '__main__':
    main()
