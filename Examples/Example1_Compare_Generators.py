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

    # generate time-series with OpenDHW
    timeseries_df_gauss = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        weekend_weekday_factor=1.2,
        mean_vol_per_drawoff=8,
        mean_drawoff_vol_per_day=200,
        drawoff_method='gauss_combined',
        initial_day=0,
    )

    # Load time-series from DHWcalc
    timeseries_df_dhwcalc = OpenDHW.import_from_dhwcalc(s_step=s_step,
                                                        categories=1)

    # compare two time-series
    OpenDHW.compare_generators(
        timeseries_df_1=timeseries_df_dhwcalc,
        timeseries_df_2=timeseries_df_gauss,
        start_plot=start_plot,
        end_plot=end_plot,
    )


if __name__ == '__main__':
    main()
