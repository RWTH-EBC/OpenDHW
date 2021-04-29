# -*- coding: utf-8 -*-
from OpenDHW import OpenDHW as OpenDHW

"""
This example is a final test of OpenDHW.
It uses the main step widths used in dynamic simulations: 15mins and 1hour.
It uses a broad range of daily demands and both numbers of categories.
"""

# --- Parameters ---
daily_demands = [160, 200, 240, 2000]  # 160, 200, 240, 2000
s_steps = [60, 900, 3600]   # 60, 360, 600, 900
categories_lst = [1, 4]
resample_method = True
plot_date_slice = False
start_plot = '2019-04-05'
end_plot = '2019-04-07'

# --- Constants ---


def main():

    for daily_demand in daily_demands:

        for s_step in s_steps:

            for categories in categories_lst:

                # Load time-series from DHWcalc
                dhwcalc_df = OpenDHW.import_from_dhwcalc(
                    s_step=s_step,
                    categories=categories,
                    mean_drawoff_vol_per_day=daily_demand,
                    daylight_saving=False
                )

                if not resample_method:
                    # generate time-series with OpenDHW
                    open_dhw_df = OpenDHW.generate_dhw_profile(
                        s_step=s_step,
                        categories=categories,
                        mean_drawoff_vol_per_day=daily_demand,
                    )

                else:

                    # generate time-series with OpenDHW
                    open_dhw_df = OpenDHW.generate_dhw_profile(
                        s_step=60,
                        categories=categories,
                        mean_drawoff_vol_per_day=daily_demand,
                    )

                    # resample to the desired stepwidth
                    open_dhw_df = OpenDHW.resample_water_series(
                        timeseries_df=open_dhw_df, s_step_output=s_step)

                # compare both time-series
                OpenDHW.compare_generators(
                    timeseries_df_1=dhwcalc_df,
                    timeseries_df_2=open_dhw_df,
                    start_plot=start_plot,
                    end_plot=end_plot,
                    plot_date_slice=plot_date_slice,
                    plot_detailed_distribution=False
                )


if __name__ == '__main__':
    main()
