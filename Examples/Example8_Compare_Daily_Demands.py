# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example loads DHWcalc Timeseries (1 category) with different daily water 
demands and compares them to their OpenDHW Equivalent.

This has to do with the Behaviour of DHWcalc when cutting the original Gauss 
Distribution and adding noise. The behaviour is still not perfectly 
reverse-engineered:
(OpenDHW/DHWcalc_Screenshots/99_Theory_Drawoffs_1Category.png)
"""

# --- Parameters ---
daily_demands = [160, 200, 240]  # L
s_step = 60
start_plot = '2019-04-05'
end_plot = '2019-04-07'
plot_date_slice = False
plot_detailed_distribution = True

# --- Constants ---
categories = 1


def main():

    for daily_deamnd in daily_demands:

        # Load time-series from DHWcalc
        dhwcalc_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=categories,
            mean_drawoff_vol_per_day=daily_deamnd,
            daylight_saving=False
        )

        # generate time-series with OpenDHW
        open_dhw_df = OpenDHW.generate_dhw_profile(
            s_step=s_step,
            mean_drawoff_vol_per_day=daily_deamnd
        )

        # compare both time-series
        OpenDHW.compare_generators(
            timeseries_df_1=dhwcalc_df,
            timeseries_df_2=open_dhw_df,
            start_plot=start_plot,
            end_plot=end_plot,
            plot_date_slice=plot_date_slice,
            plot_detailed_distribution=plot_detailed_distribution
        )


if __name__ == '__main__':
    main()
