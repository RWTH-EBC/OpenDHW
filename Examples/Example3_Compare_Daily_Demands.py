# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example loads DHWcalc Timeseries with different daily water demands and 
compares them to their OpenDHW Equivalent. Once can see that the daily Peak 
and the total number of drawoffs still differs significantly:

L/d Mean L/Draw	Mean Draws/d	Theroretical Draws/a	Actual Draws/a	Diff
160	8	        20	            7300	                7017	        283
200	8	        25	            9125	                8753	        372
240	8	        30	            10950	                10363	        587

This has to do with the Behaviour of DHWcalc when cutting the original Gauss 
Distribution. (OpenDHW/DHWcalc_Screenshots/99_Theory_Drawoffs_1Category.png)
"""

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-04-01'
end_plot = '2019-04-14'


def main():

    # Load time-series from DHWcalc
    dhwcalc_60_1_200_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step, categories=1, mean_drawoff_vol_per_day=200,
        daylight_saving=False)

    dhwcalc_60_1_240_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step, categories=1, mean_drawoff_vol_per_day=240,
        daylight_saving=False)
    # todo: why is maxflow of 1200 not reached here?

    dhwcalc_60_1_160_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step, categories=1, mean_drawoff_vol_per_day=160,
        daylight_saving=False)

    # generate time-series with OpenDHW
    open_dhw_60_200_df = OpenDHW.generate_dhw_profile(
        s_step=s_step, mean_drawoff_vol_per_day=200)

    open_dhw_60_240_df = OpenDHW.generate_dhw_profile(
        s_step=s_step, mean_drawoff_vol_per_day=240)

    open_dhw_60_160_df = OpenDHW.generate_dhw_profile(
        s_step=s_step, mean_drawoff_vol_per_day=160)

    # compare  time-series from DWHcalc and OpenDHW
    OpenDHW.compare_generators(
        timeseries_df_1=dhwcalc_60_1_200_df,
        timeseries_df_2=open_dhw_60_200_df,
        start_plot=start_plot,
        end_plot=end_plot,
    )

    OpenDHW.compare_generators(
        timeseries_df_1=dhwcalc_60_1_240_df,
        timeseries_df_2=open_dhw_60_240_df,
        start_plot=start_plot,
        end_plot=end_plot,
    )

    OpenDHW.compare_generators(
        timeseries_df_1=dhwcalc_60_1_160_df,
        timeseries_df_2=open_dhw_60_160_df,
        start_plot=start_plot,
        end_plot=end_plot,
    )


if __name__ == '__main__':
    main()
