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
"""
# Todo: why DHWcalc yearly Drawoffs differ from expected yearly Drawoffs?

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-04-01'
end_plot = '2019-04-14'


def main():

    # Load time-series from DHWcalc
    water_dhwcalc_60_1_200 = OpenDHW.import_from_dhwcalc(s_step=60,
                                                         categories=1,
                                                         daily_demand=200)

    water_dhwcalc_60_1_240 = OpenDHW.import_from_dhwcalc(s_step=60,
                                                         categories=1,
                                                         daily_demand=240)

    water_dhwcalc_60_1_160 = OpenDHW.import_from_dhwcalc(s_step=60,
                                                         categories=1,
                                                         daily_demand=160)

    # generate time-series with OpenDHW
    water_open_dhw_60_200 = OpenDHW.generate_dhw_profile(
        s_step=s_step, mean_drawoff_vol_per_day=200)

    water_open_dhw_60_160 = OpenDHW.generate_dhw_profile(
        s_step=s_step, mean_drawoff_vol_per_day=160)

    water_open_dhw_60_240 = OpenDHW.generate_dhw_profile(
        s_step=s_step, mean_drawoff_vol_per_day=240)

    # compare two time-series from DWHcalc
    OpenDHW.compare_generators(
        first_method='OpenDHW-240',
        first_series_LperH=water_open_dhw_60_240,
        second_method='DHWcalc-240',
        second_series_LperH=water_dhwcalc_60_1_240,
        s_step=s_step,
        start_plot=start_plot,
        end_plot=end_plot,
    )

    # compare two time-series from DWHcalc
    OpenDHW.compare_generators(
        first_method='OpenDHW-160',
        first_series_LperH=water_open_dhw_60_160,
        second_method='DHWcalc-160',
        second_series_LperH=water_dhwcalc_60_1_160,
        s_step=s_step,
        start_plot=start_plot,
        end_plot=end_plot,
    )

    # compare two time-series from DWHcalc
    OpenDHW.compare_generators(
        first_method='OpenDHW-200',
        first_series_LperH=water_open_dhw_60_200,
        second_method='DHWcalc-200',
        second_series_LperH=water_dhwcalc_60_1_200,
        s_step=s_step,
        start_plot=start_plot,
        end_plot=end_plot,
    )


if __name__ == '__main__':
    main()
