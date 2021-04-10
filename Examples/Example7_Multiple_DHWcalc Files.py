# -*- coding: utf-8 -*-
import OpenDHW
import pandas as pd

"""
This Example loads multiple TimeSeries from DHWcalc, with different max flow 
rates.
"""

# ------- Parameter Section ---------
start_plot = '2019-03-04-06'
end_plot = '2019-03-04-10'


def main():

    # Load time-series from DHWcalc
    dhwcalc_1188_df = OpenDHW.import_from_dhwcalc(
        s_step=60, daylight_saving=False, max_flowrate=1188)

    dhwcalc_1194_df = OpenDHW.import_from_dhwcalc(
        s_step=60, daylight_saving=False, max_flowrate=1194)

    dhwcalc_1200_df = OpenDHW.import_from_dhwcalc(
        s_step=60, daylight_saving=False, max_flowrate=1200)

    dhwcalc_1206_df = OpenDHW.import_from_dhwcalc(
        s_step=60, daylight_saving=False, max_flowrate=1206)

    dhwcalc_1212_df = OpenDHW.import_from_dhwcalc(
        s_step=60, daylight_saving=False, max_flowrate=1212)

    # put all dataframes in a list
    df_lst = [dhwcalc_1188_df, dhwcalc_1194_df, dhwcalc_1200_df,
              dhwcalc_1206_df, dhwcalc_1212_df]

    # plot that list
    OpenDHW.plot_multiple_timeseries(
        timeseries_lst=df_lst,
        start_plot=start_plot,
        end_plot=end_plot,
        plot_hist=False
    )


if __name__ == '__main__':
    main()
