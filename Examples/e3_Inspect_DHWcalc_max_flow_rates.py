# -*- coding: utf-8 -*-
import OpenDHW
import pandas as pd

"""
This Example loads multiple TimeSeries from DHWcalc for one category, 
with different max flow rates.

It looks like the adjusted maximum flow rate also results in new seeds 
for the main gauss curve, but not for drawoffs happening at the same time? 
"""

# --- Parameters ---
start_plot = '2019-03-04-06'
end_plot = '2019-03-04-10'

# --- Constants ---
s_step = 60
max_flowrates = [1188, 1194, 1200, 1206, 1212]


def main():

    timeseries_lst = []
    drawoffs_lst = []

    # for each chosen max. flowrate, import the DHWcalc Timeseries.
    for max_flowrate in max_flowrates:

        # Load time-series from DHWcalc
        dhwcalc_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            daylight_saving=False,
            categories=1,
            max_flowrate=max_flowrate
        )

        timeseries_lst.append(dhwcalc_df)
        drawoffs_lst.append(OpenDHW.get_drawoffs(dhwcalc_df))

    # plot all generated timeseries
    OpenDHW.plot_multiple_timeseries(
        timeseries_lst=timeseries_lst,
        start_plot=start_plot,
        end_plot=end_plot,
        plot_hist=False
    )

    pass


if __name__ == '__main__':
    main()
