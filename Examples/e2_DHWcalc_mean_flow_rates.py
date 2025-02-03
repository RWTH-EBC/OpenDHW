# -*- coding: utf-8 -*-
import OpenDHW
import pandas as pd

"""
This Example loads multiple TimeSeries from DHWcalc for one category, 
with different mean daily drawoff volumes.

One can see that the adjusted mean daily drawoff volumes results in new seeds
for the generation and distribution of the drawoff event.
"""

# --- Parameters ---
start_plot = '2019-03-04-06'
end_plot = '2019-03-04-10'

# --- Constants ---
categories = 1
s_steps = [60, 900, 3600]
mean_drawoff_vols_per_day = [39.4, 39.6, 39.8, 40, 40.2, 40.4, 40.6]
occupancy = 5


def main():

    for s_step in s_steps:

        timeseries_lst = []
        drawoffs_lst = []

        # for each chosen mean drawoff volume, import the DHWcalc Timeseries.
        for mean_drawoff_vol_per_day in mean_drawoff_vols_per_day:

            # Load time-series from DHWcalc
            dhwcalc_df = OpenDHW.import_from_dhwcalc(
                s_step=s_step,
                occupancy=occupancy,
                daylight_saving=False,
                categories=categories,
                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day
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


if __name__ == '__main__':
    main()
