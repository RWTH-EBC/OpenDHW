# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example load a single TimeSeries from DHWcalc and generates a Histogram 
and a Lineplot from it. This is tested for multiple timestep widths and two 
numbers of DHWcalc categories (1 and 4).

One can see very different distributions for all settings.
"""

# --- Parameters ---
s_steps = [60, 360, 600, 900, 3600]
start_plot = '2019-03-04'
end_plot = '2019-03-08'
draw_lineplot = False   # does not really add insights here

# --- Constants ---
mean_drawoff_vol_per_day = 200


def main():

    for s_step in s_steps:

        # 1 category
        timeseries_1cat_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=1,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False)

        drawoffs_1cat = OpenDHW.get_drawoffs(timeseries_1cat_df)

        drawoffs_1cat_peak = timeseries_1cat_df[timeseries_1cat_df['Water_LperH']
                                                > 1100]['Water_LperH']

        OpenDHW.draw_histplot(timeseries_df=timeseries_1cat_df)

        if draw_lineplot:
            OpenDHW.draw_lineplot(timeseries_df=timeseries_1cat_df,
                                  start_plot=start_plot,
                                  end_plot=end_plot)

        # 4 categories
        timeseries_4cat_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=4,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False)

        drawoffs_4cat = OpenDHW.get_drawoffs(timeseries_4cat_df)

        drawoffs_4cat_peak = timeseries_4cat_df[timeseries_4cat_df['Water_LperH']
                                                > 1100]['Water_LperH']

        OpenDHW.draw_histplot(timeseries_df=timeseries_4cat_df)

        if draw_lineplot:
            OpenDHW.draw_lineplot(timeseries_df=timeseries_4cat_df,
                                  start_plot=start_plot,
                                  end_plot=end_plot)


if __name__ == '__main__':
    main()
