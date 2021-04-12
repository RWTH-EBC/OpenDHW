# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example load a single TimeSeries from DHWcalc and generates a Histogram 
and a Lineplot from it. This is tested for two timestep widths: 60 and 600 
seconds (1 min and 10 mins) and two number of DHWcalc categories (1 and 4).

One can see very different distributions for the 4 settings and the challenge
to reverse-engineer the DHWcalc algorithm.
"""

# ------- Parameter Section ---------
s_steps = [60, 360, 600, 900]
start_plot = '2019-03-04'
end_plot = '2019-03-08'
draw_lineplot = True   # does not really add insights here


def main():

    for s_step in s_steps:

        # 1 category
        timeseries_1cat_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=1,
            mean_drawoff_vol_per_day=200,
            daylight_saving=False)

        OpenDHW.draw_histplot(timeseries_df=timeseries_1cat_df)

        if draw_lineplot:
            OpenDHW.draw_lineplot(timeseries_df=timeseries_1cat_df,
                                  start_plot=start_plot,
                                  end_plot=end_plot)

        # 4 categories
        timeseries_4cat_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=4,
            mean_drawoff_vol_per_day=200,
            daylight_saving=False)

        OpenDHW.draw_histplot(timeseries_df=timeseries_4cat_df)

        if draw_lineplot:
            OpenDHW.draw_lineplot(timeseries_df=timeseries_4cat_df,
                                  start_plot=start_plot,
                                  end_plot=end_plot)


if __name__ == '__main__':
    main()
