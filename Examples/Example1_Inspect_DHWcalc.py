# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example load a single TimeSeries from DHWcalc and generates a Histogram 
and a Lineplot from it. This is tested for two timestep widths: 60 and 600 
seconds (1 min and 10 mins)
"""

# ------- Parameter Section ---------
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    # 60 seconds time step
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=60, categories=1,
                                                mean_drawoff_vol_per_day=200,
                                                daylight_saving=False)

    OpenDHW.draw_histplot(profile_df=timeseries_df)

    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)

    # 600 seconds timestep
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=600, categories=1,
                                                mean_drawoff_vol_per_day=200,
                                                daylight_saving=False)

    OpenDHW.draw_histplot(profile_df=timeseries_df)

    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)

    # 4 categories, 60 seconds
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=60, categories=4,
                                                mean_drawoff_vol_per_day=200,
                                                daylight_saving=True)

    OpenDHW.draw_histplot(profile_df=timeseries_df)

    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)

    # 4 categories, 600 seconds
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=600, categories=4,
                                                mean_drawoff_vol_per_day=200,
                                                daylight_saving=True)

    OpenDHW.draw_histplot(profile_df=timeseries_df)

    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)


if __name__ == '__main__':
    main()
