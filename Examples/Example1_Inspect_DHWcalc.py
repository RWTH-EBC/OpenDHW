# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example load a single TimeSeries from DHWcalc and generates a Histogram 
and a Lineplot from it. This is tested for two timestep widths: 60 and 600 
seconds (1 min and 10 mins)
"""

# ------- Parameter Section ---------
s_step = [60, 600]
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    # Load time-series from DHWcalc
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=s_step[0], categories=1,
                                                mean_drawoff_vol_per_day=200,
                                                daylight_saving=False)

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot(profile_df=timeseries_df)

    # Generate Lineplot from the loaded timeseries
    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)

    # Load time-series from DHWcalc
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=s_step[1], categories=1,
                                                mean_drawoff_vol_per_day=200,
                                                daylight_saving=False)

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot(profile_df=timeseries_df)

    # Generate Lineplot from the loaded timeseries
    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)


if __name__ == '__main__':
    main()
