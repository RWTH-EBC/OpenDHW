# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example load a single TimeSeries from DHWcalc and generates a Histogram 
and a Lineplot from it. Then computes the corresponding Heat TimeSeries and 
also plots it.
"""

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():
    # Load time-series from DHWcalc
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=60, categories=1,
                                                mean_drawoff_vol_per_day=200)

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot(profile_df=timeseries_df)

    # Generate Lineplot from the loaded timeseries
    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot)

    # Compute Heat from Water TimeSeries
    heat_dhwcalc = OpenDHW.compute_heat(timeseries_df=timeseries_df,
                                        temp_dT=35)

    # Generate Lineplot for the heat timeseries
    OpenDHW.draw_lineplot(timeseries_df=heat_dhwcalc, plot_var='heat',
                          start_plot=start_plot, end_plot=end_plot)


if __name__ == '__main__':
    main()
