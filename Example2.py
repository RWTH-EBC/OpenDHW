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
    water_dhwcalc_60_1 = OpenDHW.import_from_dhwcalc(s_step=60, categories=1)

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot_from_profile(dhw_profile_LperH=water_dhwcalc_60_1,
                                       s_step=60)

    # Generate Lineplot from the loaded timeseries
    OpenDHW.draw_lineplot(method='DHWcalc', s_step=s_step,
                          series=water_dhwcalc_60_1, start_plot=start_plot,
                          end_plot=end_plot)

    # Compute Heat from Water TimeSeries
    heat_dhwcalc = OpenDHW.compute_heat(s_step=s_step,
                                        water_LperH=water_dhwcalc_60_1,
                                        temp_dT=35)

    # Generate Lineplot for the heat timeseries
    OpenDHW.draw_lineplot(method='DHWcalc', s_step=s_step,
                          series=heat_dhwcalc, series_type='heat',
                          start_plot=start_plot, end_plot=end_plot)


if __name__ == '__main__':
    main()
