# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example generates multiple TimeSeries at once and plots the resulting 
Dataframe.
"""

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    # generate multiple profiles
    dhw_demands_df, drawoffs_df = OpenDHW.generate_multiple_profiles(
        s_step=s_step, runs=5)

    av_profile = dhw_demands_df['Average']

    # Load time-series from DHWcalc
    water_dhwcalc_60 = OpenDHW.import_from_dhwcalc(s_step=s_step, categories=1)

    # compare two time-series
    OpenDHW.compare_generators(
        first_method='DHWcalc',
        first_series_LperH=water_dhwcalc_60,
        second_method='OpenDHW',
        second_series_LperH=av_profile,
        s_step=s_step,
        start_plot=start_plot,
        end_plot=end_plot,
    )

    # plot them. Basic Implementation, not so much info..
    OpenDHW.plot_multiple_runs(dhw_demands_df=dhw_demands_df,
                               drawoffs_df=drawoffs_df,
                               plot_demands_overlay=True,
                               start_plot=start_plot, end_plot=end_plot,
                               plot_kde=True, plot_hist=True)


if __name__ == '__main__':
    main()
