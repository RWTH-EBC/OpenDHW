# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example generates multiple TimeSeries at once and plots them.
"""

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    # Load time-series
    timeseries_df = OpenDHW.generate_dhw_profile(s_step=s_step)

    # add 4 runs, so that in total, 5 runs are generated
    timeseries_df = OpenDHW.add_additional_runs(timeseries_df=timeseries_df,
                                                total_runs=5)

    # plot these 5 runs
    OpenDHW.plot_multiple_runs(timeseries_df=timeseries_df,
                               plot_demands_overlay=True,
                               start_plot=start_plot, end_plot=end_plot,
                               plot_kde=True, plot_hist=True)


if __name__ == '__main__':
    main()
