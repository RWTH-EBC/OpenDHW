# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example generates multiple TimeSeries at once and saves them as a CSV.
"""

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    timeseries_df = OpenDHW.generate_dhw_profile(s_step=s_step)

    timeseries_df = OpenDHW.add_additional_runs(
        timeseries_df=timeseries_df, total_runs=50, save_to_csv=True)


if __name__ == '__main__':
    main()
