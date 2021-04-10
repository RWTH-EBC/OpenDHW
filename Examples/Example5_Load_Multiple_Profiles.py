# -*- coding: utf-8 -*-
import OpenDHW
import pandas as pd
import random

"""
This Example loads multiple TimeSeries at once from a CSV file generated in 
Example 4. Then the drawoffs are averaged and distributed over a year.

Thus, the drawoff distribution is averaged, not the placement within a year!
"""

# ------- Parameter Section ---------
save_path = '/Users/jonasgrossmann/git_repos/' \
            'OpenDHW/Saved_Timeseries/OpenDHW_5runs_200L_10min_8LperDrawoff.csv'

start_plot = '2019-01-01'
end_plot = '2019-01-31'


def main():

    # get large run of OpenDHW results from the csv generated in Example 4.
    timeseries_df = pd.read_csv(save_path, index_col=0, parse_dates=True)

    s_step = OpenDHW.get_s_step(timeseries_df=timeseries_df)

    # get the Drawoffs (elements that are not Zeros), and compute average
    drawoffs_df, water_LperH_df = OpenDHW.get_drawoffs(
        timeseries_df=timeseries_df)
    av_drawoffs = drawoffs_df.mean(axis=1)
    # todo: understand what .mean(axis=1) actually does

    # get sums of drawoffs to compare with the average sum.
    sum_av = sum(av_drawoffs)
    sum_1 = sum(drawoffs_df.iloc[:, 0].fillna(0))
    sum_2 = sum(drawoffs_df.iloc[:, 1].fillna(0))
    sum_3 = sum(drawoffs_df.iloc[:, 2].fillna(0))
    sum_4 = sum(drawoffs_df.iloc[:, 3].fillna(0))
    sum_5 = sum(drawoffs_df.iloc[:, 4].fillna(0))

    # distribute that av. list over a year
    p_norm_integral = OpenDHW.generate_yearly_probability_profile(
        s_step=s_step,
        weekend_weekday_factor=1.2,
        initial_day=0
    )

    min_rand = min(p_norm_integral)
    max_rand = max(p_norm_integral)
    p_drawoffs = [random.uniform(min_rand, max_rand) for _ in av_drawoffs]

    timeseries_df_av = OpenDHW.distribute_drawoffs(
        drawoffs=av_drawoffs,
        p_drawoffs=p_drawoffs,
        p_norm_integral=p_norm_integral,
        s_step=s_step
    )

    # copy some constants from the large-study df to the new av-df
    timeseries_df_av['method'] = timeseries_df['method']
    timeseries_df_av['mean_drawoff_vol_per_day'] = timeseries_df[
        'mean_drawoff_vol_per_day']

    # Load time-series from DHWcalc
    timeseries_df_dhwcalc = OpenDHW.import_from_dhwcalc(
        s_step=s_step, categories=1, daylight_saving=False
    )

    # compare two time-series
    OpenDHW.compare_generators(
        timeseries_df_1=timeseries_df_dhwcalc,
        timeseries_df_2=timeseries_df_av,
        start_plot=start_plot,
        end_plot=end_plot,
    )


if __name__ == '__main__':
    main()
