# -*- coding: utf-8 -*-
import OpenDHW

"""
Same as example 1, but with 4 categories instead of 1.
"""

# --- Parameter Section ---
s_step = 60
people = 5

# --- Plot Parameters ---
start_plot = '2019-03-31'
end_plot = '2019-04-01'

# --- Constants ---
mean_drawoff_vol_per_day_and_person = 40
mean_drawoff_vol_per_day = mean_drawoff_vol_per_day_and_person * people


def main():

    # generate time-series with OpenDHW
    timeseries_df = OpenDHW.generate_dhw_profile_cats(
        s_step=s_step,
    )

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot(timeseries_df=timeseries_df)

    drawoffs_df = OpenDHW.get_drawoffs(timeseries_df, col_part='Water_LperH')


if __name__ == '__main__':
    main()
