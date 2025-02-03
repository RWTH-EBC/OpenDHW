# -*- coding: utf-8 -*-
import OpenDHW
from pathlib import Path

"""
This Example generates multiple TimeSeries at once and saves them as a CSV. 
First, a single DHW profile is generated. Then, additional profiles with the 
same settings as the original profile are generated and appended to the main 
dataframe.
"""

# --- Parameters ---
s_step = 600
runs = 5
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "School", "OB", "Grocery_store"

# --- constants ---
mean_drawoff_vol_per_day = 40
categories = 4
dir_output = Path.cwd().parent / "Saved_Timeseries"
occupancy = 5 # Number of occupants in the building
holidays = [1, 93, 96, 121, 134, 145, 155, 275, 305, 358, 359, 360, 365],  # Julian day number of the holidays in NRW in 2015

def main():

    # generate timeseries
    timeseries_df = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        categories=categories,
        occupancy=occupancy,
        building_type=building_type,
        weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
        holidays=holidays,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
    )

    timeseries_df = OpenDHW.add_additional_runs(
        timeseries_df=timeseries_df,
        total_runs=runs,
        occupancy=occupancy,
        building_type=building_type,
        holidays=holidays,
        dir_output=dir_output)


if __name__ == '__main__':
    main()
