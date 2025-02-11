# -*- coding: utf-8 -*-
import OpenDHW

"""
Resampling a base timestep of 60S yields different distributions compared to 
generating a distribution with different initial timesteps. (why?)
"""

# --- Parameters ---
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "School", "OB", "Grocery_store"
s_steps = [60, 360, 600, 900]
mean_drawoff_vol_per_day = 40
categories = 4
holidays = OpenDHW.get_holidays(country_code = "DE", year = 2015, state = "NW") # Get the holiday data for the specified country, state and year.
occupancy = 5


def main():

    # generate base time-series with OpenDHW
    timeseries_df_base = OpenDHW.generate_dhw_profile(
        s_step=60,
        categories=categories,
        occupancy=occupancy,
        building_type=building_type,
        weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
        holidays=holidays,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
    )

    for s_step in s_steps:

        # either take the base timeseries and resample it
        timeseries_df_resampled = OpenDHW.resample_water_series(
            timeseries_df=timeseries_df_base, s_step_output=s_step)

        # or make a new timeseries with a different s_step
        timeseries_df = OpenDHW.generate_dhw_profile(
            s_step=s_step,
            categories=categories,
            occupancy=occupancy,
            building_type=building_type,
            weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
            holidays=holidays,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

        # import from DHWcalc
        timeseries_df_dhwcalc = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            occupancy=occupancy,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False)

        # compare the resampled series with the non-resampled one.
        OpenDHW.plot_three_histplots(
            timeseries_df_1=timeseries_df_resampled,
            timeseries_df_2=timeseries_df,
            timeseries_df_3=timeseries_df_dhwcalc
        )


if __name__ == '__main__':
    main()
