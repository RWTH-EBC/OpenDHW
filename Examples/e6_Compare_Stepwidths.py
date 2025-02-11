# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example loads DHWcalc Timeseries with different timesteps and compares 
them to their OpenDHW Equivalent.
"""

# --- Parameters ---
resample_method = False
start_plot = '2019-03-04'
end_plot = '2019-03-08'
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "School", "OB", "Grocery_store"

# --- Constants ---
s_steps = [60, 600, 900, 3600]
categories = 1
holidays = OpenDHW.get_holidays(country_code = "DE", year = 2015, state = "NW") # Get the holiday data for the specified country, state and year.
mean_drawoff_vol_per_day = 40 # Mean daily water consumption per person in liters
occupancy = 5


def main():

    for s_step in s_steps:

        # Load time-series from DHWcalc
        dhwcalc_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=categories,
            occupancy=occupancy,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False
        )

        if not resample_method:
            # generate time-series with OpenDHW
            open_dhw_df = OpenDHW.generate_dhw_profile(
                s_step=s_step,
                categories=categories,
                occupancy=occupancy,
                building_type=building_type,
                weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
                holidays=holidays,
                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day
            )

        else:

            # generate time-series with OpenDHW
            open_dhw_df = OpenDHW.generate_dhw_profile(
                s_step=60,
                categories=categories,
                occupancy=occupancy,
                building_type=building_type,
                weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
                holidays=holidays,
                mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            )

            # resample to the desired stepwidth
            open_dhw_df = OpenDHW.resample_water_series(
                timeseries_df=open_dhw_df, s_step_output=s_step)

        # compare time-series
        OpenDHW.compare_generators(
            timeseries_df_1=dhwcalc_df,
            timeseries_df_2=open_dhw_df,
            start_plot=start_plot,
            end_plot=end_plot,
            plot_date_slice=False,
        )


if __name__ == '__main__':
    main()
