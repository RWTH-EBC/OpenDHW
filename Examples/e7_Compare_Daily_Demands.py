# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example loads DHWcalc Timeseries with different daily water 
demands and compares them to their OpenDHW Equivalent.
"""

# --- Parameters ---
resample_method = False
start_plot = '2019-04-05'
end_plot = '2019-04-07'
plot_date_slice = False
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "School", "OB", "Grocery_store"

# --- Constants ---
daily_demands = [32, 40, 48, 400]  # 32, 40, 45, 400
s_step = 900
categories = 4
occupancy = 5 # Number of occupants in the building
holidays = OpenDHW.get_holidays(country_code = "DE", year = 2015, state = "NW") # Get the holiday data for the specified country, state and year.

def main():

    for daily_demand in daily_demands:

        # Load time-series from DHWcalc
        dhwcalc_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            occupancy=occupancy,
            categories=categories,
            mean_drawoff_vol_per_day=daily_demand,
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
                mean_drawoff_vol_per_day=daily_demand,
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
                mean_drawoff_vol_per_day=daily_demand,
            )

            # resample to the desired stepwidth
            open_dhw_df = OpenDHW.resample_water_series(
                timeseries_df=open_dhw_df, s_step_output=s_step)

        # compare both time-series
        OpenDHW.compare_generators(
            timeseries_df_1=dhwcalc_df,
            timeseries_df_2=open_dhw_df,
            start_plot=start_plot,
            end_plot=end_plot,
            plot_date_slice=plot_date_slice,
            plot_detailed_distribution=True
        )


if __name__ == '__main__':
    main()
