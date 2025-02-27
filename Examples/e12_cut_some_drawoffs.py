# -*- coding: utf-8 -*-
import OpenDHW

"""
Introduces the 'reduce_no_drawoffs' function.
"""


# --- Parameters ---
resample_method = False
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "SC", "OB", "GS", "RE"
start_plot = '2019-03-31'
end_plot = '2019-04-01'

# --- Constants ---
s_step = 900
holidays = OpenDHW.get_holidays(country_code = "DE", year = 2015) # Get the holiday data for the specified country, state and year.
mean_drawoff_vol_per_day = 40
categories = 4
occupancy=5

def main():

    if not resample_method:
        # generate time-series with OpenDHW
        timeseries_df = OpenDHW.generate_dhw_profile(
            s_step=s_step,
            occupancy=occupancy,
            building_type=building_type,
            weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
            categories=categories,
            holidays=holidays,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

    else:

        # generate time-series with OpenDHW
        timeseries_df = OpenDHW.generate_dhw_profile(
            s_step=60,
            categories=categories,
            occupancy=occupancy,
            building_type=building_type,
            weekend_weekday_factor=1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
            holidays=holidays,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

        # resample to the desired stepwidth
        timeseries_df = OpenDHW.resample_water_series(
            timeseries_df=timeseries_df, s_step_output=s_step)

    # Load time-series from DHWcalc
    dhwcalc_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step,
        categories=categories,
        occupancy=occupancy,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        daylight_saving=False
    )

    # reduce some drawoffs
    timeseries_df_reduced = OpenDHW.reduce_no_drawoffs(
        timeseries_df=timeseries_df)

    # Generate Histogram from the loaded timeseries
    OpenDHW.plot_three_histplots(timeseries_df, timeseries_df_reduced,
                                 dhwcalc_df)


if __name__ == '__main__':
    main()
