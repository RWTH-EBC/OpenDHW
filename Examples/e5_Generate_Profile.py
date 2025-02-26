# -*- coding: utf-8 -*-
import OpenDHW

"""
This Example generates an OpenDHW Timeseries.

Like DHWcalc, OpenDHW need a parametrisation. The usual parametrisation for 
households is 40 L/(person*day). This value is set 
as 'mean_drawoff_vol_per_day'. For example, a single-family house 
(SFH) with 5 occupants would have a total mean daily water consumption of 
200 L/day (5 people * 40 L/person/day).

For a multi-family house (MFH) with 10 flats and 5 people per flat, the mean 
drawoff volume is 2000 L/day (10 flats * 5 people/flat * 40 L/person/day).

For non-residental buildings, the daily probability functions are based on 
the "people profile" described in the SIA Standard-Nutzungsbedingungen
für die Energie- und Gebäudetechnik (Merkblatt 2024).

As an alternative for timesteps different from 60seconds, a second method has 
been implemented in OpenDHW: the 'resample_water_series' function.

The idea is to always generate a timeseries with s_step=60s and then resample it
afterwards to the desired output stepwidth. The disadvantage is a higher 
computing time.
"""

# --- Building Type Description ---
"""
Building types supported by the script:

Residential buildings:
    - SFH: Single Family House
    - TH: Terraced House
    - MFH: Multi-Family House
    - AB: Apartment Block

Non-residential buildings:
    - SC: School 
    - OB: Office building
    - GS: Grocery store
"""

# --- Parameters ---
resample_method = False
start_plot = '2019-03-31-06'
end_plot = '2019-03-31-09'
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "SC", "OB", "GS"

# --- Constants ---
holidays = OpenDHW.get_holidays(country_code = "DE", year = 2015) # Get the holiday data for the specified country, state and year.
s_step = 60
categories = 1
occupancy = 5
temp_dT = 35    # K
mean_drawoff_vol_per_day = 40 # Mean daily water consumption per person in liters

def main():

    if not resample_method:
        # generate time-series with OpenDHW
        timeseries_df = OpenDHW.generate_dhw_profile(
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

    # Compute Heat from Water TimeSeries
    timeseries_df = OpenDHW.compute_heat(
        timeseries_df=timeseries_df,
        temp_dT=temp_dT
    )

    # example timeseries which could be fed into Dynamic Simulations
    water_series = timeseries_df['Water_L']
    heat_series = timeseries_df['Heat_kWh']

    # inspect the drawoff events
    drawoffs_df = timeseries_df[timeseries_df['Water_LperH'] != 0]

    # Generate Histogram from the loaded timeseries
    OpenDHW.draw_histplot(timeseries_df=timeseries_df, extra_kde=False,
                          save_fig=True)

    # Generate Lineplot from the loaded timeseries
    OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                          end_plot=end_plot, save_fig=True)

    # Generate detailed Histogram from the loaded timeseries
    OpenDHW.draw_detailed_histplot(timeseries_df=timeseries_df)


if __name__ == '__main__':
    main()
