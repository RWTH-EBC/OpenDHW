# -*- coding: utf-8 -*-
import OpenDHW
from OpenDHW.utils import OpenDHW_Utilities as Utils
from pathlib import Path

"""
This Example loads a timeseris from DHWcalc, computes the corresponding Heat 
and converts the DHW Timeseries to a Storage Timeseries using the StorageLoad
function of the Utilities Script. Does the same for an OpenDHW Timeseries. The 
two can then be compared.
"""

# --- Parameters ---
start_plot = '2019-03-04'
end_plot = '2019-03-05'
dir_output = Path.cwd().parent / "Saved_Timeseries"
building_type = "SFH"  # "SFH", "TH", "MFH", "AB", "SC", "OB", "GS", "RE"

# --- constants ---
s_step = 600
categories = 1
mean_drawoff_vol_per_day = 40 # Mean daily water consumption per person in liters
holidays = OpenDHW.get_holidays(country_code = "DE", year = 2019) # Get the holiday data for the specified country, state and year.
occupancy = 5
temp_dT = 35    # K


def main():

    # Load time-series from DHWcalc
    timeseries_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step,
        categories=categories,
        occupancy=occupancy,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        daylight_saving=False
    )

    # Compute Heat from Water TimeSeries
    timeseries_df = OpenDHW.compute_heat(
        timeseries_df=timeseries_df,
        temp_dT=temp_dT
    )

    timeseries_df = Utils.convert_dhw_load_to_storage_load(
        timeseries_df=timeseries_df,
        start_plot=start_plot,
        end_plot=end_plot,
        dir_output=dir_output,
        plot_cum_demand=True,
        save_fig=False
    )

    # generate time-series with OpenDHW
    timeseries_df_opendhw = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        categories=categories,
        occupancy=occupancy,
        building_type=building_type,
        weekend_weekday_factor = 1.2 if building_type in {"SFH", "TH", "MFH", "AB"} else 1,
        holidays=holidays,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        initial_day=1  # Tuesday
    )

    # Compute Heat from Water TimeSeries
    timeseries_df_opendhw = OpenDHW.compute_heat(
        timeseries_df=timeseries_df_opendhw,
        temp_dT=temp_dT
    )

    timeseries_df_opendhw = Utils.convert_dhw_load_to_storage_load(
        timeseries_df=timeseries_df_opendhw,
        start_plot=start_plot,
        end_plot=end_plot,
        dir_output=dir_output,
        plot_cum_demand=True,
        save_fig=False
    )


if __name__ == '__main__':
    main()
