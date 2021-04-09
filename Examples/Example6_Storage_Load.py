# -*- coding: utf-8 -*-
import OpenDHW
import OpenDHW_Utilities as Utils
from pathlib import Path
from datetime import datetime

"""
This Example loads a timeseris from DHWcalc, computes the corresponding Heat 
and converts the DHW Timeseries to a Storage Timeseries using the StorageLoad
function of the Utilities. Does the same for an OpenDHW Timeseries. The two 
can then be compared.
"""

# --- Parameter Section ---
s_step = 600
start_plot = '2019-03-04'
end_plot = '2019-03-05'
people = 5
mean_drawoff_vol_per_day = people * 40


dir_output = Path.cwd().parent / "Saved_Timeseries"


def main():
    # Load time-series from DHWcalc
    timeseries_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step,
        categories=1,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day
    )

    # Compute Heat from Water TimeSeries
    timeseries_df = OpenDHW.compute_heat(
        timeseries_df=timeseries_df,
        temp_dT=35
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
    timeseries_df_gauss = OpenDHW.generate_dhw_profile(
        s_step=s_step,
        weekend_weekday_factor=1.2,
        mean_vol_per_drawoff=8,
        mean_drawoff_vol_per_day=200,
        drawoff_method='gauss_combined',
        initial_day=0,
    )

    # Compute Heat from Water TimeSeries
    timeseries_df_gauss = OpenDHW.compute_heat(
        timeseries_df=timeseries_df_gauss,
        temp_dT=35
    )

    timeseries_df_gauss = Utils.convert_dhw_load_to_storage_load(
        timeseries_df=timeseries_df_gauss,
        start_plot=start_plot,
        end_plot=end_plot,
        dir_output=dir_output,
        plot_cum_demand=True,
        save_fig=False
    )

    pass


if __name__ == '__main__':
    main()
