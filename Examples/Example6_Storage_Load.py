# -*- coding: utf-8 -*-
import OpenDHW
import OpenDHW_Utilities as Utils
from pathlib import Path
from datetime import datetime

"""

"""

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-07'

save_name = 'Storage_Load_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
dir_output = Path.cwd().parent / "Saved_Timeseries" / save_name


def main():
    # Load time-series from DHWcalc
    timeseries_df = OpenDHW.import_from_dhwcalc(s_step=600, categories=1,
                                                mean_drawoff_vol_per_day=200)

    # Compute Heat from Water TimeSeries
    timeseries_df = OpenDHW.compute_heat(timeseries_df=timeseries_df,
                                         temp_dT=35)

    timeseries_df = Utils.convert_dhw_load_to_storage_load(
        timeseries_df=timeseries_df,
        start_plot=start_plot,
        end_plot=end_plot,
        plot_cum_demand=True,
    )


if __name__ == '__main__':
    main()
