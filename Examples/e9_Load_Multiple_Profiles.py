# -*- coding: utf-8 -*-
from OpenDHW import OpenDHW as OpenDHW
import pandas as pd
from pathlib import Path

"""
This Example loads multiple TimeSeries at once from a CSV file generated in 
the previous Example.
"""

# --- Parameters ---
start_plot = '2019-01-01'
end_plot = '2019-01-31'

save_name = "OpenDHW_5runs_200L_10min.csv"
save_dir = Path.cwd().parent / "Saved_Timeseries"
save_path = save_dir / save_name


def main():

    # get large run of OpenDHW results from the csv generated in Example 4.
    timeseries_df_study = pd.read_csv(save_path, index_col=0, parse_dates=True)

    # plot the csv
    OpenDHW.plot_multiple_runs(timeseries_df=timeseries_df_study)


if __name__ == '__main__':
    main()
