# -*- coding: utf-8 -*-
import OpenDHW

"""
This example shows that the table in DHWcalc when choosing 4 categories is 
not really true.

More specifically, the row "mean no. of drawoffs per day" seems to be 
incorrect, as during the distribution of the drawoffs, some are cut of and 
rearranged. Thus, the total number of yearly drawoffs is not equal to the 
number implied by the table.

For 200L, DHWcalc assumes a total of 15848 drawoffs per year:

                                        cat 1   cat2    cat3    cat4    sum
    drawoff duration [min]              1       1       10      5                                        
    mean no. drawoffs per day [-]       28      12      0.142   2
    mean no. drawoffs per year [-]      10220   4380    51.8    730     15848
    mean no. entries in list [-]        10220   4380    518     3650    18768   
"""

# -- Parameter Section --
runs = 5    # actually only one run needed as DHWcalc has no random seeds.

# --- Constants ---
categories = 4
s_step = 60
mean_drawoff_vol_per_day = 200  # L


def main():

    no_drawoffs_runs = []

    for run in range(runs):

        # 4 categories
        timeseries_4cat_df = OpenDHW.import_from_dhwcalc(
            s_step=s_step,
            categories=categories,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
            daylight_saving=False)

        drawoffs_df = OpenDHW.get_drawoffs(timeseries_4cat_df)

        no_drawoffs_runs.append(len(drawoffs_df))

    drawoffs_per_year = sum(no_drawoffs_runs)/len(no_drawoffs_runs)

    print("The mean no. of drawoffs per year after {} runs is {}".format(
        runs, drawoffs_per_year))


if __name__ == '__main__':
    main()
