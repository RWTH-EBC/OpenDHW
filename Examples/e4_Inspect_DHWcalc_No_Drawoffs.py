# -*- coding: utf-8 -*-
import OpenDHW

"""
This example shows that the table in DHWcalc when choosing 4 categories can 
be misleading, as multiple drawoffs can happen at the same timestep.

Consequently, the row "mean no. of drawoffs per day" is not equal to the 
total number of yearly unique drawoffs generated by the programme

For 40L, DHWcalc assumes a total of 18768 drawoffs per year:

                                        cat 1   cat2    cat3    cat4    sum
    drawoff duration [min]              1       1       10      5                                        
    mean no. drawoffs per day [-]       28      12      0.142   2
    mean no. drawoffs per year [-]      10220   4380    51.8    730     15848
    mean no. drawoffs * duration [-]    10220   4380    518     3650    18768 
      
"""

# --- Constants ---
categories = 4
s_step = 60
mean_drawoff_vol_per_day = 40  # L
occupancy = 5


def main():

    # 4 categories
    timeseries_df = OpenDHW.import_from_dhwcalc(
        s_step=s_step,
        occupancy=occupancy,
        categories=categories,
        mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        daylight_saving=False)

    drawoffs_df = timeseries_df[timeseries_df['Water_LperH'] != 0][
        'Water_LperH']

    print("The mean no. of drawoffs per year is {}".format(len(drawoffs_df)))


if __name__ == '__main__':
    main()
