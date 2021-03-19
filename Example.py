# -*- coding: utf-8 -*-
import OpenDHW

# ------- Parameter Section ---------
s_step = 60
start_plot = '2019-03-04'
end_plot = '2019-03-08'


def main():

    # generate time-series with OpenDHW
    x, water_open_dhw_60 = OpenDHW.generate_dhw_profile_open_dhw(
        s_step=s_step,
        weekend_weekday_factor=1.2,
        mean_vol_per_drawoff=8,
        mean_drawoff_vol_per_day=200,
        drawoff_method='beta',
        initial_day=0,
    )

    # Load time-series from DHWcalc
    x, water_dhwcalc_60 = OpenDHW.import_from_dhwcalc(s_step=s_step)

    # compare the two time-series for a specific time-period
    OpenDHW.compare_generators(
        first_method='DHWcalc',
        first_series_LperH=water_dhwcalc_60,
        second_method='OpenDHW',
        second_series_LperH=water_open_dhw_60,
        s_step=s_step,
        start_plot=start_plot,
        end_plot=end_plot,
    )


if __name__ == '__main__':
    main()
