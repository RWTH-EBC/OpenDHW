# OpenDHWcalc
Small script to generate Domestic Hot Water Profiles for Households. Inspired by the DHWcalc programme from Uni Kassel.

Created for [RWTH Aachen University, E.ON Energy
Research Center, Institute for Energy Efficient Buildings and Indoor
Climate](https://www.ebc.eonerc.rwth-aachen.de/cms/~dmzz/E-ON-ERC-EBC/?lidx=1).


### Usage

You can create a DHW profile simply by calling the "generate_dhw_profile_open_dhwcalc" function:

```Python
import OpenDHW

heat, water = OpenDHW.generate_dhw_profile_open_dhwcalc(s_step=s_step)

heat2, water2 = OpenDHW.import_from_dhwcalc(s_step=s_step)

```

You can compare the OpenDHW Generator with DHWcalc files by calling the "compare_generators" function:

```Python

OpenDHW.compare_generators(
    first_method='DHWcalc',
    first_series_LperH=water,
    second_method='OpenDHW',
    second_series_LperH=water2,
    s_step=60,
    start_plot='2019-03-01',
    end_plot='2019-03-05',
)

```
