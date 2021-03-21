# OpenDHWcalc
Small script to generate Domestic Hot Water Profiles for Households. Inspired by the DHWcalc programme from Uni Kassel.

Created for [RWTH Aachen University, E.ON Energy
Research Center, Institute for Energy Efficient Buildings and Indoor
Climate](https://www.ebc.eonerc.rwth-aachen.de/cms/~dmzz/E-ON-ERC-EBC/?lidx=1).


### Usage

To simplify the Usage of [OpenDHW](https://github.com/jonasgrs/OpenDHW/blob/main/OpenDHW.py), a few [Examples](https://github.com/jonasgrs/OpenDHW/tree/main/Examples) are given.

You can create a DHW profile simply by calling the "generate_dhw_profile" function:

```Python
import OpenDHW

water = OpenDHW.generate_dhw_profile(s_step=s_step)

```

You can also load a DHW profile from DHWcalc, as long as it is stored in [DHWcalc_Files](https://github.com/jonasgrs/OpenDHW/tree/main/DHWcalc_Files)

```Python

water = OpenDHW.import_from_dhwcalc(s_step=s_step, categories=1)

```


You can compare the OpenDHW Generator with the DHWcalc file by calling the "compare_generators" function:

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
