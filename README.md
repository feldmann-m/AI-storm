# AI-storm
Evaluation of AI-models for convective parameters

Benchmark datasets: 

    - Weatherbench (ERA5 and TIGGE for 2020) -> limited levels 
    https://weatherbench2.readthedocs.io/en/latest/data-guide.html 

    - Observations -> soundings from University of Wyoming 

Convective parameters: 

    - Instability -> lifted index (surface and 500 hPa levels required), CAPE 

    - Shear -> surface and 500 hPa 

    https://github.com/traupach/xarray_parcel; xcape 

Target: 

    Severe environments as a 0-1 binary

    Values of instability and shear 

    Investigation of bias? 

    Global vs regional -> issues over ocean areas, relevant areas for convection limited 

    Evaluation of convective season NH -> Europe or US 

    Year 2020 

 

Notes: 

    No surface humidity, but pressure levels up to 1000 hPa;

    What do the models do, when 1000 hPa is below ground? Fields do not have holes -> look up what ERA5 does

    FCN-V2 outside of 0-100% RH bounds by ~20% 
