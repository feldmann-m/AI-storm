# AI-storm
Evaluation of AI-models for convective parameters

Benchmark datasets: 

    - Weatherbench (ERA5 and TIGGE for 2020) -> limited levels 
    https://weatherbench2.readthedocs.io/en/latest/data-guide.html 

    - Observations -> soundings from University of Wyoming 

Convective parameters: 

    - Instability -> CAPE 

    - Shear -> surface and 500 hPa 

    wrf-python

Target: 

    Severe environments as a 0-1 binary -> fractional skill score (FSS)

    Values of instability and shear -> RMSE, BIAS, SAL-score

    Investigation of bias -> role of moisture

    Evaluation of convective season NH&SH -> North America, Europe, Australia, Argentina

    Year 2020 

 

Notes: 

    No surface humidity, but pressure levels up to 1000 hPa;

    FCN-V2 outside of 0-100% RH bounds by ~20% 

    FCN-V2 issues with humidity overall
