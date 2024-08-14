# Derived Yeild Stresses of Antarctic Glaciers in Cloud-Native Framework


The accessing cloud hosted data portion of this workflow is modelled after Emma Marshall's github found [here](https://e-marshall.github.io/itslive/intro.html#). For introductory data acquisition and plotting of velocities, I highly recommend follwing her workflow. It will give a much more in depth look at data cubes and chunking with dask that is looked over here.

I have written functions that will open ITS_LIVE datacubes from a given shapefile at monthly timesteps. To fully understand the workings of these functions, how to work with large scale ITS_LIVE velocities, and other data from the AWS buckets, I recommend following her tutorial. 

## Project Summary

This project is a cloud-native framework that can derive the principle stresses of glacial ice for any region serviced by ITS_LIVE velocity data.

The example framework demonstrates the calculation of Principle Stresses on the Shirase Glacier, Antarctica. This is a blue ice glacier, meaning there is no snow or firn coverage. 

## How to use this Repository

This repository goes increasingly in depth, where the first two notebooks describe the process of loading in the data and computing stresses, both of which are written in as functions and can be called in with one line.

Notebooks Order:
- access-itslive-data.ipynb
    - Walks through some of the process behind `bit.get_data_cube`
- derive-strain-rates-stress.ipynb
    - Walks through the derivation of strain rates, rotation fo strain rates along flow and calculating the principle stresses
    - Introduces and shows an example of calling `bit.compute_strain_stress`
- monthly-stress-analysis.ipynb
    - Walks through light analysis of Principle and Von Mises stresses on a monthly timescale
    - Introduces seasonality utilizing Xarray's built in group by functions
- fracture-map-analysis.ipynb
    - Introduces Trystan Surawy-Stepny's monthly fracture composites
    - Shows tracking of crevasse features as they move through space and time.

## Background
Little to no in depth stress analysis on a time-varying scale. Many frameworks are on a singular timescale

## Summary of Results
Cloud-native framework that efficiently derives high resolution stress from velocity on a monthly timescale from January, 2015 - January 2023.

## Future Directions
This framework can be used to evaluate the yield stress of glacial ice, an important metric for climate models. 

## References
- Jupyter
- ITS_LIVE
- Emma Marshall
- Trystan Surawy-Stepny
- Cuffey & Patterson 2010
- Vaughan 1993
- Grinsted 2018
- Alley et. al. 2018