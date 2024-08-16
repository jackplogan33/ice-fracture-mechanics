# Derived Yeild Stresses of Antarctic Glaciers in Cloud-Native Framework
The first notebook of this repository is modelled after Emma Marshall's github found [here](https://e-marshall.github.io/itslive/intro.html#). I have written functions that will open ITS_LIVE data cubes from a given shapefile at monthly timesteps. To fully understand the workings of these functions, how to work with large scale ITS_LIVE velocities, and other data from the AWS buckets, I recommend following her tutorial. 

## Project Summary
This project is a cloud-native framework that can derive the principle stresses of glacial ice for any region serviced by ITS_LIVE velocity data. The project was developed entirely in CryoCloud, a cloud computing evironment. 

The example in this repository demonstrates the calculation of Principle Stresses on the Shirase Glacier, Antarctica. This is a blue ice glacier, meaning there is no snow or firn coverage. This provides the unique opportunity to observe stress exerted directly on glacial ice. 

Using this framework, we can load in time-varying velocity data, derive strain rates, and compute the principle stresses following the process described by Cuffey and Paterson, 2010. This portion of the workflow is reproducible for any location covered by ITS_LIVE's AWS buckets.

The final portion of this framework combines monthly fracture composites produced by Trystan Surawy-Stepney, 2023. This portion of the workflow can be reproduced for any portion of Antarctica with sufficient Sentinel-1 SAR Backscatter imagery. 

## How to use this Repository
This repository goes increasingly in depth, where the first two notebooks describe the process of loading in the data and computing stresses, both of which are written in as functions and can be called in with one line.

Notebooks Order:
- 01-access-itslive-data.ipynb
    - Walks through some of the process behind `bit.get_data_cube`
- 02-derive-strain-stress.ipynb
    - Walks through the derivation of strain rates, rotation of strain rates along flow and calculating the principle stresses
    - Introduces the function `bit.compute_strain_stress` with an example
- 03-monthly-stress-analysis.ipynb
    - Walks through light analysis of Principle and Von Mises stresses on monthly timesteps
    - Introduces seasonality utilizing Xarray's built in group by functions
- 04-fracture-map-analysis.ipynb
    - Introduces Trystan Surawy-Stepney's monthly fracture composites
    - Shows tracking of crevasse features as they move through space and time.

## Summary of Results
Cloud-native framework that efficiently derives high resolution stress from velocity on a monthly timescale from January, 2015 - January 2023 anywhere in the world. When paired with the fracture composites of continental Antarctica, stresses and fracture data can be compared at a monthly timestep from July 2018 - January 2023.

## Future Directions
This framework can be used to evaluate the yield stress of glacial ice, an important metric for climate models. The data provided here is high resolution and time-varying, which could provide valuable insights for the fracture mechanics and yeild stress of glacial ice. 

## References
- Alley, K. E., Scambos, T. A., Anderson, R. S., Rajaram, H., Pope, A., & Haran, T. M. (2018). Continent-wide estimates of Antarctic strain rates from Landsat 8-derived velocity grids. Journal of Glaciology, 64(244), 321–332. https://doi.org/10.1017/jog.2018.23
- Grinsted, A., Nicholas Mossor Rathmann, Mottram, R., Anne Munck Solgaard, Mathiesen, J., & Christine Schøtt Hvidberg. (2024). Failure strength of glacier ice inferred from Greenland crevasses. the Cryosphere, 18(4), 1947–1957. https://doi.org/10.5194/tc-18-1947-2024
- Cuffey, K., & Paterson, W. S. B. (2010). The physics of glaciers (4th ed.). Burlington, MA: Butterworth-Heinemann/Elsevier. ISBN 978-0-12-369461-4.
- Marshall, E., Henderson, S., & Cherian, D. (2022). e-marshall/itslive: itslive tutorial updates (v.1.0.0). Zenodo. https://doi.org/10.5281/zenodo.6804325
- Gardner, A. S., M. A. Fahnestock, and T. A. Scambos, 2024: MEaSUREs ITS_LIVE Landsat Image-Pair Glacier and Ice Sheet Surface Velocities: Version 1. Data archived at National Snow and Ice Data Center. https://doi.org/10.5067/IMR9D3PEI28U
- Snow, T., Millstein, J., Scheick, J., Sauthoff, W., Leong, W. J., Colliander, J., Pérez, F., James Munroe, Felikson, D., Sutterley, T., & Siegfried, M. (2023). CryoCloud JupyterBook (2023.01.26). Zenodo. https://doi.org/10.5281/zenodo.7576602
- Surawy-Stepney, T., & Hogg, A. E. (2023). Data accompanying the article "Mapping Antarctic Crevasses and their Evolution with Deep Learning Applied to Satellite Radar Imagery" (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8296992
- Vaughan, D. G. (1993). Relating the occurrence of crevasses to surface strain rates. Journal of Glaciology, 39(132), 255–266. https://doi.org/10.3189/S0022143000015926