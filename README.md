# *Caravan-Qual*: A global scale integration of water quality observations into a large-sample hydrology dataset

*Caravan-Qual* is an open access dataset that brings water quality to the research paradigm of large sample hydrology (LSH), integrating water quality data from 100 constituents with catchment attributes, meteorological forcing and streamflow observations. The dataset covers the time period 1980 - 2025 and currently contains ~70 million river water quality observations from across 137,373 monitoring stations, in addition to streamflow data from 25,839 gauge stations. 

The full dataset, including data necessary for recreating or extending *Caravan-Qual*, can be accessed [here](https://doi.org/10.24416/UU01-S8QW8O).

A lightweight version of the dataset is also avaliable, hosted on [Zenodo](https://doi.org/10.5281/zenodo.17787066). This version contains monthly (instead of daily) weather data to conform with Zenodo's data storage policies. 

A manuscript associated with the dataset is currently under submission at Scientific Data. A preprint is available at: https://doi.org/10.31223/X54J39.


## About this repository
This repository contains the code recreating or extending the *Caravan-Qual* dataset.

This code is split across three folders:
* Caravan/: scripts for downloading *Caravan* from source, and include subsquent extensions. 
* add_data/: scripts for adding water quality data, deriving upstream catchment polygons and linking to streamflow gauges.  
* Caravan-Qual/: scripts for processing water quality and streamflow data to create the *Caravan-Qual* dataset, both as .csv's and in .zarr format.

A full list of variables included in *Caravan-Qual* is also provided in this repository (Caravan-Qual_zarr_variables.csv)

## *Caravan*
*Caravan-Qual* adds water quality observations to the existing *Caravan* [dataset](https://zenodo.org/records/15529786). The original Caravan paper can be accessed [here](https://www.nature.com/articles/s41597-023-01975-w), while developments and community extensions are documented on Caravan's [GitHub](https://github.com/kratzert/Caravan/tree/main) page.

Meteorological data and catchment attributes for water quality monitoring stations are derived using the approach developed for *Caravan*, which are available [here](https://github.com/kratzert/Caravan/tree/main/code). 


## Contact
***Caravan-Qual***: Edward R. Jones (e.r.jones@uu.nl)

***Caravan***: Frederik Kratzert (kratzert@google.com)
