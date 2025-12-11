# *Caravan-Qual*: A global scale integration of water quality observations into a large-sample hydrology dataset

*Caravan-Qual* is an open access dataset that brings water quality to the research paradigm of large sample hydrology (LSH), integrating water quality data from 100 constituents with catchment attributes, meterological forcing and streamflow observations. The dataset covers the time period 1980 - 2025 and currently contains ~70 million river water quality observations from across 137,373 monitoring stations, in addition to streamflow data from 25,839 gauge stations. 

A manuscript associated with the dataset is currently under submission at Scientific Data [ADD LINK TO PREPRINT].

## About this repository
This repository contains the code recreating or extending the *Caravan-Qual* dataset.

This is split across three folders:
1. Caravan/: scripts for downloading *Caravan* from source, and include subsquent extensions. 
2. add_data/: scripts for adding water quality data, deriving upstream catchment polygons and linking to streamflow gauges.  
3. Caravan-Qual/: scripts for processing water quality and streamflow data to create the *Caravan-Qual* dataset, both as .csv's and in .zarr format.

## *Caravan*

*Caravan-Qual* adds water quality observations to the existing *Caravan* [dataset](https://zenodo.org/records/15529786). The original Caravan paper can be accessed [here](https://www.nature.com/articles/s41597-023-01975-w), while developments and community extensions are documented on Caravan's [GitHub](https://github.com/kratzert/Caravan/tree/main) page.

Meteorological data and catchment attributes for water quality monitoring stations are derived using the approach developed for *Caravan*, which are detailed [here](https://github.com/kratzert/Caravan/tree/main/code). 

## Contact
***Caravan-Qual***: Edward R. Jones (e.r.jones@uu.nl)

***Caravan***: Frederik Kratzert (kratzert@google.com)
