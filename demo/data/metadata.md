# Descriptions

date_created='2023-07-31';\
date_modified='2023-07-31';\
title='Dataset for Ocean Carbon Regimes Detection';\
creator_name='Sweety Mohanty';\
creator_url='smohanty@geomar.de';\
creator_orcid='https://orcid.org/0009-0004-2733-290X';\
institution='GEOMAR Helmholtz Centre for Ocean Research';\
institution_id='https://ror.org/02h2x0161';\
research_division='Ocean Circulation and climate Dynamics';\
research_unit='Ocean Dynamics';\
project='Neural Network Reveals Ocean Carbon Uptake';
<!-- creator_name=$pi;\
creator_email=$pi_email;\
creator_url=$pi_url;\ -->
license='CC BY 4.0  (https://creativecommons.org/licenses/by/4.0/deed.en)';\
naming_authority='de.geomar';\
publisher_name='GEOMAR Helmholtz Centre for Ocean Research Kiel';\
publisher_email='datamanagement@geomar.de';\
handle='https://hdl.handle.net/20.500.12085/b520400a-6629-436f-9041-fe3c47308409';\
references='Mohanty, S., Kazempour, D., Patara, L., Kröger, P. (2023). Interactive Detection and Visualization of Ocean Carbon Regimes, ACM Digital Library';\
keywords='OCEAN CARBON UPTAKE, MULTIVARIATE LINEAR REGRESSION, HIERARCHICAL CLUSTERING';
<!-- # search here: https://gcmd.earthdata.nasa.gov/KeywordViewer/ -->

<!-- # for example:
# keywords: OCEAN GENERAL CIRCULATION MODELS/(OGCM), REGIONAL OCEAN MODELS, CLIMATE INDICATORS, ATMOSPHERIC/OCEAN INDICATORS, ATLANTIC OCEAN, WATER TEMPERATURE, keywords_vocabulary='GCMD:GCMD Keywords'; -->
<!-- contributor_name='';   # if anyone -->
<!-- contributor_email='';   # if anyone -->
<!-- contributor_url='';      # if anyone -->
acknowledgment='This project is funded by Helmholtz Association (grant HIDSS-0005) and North German Supercomputing Alliance (HLRN).';
<!-- summary='';           # -->
<!-- comment='';           # -->

<!-- #### probable not applicable for your data, relevant for NetCDF in the THREDDS context -->
<!-- #processing_level='Level 4 (numerical simulation output)';
#cdm_data_type='Grid';
#realm='ocean seaIce atmos land'; -->
<!-- ##################################################################### -->

----
# Read the data files
- Download and install Python 3.8.0
https://www.python.org/downloads/release/python-380/ 

- Install below requirements
`pip install pandas numpy matplotlib`

- Load the data in a tabular form
```
import pandas as pd
table = pd.read_pickle("path/to/file/filename.pkl")
print(table)
```
- Data has been utilized in below projects: \   
https://github.com/swemoh/Carbon-Regime-Detection-Tool \
https://github.com/swemoh/Detecting-and-tracking-Ocean-Carbon-Regimes 