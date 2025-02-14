# Code repository to generate global ARCO vegetation indices from Sentinel-2

### Setup:
- conda env create -f env.yaml
- conda activate spectral

### Run code: 
- change configuration file to adapt export options.  
	- config.json
- add credentials for GEE service account to auth/
	- alternatively, uncomment the relevant lines and use standard export settings. 
- python code/srcSpectral.py
