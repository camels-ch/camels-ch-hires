from forcing import Forcing

# Define paths to data
HYDRO_ENTITIES_SHAPEFILE = '/path/to/hydro_entities.shp'
FORCING_DATA_DIR = '/path/to/forcing/data/'
OUTPUT_DIR = '/path/to/output/'

def main():
    forcing = Forcing(HYDRO_ENTITIES_SHAPEFILE)
    forcing.extract_from_gridded_data(path=FORCING_DATA_DIR)
    forcing.save_as(OUTPUT_DIR + 'forcing_data.nc')
    print(f"Forcing data saved to {OUTPUT_DIR + 'forcing_data.nc'}")
