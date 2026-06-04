from hydro_snap import recondition_dem

# Recondition the DEM
recondition_dem(
    dem_raster='path/to/DEM_raster.tif',
    streams_shp='path/to/streams.shp',
    output_dir='path/to/output',
    catchment_shp='path/to/catchment.shp',
    stream_orientation='upstream',
)
