import pandas
import geopandas
import numpy as np

def get_landuse_and_zones(parameter):
    """
    Function loads the zones.geojson and landuse.csv files. 
    We also add 'car_ownership' information to the landuse. 
    
    RETURN 
        landuse: Dataframe
        zones  : Dataframe 
    """
    #### Loading Zone Data ####
    zones = geopandas.read_file("zones.geojson")
       
    ####to create a column with centroids ####
    centroidFunction = lambda row: (row['geometry'].centroid.y, row['geometry'].centroid.x)
    zones['centroid'] = zones.apply(centroidFunction, axis=1)
    
    #### Loading Landuse Data ####
    landuse = pandas.read_csv("landuse.csv", sep=";")
    landuse['car_ownership'] = 0.75
    
    
    # Here you have to calculate the car ownership probability given the parameters

  
    
    # Add zone information to landuse data
    zones.set_index('area', inplace=True)
    landuse = landuse.join(zones, on='area')
    
    return landuse, zones   

