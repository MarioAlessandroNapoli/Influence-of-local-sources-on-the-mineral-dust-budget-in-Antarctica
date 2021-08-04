import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import plotly.express as px
import plotly.graph_objects as go

import shapely
import shapely.ops as ops
from shapely.ops import nearest_points
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull as ch
from descartes import PolygonPatch
import pyproj
from functools import partial

pd.options.mode.chained_assignment = None  # default='warn'


def load_data():
    # Loading rock_cropout
    rock_cropout = gpd.read_file("data/rockoutcrop/add_rockoutcrop_landsatWGS84.shp")
    # Loading DEM elevation file, rescale and mean
    df = pd.read_csv("data/DEM/bamber.5km97.dat", sep=' ', header=None,
                     names=['latitude', 'longitude', 'difference', 'elevation'])
    df.loc[:, 'latitude'] = np.around(df.loc[:, 'latitude'], 0)
    df.loc[:, 'longitude'] = np.around(df.loc[:, 'longitude'], 0)
    mean_elev = df.groupby(['latitude', 'longitude']).elevation.mean().reset_index()
    # Loading stations
    stazioni = pd.read_csv('data/stazioni.csv', encoding='utf-8')
    stazioni = gpd.GeoDataFrame(
        stazioni, geometry=gpd.points_from_xy(stazioni.longitude, stazioni.latitude)).set_crs('epsg:4326')
    # Loading geo_units and coastlines
    geo_units = gpd.read_file('GeoUnits/shapefile/geo_units.shp')
    coastline = gpd.read_file('coastline/add_coastline_medium_res_line_v7_4.shp')

    return rock_cropout, mean_elev, stazioni, geo_units, coastline


def find_nearest_value(origin_df, lon, lat, var_name):
    try:
        return origin_df[(origin_df.longitude == int(lon)) & (origin_df.latitude == int(lat))][var_name].values[0]
    except Exception as ex:
        print('Cannot find nearest value, error: ', ex)
        return None


def get_elevation(rock_cropout, mean_elev):
    return pd.Series(rock_cropout.apply(lambda row: find_nearest_value(
        mean_elev, row.geometry.centroid.x, row.geometry.centroid.y, 'elevation'), axis=1))
