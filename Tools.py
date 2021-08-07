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
    geo_units = gpd.read_file('data/GeoUnits/shapefile/geo_units.shp')
    unconsolidated_classes = ['Hs', 'Qk', 'Quc', 'Qc', 'Qu', 'Qa', 'Qs']
    topography_classes = ['w', 'water', 'ice', '?', 'unknown']

    unconsolidated = geo_units[geo_units.MAPSYMBOL.isin(unconsolidated_classes)]
    rocks = geo_units[
        ~(geo_units.MAPSYMBOL.isin(unconsolidated_classes)) & ~(geo_units.MAPSYMBOL.isin(topography_classes))]

    del geo_units

    coastline = gpd.read_file('data/coastline/add_coastline_medium_res_line_v7_4.shp')

    return rock_cropout, mean_elev, stazioni, unconsolidated, rocks, coastline


def find_nearest_value(origin_df, lon, lat, var_name):
    try:
        return origin_df[(origin_df.longitude == int(lon)) & (origin_df.latitude == int(lat))][var_name].values[0]
    except Exception as ex:
        print('Cannot find nearest value, error: ', ex)
        return None


def get_elevation(rock_cropout, mean_elev):
    return pd.Series(rock_cropout.apply(lambda row: find_nearest_value(
        mean_elev, row.geometry.centroid.x, row.geometry.centroid.y, 'elevation'), axis=1))


def get_basemap(projection='spstere', boundinglat=-60, lon_0=180, resolution='h'):
    m = Basemap(projection=projection, boundinglat=boundinglat, lon_0=lon_0, resolution=resolution)
    return m


def get_basemap_projection(geometry, basemap=None):
    if not basemap:
        basemap = get_basemap()
    if not geometry.crs:
        "CRS non definito"
        return None
    geometry = geometry.to_crs("EPSG:3031")
    centr_x = geometry.centroid.x
    centr_y = geometry.centroid.y
    centroids = gpd.GeoDataFrame(geometry=gpd.points_from_xy(centr_x, centr_y)) \
        .set_crs('epsg:3031').to_crs('epsg:4326')
    points_x, points_y = basemap(centroids.geometry.x, centroids.geometry.y)
    return pd.DataFrame({'x': points_x, 'y': points_y})


def viz_init():
    m = get_basemap()
    plt.figure(figsize=(15, 15))
    m.drawcoastlines()
    m.fillcontinents(color='white', lake_color='aqua')
    m.drawmapboundary(fill_color='lightblue')
    return m


def get_cluster_data(data):
    data = data.to_crs("epsg:3031")
    cluster_data = pd.DataFrame({'lon': data.geometry.centroid.x,
                                 'lat': data.geometry.centroid.y,
                                 'area': data.geometry.area,
                                 'elev': data.elevation})
    cluster_data = gpd.GeoDataFrame(
        cluster_data, geometry=gpd.points_from_xy(cluster_data.lon, cluster_data.lat)).set_crs('epsg:3031')
    cluster_data = cluster_data.drop(columns=['lon', 'lat'])
    cluster_data['x'] = cluster_data.geometry.apply(lambda point: point.x)
    cluster_data['y'] = cluster_data.geometry.apply(lambda point: point.y)
    return cluster_data


def scale_data(data):
    scaler = MinMaxScaler()
    data = data.drop(columns=['geometry'])
    return scaler.fit_transform(data)
