import numpy as np
import pandas as pd
import geopandas as gpd
import os

import shapely.wkt as poly_parse
from glob import glob
from datetime import timedelta
from tqdm.notebook import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px
import plotly.graph_objects as go

import shapely
import shapely.ops as ops
import shapely.wkt as poly_parse
from shapely.geometry.polygon import Polygon

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import ConvexHull as ch
from descartes import PolygonPatch

import rasterio
pd.options.mode.chained_assignment = None  # default='warn'

from scipy.interpolate import griddata




def get_elevation_data():
    img = rasterio.open(f'data{os.sep}step_1_data_input{os.sep}DEM{os.sep}1km.tif')
    layer = img.read(1)
    xs = []
    ys = []
    values = []
    layer[layer == -999.0] = np.nan
    resolution = 3
    lower = int(np.floor(resolution / 2))
    upper = int(np.ceil(resolution / 2))

    for i in range(resolution, 5601, resolution):
        for j in range(resolution, 5601, resolution):
            value = layer[(i - lower):(i + upper), (j - lower):(j + upper)].mean()
            if not np.isnan(value):
                values.append(value)
                y, x = img.transform * (i, j)
                xs.append(-x)
                ys.append(-y)

    elevation = gpd.GeoDataFrame({'elevation': values}, geometry=gpd.points_from_xy(xs, ys)).set_crs(
        'epsg:3031').to_crs('epsg:4326')
    elevation['longitude'] = elevation.geometry.x
    elevation['latitude'] = elevation.geometry.y
    # viz_elevation = get_basemap_projection(elevation)
    # fig, ax = plt.subplots(figsize=(12,12))
    # basemap = viz_init(ax)
    # ax.scatter(viz_elevation.x, viz_elevation.y, c=elevation['elevation'], cmap=plt.cm.gnuplot2, zorder=3, s=3)
    return elevation

def load_data():
    # Loading rock_cropout
    rock_cropout = gpd.read_file(f"data{os.sep}step_1_data_input{os.sep}rockoutcrop{os.sep}add_rockoutcrop_landsatWGS84.shp").to_crs('epsg:3031')
    # Loading DEM elevation file, rescale and mean
    #df = pd.read_csv("data/DEM/bamber.5km97.dat", sep=' ', header=None,
    #                 names=['latitude', 'longitude', 'elevation', 'difference'])
    df = get_elevation_data()

    df.loc[:, 'latitude'] = np.around(df.loc[:, 'latitude'], 1)
    df.loc[:, 'longitude'] = np.around(df.loc[:, 'longitude'], 1)

    mean_elev = df.groupby(['latitude', 'longitude']).elevation.mean().reset_index()
    # Loading stations
    stazioni = pd.read_csv(f'data{os.sep}step_1_data_input{os.sep}stazioni.csv', encoding='utf-8')
    stazioni = gpd.GeoDataFrame(
        stazioni, geometry=gpd.points_from_xy(stazioni.longitude, stazioni.latitude)).set_crs('epsg:4326').to_crs('epsg:3031')
    # Loading geo_units and coastlines
    geo_units = gpd.read_file(f'data{os.sep}step_1_data_input{os.sep}GeoUnits{os.sep}shapefile{os.sep}geo_units.shp')
    unconsolidated_classes = ['Hs', 'Qk', 'Quc', 'Qc', 'Qu', 'Qa', 'Qs']
    topography_classes = ['w', 'water', 'ice', '?', 'unknown']

    unconsolidated = geo_units[geo_units.MAPSYMBOL.isin(unconsolidated_classes)]
    rocks = geo_units[
        ~(geo_units.MAPSYMBOL.isin(unconsolidated_classes)) & ~(geo_units.MAPSYMBOL.isin(topography_classes))]

    del geo_units

    coastline = gpd.read_file(f'data{os.sep}step_1_data_input{os.sep}coastline{os.sep}add_coastline_medium_res_line_v7_4.shp')

    return rock_cropout, mean_elev, stazioni, unconsolidated, rocks, coastline


def find_nearest_value(origin_df, lon, lat, var_name):
    try:
        return origin_df[(origin_df.longitude == int(lon)) & (origin_df.latitude == int(lat))][var_name].values[0]
    except Exception as ex:
        # print('Cannot find nearest value, error: ', ex)
        return None


def get_elevation(rock_cropout, mean_elev):
    return pd.Series(rock_cropout.to_crs("epsg:4326").apply(lambda row: find_nearest_value(
        mean_elev, row.geometry.centroid.x, row.geometry.centroid.y, 'elevation'), axis=1))


def get_basemap(projection='spstere', boundinglat=-60, lon_0=180, resolution='h'):
    m = Basemap(projection=projection, boundinglat=boundinglat, lon_0=lon_0, resolution=resolution)
    return m


def get_basemap_projection_util(gdf, basemap=None):
    if not basemap:
        basemap = get_basemap()
    if not gdf.crs:
        "CRS non definito"
        return None
    if gdf.crs != 'epsg:3031':
        gdf = gdf.to_crs("EPSG:3031")
    centr_x = gdf.geometry.centroid.x
    centr_y = gdf.geometry.centroid.y
    centroids = gpd.GeoDataFrame(geometry=gpd.points_from_xy(centr_x, centr_y)) \
        .set_crs('epsg:3031').to_crs('epsg:4326')
    points_x, points_y = basemap(centroids.geometry.x, centroids.geometry.y)
    return pd.DataFrame({'x': points_x, 'y': points_y})


def get_basemap_projection(data, basemap=None):
    if type(data) == list:
        data_frames = []
        for geom in data:
            data_frames.append(get_basemap_projection_util(geom, basemap))
        return data_frames
    else:
        return get_basemap_projection_util(data, basemap)



def viz_init(axes=None, projection='spstere', boundinglat=-60, lon_0=180, resolution='h'):
    m = get_basemap(projection=projection, boundinglat=boundinglat, lon_0=lon_0, resolution=resolution)
    if axes:
        m.drawcoastlines(ax=axes)
        m.fillcontinents(color='white', lake_color='aqua', ax=axes)
        m.drawmapboundary(fill_color='lightblue', ax=axes)
    else:
        m.drawcoastlines()
        m.fillcontinents(color='white', lake_color='aqua')
        m.drawmapboundary(fill_color='lightblue')
    return m


def visualize_exploratory_data(rock_cropout, viz_cropout, viz_stazioni, viz_rocks, viz_uncons):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle('Antartica rock cropout infos:', fontsize=24)
    # First graph, cropout and area
    basemap = viz_init(ax1)
    plot = basemap.scatter(viz_cropout.x, viz_cropout.y, c=np.log(rock_cropout['geometry'].area),
                           cmap=plt.cm.gnuplot2, zorder=3, s=0.1, ax=ax1)
    add_colorbar(plot)
    ax1.set_title('Log area in m^2', fontsize=20)
    # Second graph, cropout and elevation
    basemap = viz_init(ax2)
    plot = basemap.scatter(
        viz_cropout.x, viz_cropout.y, c=rock_cropout['elevation'], cmap=plt.cm.gnuplot2, zorder=3, s=0.1, ax=ax2)
    add_colorbar(plot)
    basemap.scatter(viz_stazioni.x, viz_stazioni.y, c='black', zorder=3, s=50, ax=ax2)
    ax2.set_title('DEM elevation model', fontsize=20)
    # Third graph, rocks and unconsolidated material
    basemap = viz_init(ax3)
    plot = basemap.scatter(viz_rocks.x, viz_rocks.y, c='gray', zorder=3, s=0.1, ax=ax3, label="Rocks")
    basemap.scatter(viz_uncons.x, viz_uncons.y, c='yellow', zorder=3, s=0.1, ax=ax3, label="Unconsolidated")
    lgnd = ax3.legend(scatterpoints=1, fontsize=12)
    lgnd.legendHandles[0]._sizes = [40]
    lgnd.legendHandles[1]._sizes = [40]
    ax3.set_title('Rocks / Unconsolidated distribution', fontsize=20)
    plt.tight_layout(pad=3)
    plt.show()


def get_cluster_data(data):
    if data.crs != "epsg:3031":
        data = data.to_crs("epsg:3031")
    cluster_data = pd.DataFrame({'lon': data.geometry.centroid.x,
                                 'lat': data.geometry.centroid.y,
                                 #'area': data.geometry.area,
                                 'elevation': data.elevation})
    cluster_data = gpd.GeoDataFrame(
        cluster_data, geometry=gpd.points_from_xy(cluster_data.lon, cluster_data.lat)).set_crs('epsg:3031')
    cluster_data = cluster_data.drop(columns=['lon', 'lat'])
    cluster_data['x'] = cluster_data.geometry.apply(lambda point: point.x)
    cluster_data['y'] = cluster_data.geometry.apply(lambda point: point.y)
    #cluster_data = cluster_data[~cluster_data.elevation.isna()]
    cluster_data['elevation'] = cluster_data['elevation'].fillna(cluster_data['elevation'].mean())
    return cluster_data


def scale_data(data):
    scaler = MinMaxScaler()
    data = data.drop(columns=['geometry'])
    return scaler.fit_transform(data)


def add_colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def append_cluster(clusters, point, num_cluster):
    for i in range(num_cluster):
        if clusters.geometry[i].contains(point):
            return i
    return -1


def get_elevation_range(rock_cropout, num_cluster):
    elevation_range = []
    for i in range(num_cluster):
        temp = rock_cropout[rock_cropout.cluster == i]
        elevation_range.append(np.max(temp.elevation) - np.min(temp.elevation))
    return pd.Series(elevation_range)


def get_clusters(num_cluster, rock_cropout, viz_cropout, viz_stazioni, cluster_data):
    from sklearn.cluster import DBSCAN
    #if not cluster_data:
    #    cluster_data = get_cluster_data(rock_cropout)
    clust = KMeans(n_clusters=num_cluster, init="k-means++", n_init=10, tol=1e-04, random_state=42)
    clust.fit(scale_data(cluster_data))
    cluster_data['label'] = clust.labels_


    m = get_basemap()

    clusters = pd.DataFrame(columns=('cluster_num', 'geometry'))
    for label in range(0, num_cluster):
        arr = cluster_data[cluster_data.label == label][['x', 'y']].to_numpy()
        hull = ch(arr)
        polylist = []
        for idx in hull.vertices:  # Indices of points forming the vertices of the convex hull.
            polylist.append(arr[idx])  # Append this index point to list
        p = Polygon(polylist)
        clusters.at[label] = [label, p]
    clusters = gpd.GeoDataFrame(clusters, crs='EPSG:{}'.format(3031), geometry='geometry')

    viz_clusters_centroids = get_basemap_projection(clusters)

    rock_cropout['cluster'] = clust.labels_
    #rock_cropout['cluster'] = rock_cropout.geometry.apply(lambda x: append_cluster(clusters, x.centroid, num_cluster))

    patches = []
    for poly in clusters.to_crs('epsg:4326').geometry:
        if poly.geom_type == 'Polygon':
            mpoly = shapely.ops.transform(m, poly)
            patches.append(PolygonPatch(mpoly))
        elif poly.geom_type == 'MultiPolygon':
            for subpoly in poly:
                mpoly = shapely.ops.transform(m, poly)
                patches.append(PolygonPatch(mpoly))
        else:
            print(poly, 'is neither a polygon nor a multi-polygon. Skipping it')

    clusters['elevation'] = rock_cropout.groupby('cluster').elevation.mean().reset_index().sort_values(by='cluster').iloc[:].elevation.values
    clusters['elevation_range'] = get_elevation_range(rock_cropout, num_cluster)
    clusters['area_km2'] = clusters.geometry.apply(lambda x: np.around(x.area / 1e6, 1))

    fig, ax1 = plt.subplots(figsize=(20, 20))
    m.drawcoastlines()
    m.fillcontinents(color='white', lake_color='aqua')
    m.drawmapboundary(fill_color='lightblue')

    m.scatter(viz_cropout.x, viz_cropout.y, c=clust.labels_, zorder=3, s=0.1)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, match_original=True, zorder=4, alpha=0.5)
    p.set_array(clusters.elevation)
    ax1.add_collection(p)
    plt.colorbar(p, pad=0.01, shrink=0.85, aspect=20)
    m.scatter(viz_clusters_centroids.x, viz_clusters_centroids.y, marker='P', c='white', zorder=4, s=100)
    # m.scatter(viz_stazioni.x, viz_stazioni.y, marker='^', c='black', zorder=4, s=150)
    # plt.savefig(f"cluster_{num_cluster}.jpg", dpi=300)
    return clusters, clust, plt


def get_distance_from_coastline(stazioni, coastline, show=True):
    distance_from_coastline = []

    i = j = 0
    fig, axs = plt.subplots(figsize=(8, 8))
    coastline.plot(ax=axs)
    for point in range(len(stazioni)):
        p1 = stazioni.iloc[point].geometry
        p2 = coastline.iloc[np.argmin([stazioni.iloc[point].geometry.distance(line) for line in coastline.geometry])].geometry
        distance = p1.distance(p2) / 1000
        distance_from_coastline.append(np.around(distance, 2))
        if show:
            axs.scatter(p1.x, p1.y, c='black', zorder=3, s=50)
            axs.scatter(p2.centroid.x, p2.centroid.y, c='red', zorder=3, s=50)
            #axs.text(-1200000, -275000, "km: " + str(np.around(distance, 2)), fontsize=18)
            axs.axis('off')
            x = [p1.x, p2.centroid.x]
            y = [p1.y, p2.centroid.y]
            axs.plot(x, y, '--', markersize=2, linewidth=1, color='gray')
    if show:
        plt.title('Distance from coastline')
        plt.tight_layout()
        plt.show()
    return distance_from_coastline


def get_total_cropout_area_under_radius(point, target, radius):
    if target.crs is not None:
        if target.crs != 'epsg:3031':
            target = target.to_crs('epsg:3031')
        arr = [point.distance(geom) / 1000 for geom in target]
        idx = [idx for idx in range(len(arr)) if arr[idx] < radius]
        return np.around(target.iloc[idx].geometry.area.values.sum()/1e6, 2)
    else:
        print('Target without crs infos')
        return None


    
def plot_patches_and_var(df, scatter=None, scatter_label_col='label', patches_col_var=None, show=True):
    m = get_basemap()
    patches = []
    for poly in df.to_crs('epsg:4326').geometry:
        if poly.geom_type == 'Polygon':
            mpoly = shapely.ops.transform(m, poly)
            patches.append(PolygonPatch(mpoly))
        elif poly.geom_type == 'MultiPolygon':
            for subpoly in poly:
                mpoly = shapely.ops.transform(m, poly)
                patches.append(PolygonPatch(mpoly))
        else:
            print(poly, 'is neither a polygon nor a multi-polygon. Skipping it')

    fig, ax1 = plt.subplots(figsize=(15, 15))
    m.drawcoastlines()
    m.fillcontinents(color='white', lake_color='aqua')
    m.drawmapboundary(fill_color='lightblue')

    if scatter is not None:
        m.scatter(scatter.x, scatter.y, c=scatter_label_col, zorder=3, s=5)
    
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, match_original=True, zorder=4, alpha=0.5)
    if patches_col_var is not None:
        p.set_array(df[patches_col_var].values)
    ax1.add_collection(p)
    plt.colorbar(p, pad=0.01, shrink=0.85, aspect=20)
    if show:
        plt.show()
    return fig, ax1

def csv_to_gpd(df_name, geom_col='geometry', crs='4326'):
    df = pd.read_csv(f"{df_name}.csv")
    df.geometry = df.geometry.apply(lambda x: poly_parse.loads(x))
    df = gpd.GeoDataFrame(df, geometry=geom_col, crs=f"EPSG:{crs}")
    return df

def load_data_from_files(root_path=None):
    if root_path == None:
        root_path = f"data{os.sep}step_1_data_output{os.sep}"
    rock_cropout = csv_to_gpd(f"{root_path}rock_cropout", crs='3031')
    mean_elev = pd.read_csv(f"{root_path}mean_elev.csv")
    unconsolidated = csv_to_gpd(f"{root_path}unconsolidated")
    rocks = csv_to_gpd(f"{root_path}rocks")
    coastline = csv_to_gpd(f"{root_path}coastline", crs='3031')
    cluster_data = csv_to_gpd(f"{root_path}cluster_data", crs='3031')
    stazioni = csv_to_gpd(f"{root_path}stazioni_enriched", crs='3031')
    clusters = csv_to_gpd(f"{root_path}clusters", crs='3031')
    return rock_cropout, mean_elev, stazioni, unconsolidated, rocks, coastline, clusters, cluster_data
