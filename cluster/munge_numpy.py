import pandas as pd
import datetime as dt
import pyproj
import numpy as np


import os

#import geopandas as gpd
#from geographiclib.geodesic import Geodesic
#from geopy.distance import vincenty
#from shapely.geometry import Point

geod = pyproj.Geod(ellps='WGS84')

def to_datetime(string):
    return dt.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

def calculate_distance(long1, lat1, long2, lat2):
    if lat1 == lat2 and long1 == long2:
        return 0
    if False in np.isfinite([long1, long2, lat1, lat2]):
        return np.nan
    if lat1 < -90 or lat1 > 90 or lat2 < -90 or lat2 > 90:
        #raise ValueError('The range of latitudes seems to be invalid.')
        return np.nan
    if long1 < -180 or long1 > 180 or long2 < -180 or long2 > 180:
        return np.nan
        #raise ValueError('The range of longitudes seems to be invalid.')
    angle1,angle2,distance = geod.inv(long1, lat1, long2, lat2)
    return distance

def calculate_velocity(distance, timedelta):
    if timedelta.total_seconds() == 0: return np.nan
    return distance / timedelta.total_seconds()

def calculate_acceleration(velocity, velocity2, timedelta):
    delta_v = velocity2 - velocity
    if timedelta.total_seconds() == 0: return np.nan
    return delta_v / timedelta.total_seconds()


headers_trajectory = ['lat', 'long', 'null', 'altitude', 'timestamp_float', 'date', 'time']
headers_metadf = ['trajectory_id', 'start_time', 'end_time', 'v_ave', 'v_med', 'a_ave', 'a_med', 'labels']


def load_trajectory_df(subfolder, filename, trajectory_id):
    df = pd.read_csv(filename, skiprows=6, header=None, names=headers_trajectory)
    df['trajectory_id'] = trajectory_id
    df['subfolder'] = subfolder
    df['labels'] = ''

    df['datetime'] = df.apply(lambda z: to_datetime(z.date + ' ' + z.time), axis=1)
    df['datetime2'] = df['datetime'].shift(1)
    df['long2'] = df['long'].shift(1)
    df['lat2'] = df['lat'].shift(1)

    df['distance'] = df.apply(lambda z: calculate_distance(z.long, z.lat, z.long2, z.lat2), axis=1)
    df['timedelta'] = df.apply(lambda z: z.datetime - z.datetime2, axis=1)
    df['velocity'] = df.apply(lambda z: calculate_velocity(z.distance, z.timedelta), axis=1)
    df['velocity2'] = df['velocity'].shift(1)
    df['acceleration'] = df.apply(lambda z: calculate_acceleration(z.velocity, z.velocity2, z.timedelta), axis=1)
    df = df.drop(['datetime2', 'long2', 'lat2', 'velocity2', 'null', 'timestamp_float', 'date', 'time'], axis=1)

    return df


def load_labels_df(filename):
    df = pd.read_csv(filename, sep='\t')
    df['start_time'] = df['Start Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    df['end_time'] = df['End Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    df['labels'] = df['Transportation Mode']
    df = df.drop(['End Time', 'Start Time', 'Transportation Mode'], axis=1)
    return df


def retrieve_metadata(df):
    df_meta = pd.DataFrame(columns=headers_metadf)
    trajectory_ids = df['trajectory_id'].unique()
    for ii in range(len(trajectory_ids)):
        trajectory_id = trajectory_ids[ii]
        df_ = df[df['trajectory_id'] == trajectory_id]
        start_time = df_.head(1)['datetime'].values[0]
        end_time = df_.tail(1)['datetime'].values[0]
        v_ave = np.nanmean(df_['velocity'].values)
        v_med = np.nanmedian(df_['velocity'].values)
        a_ave = np.nanmean(df_['acceleration'].values)
        a_med = np.nanmedian(df_['acceleration'].values)
        labels = df_['labels'].unique()
        labels = ",".join(labels)
        df_meta.loc[ii, :] = [trajectory_id, start_time, end_time, v_ave, v_med, a_ave, a_med, labels]
    return df_meta


MAIN_FOLDER = '../../GPSML/Data/'
if not os.path.exists(MAIN_FOLDER):
    MAIN_FOLDER = '../GPSML/Data/'
    assert os.path.exists(MAIN_FOLDER), 'Expected GPSML data in %s' % (MAIN_FOLDER)

labels_file = 'labels.txt'
TRAJ_FOLDER = 'Trajectory/'
directories = os.listdir(MAIN_FOLDER)
OUTPUT_FOLDER = '../processed_data/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

num_direc = len(directories)
print('Subfolder|num trajectories|progress')
for num,subfolder in enumerate(directories):
    list_df_traj = []
    subfolder_ = MAIN_FOLDER + subfolder + '/'
    traj_folder = MAIN_FOLDER + subfolder + '/' + TRAJ_FOLDER
    traj_files = os.listdir(traj_folder)
    print('%5s    |%5s           |(progress %3i of %3i)'%(subfolder, len(traj_files), num, num_direc))
    for traj_file in traj_files:
        trajectory_id = traj_file.split('.')[0]
        filename = traj_folder + traj_file
        df_traj = load_trajectory_df(subfolder, filename, trajectory_id)
        list_df_traj.append(df_traj)
    df_traj_all = pd.concat(list_df_traj)

    if labels_file in os.listdir(subfolder_):
        filename = subfolder_ + labels_file
        df_labels = load_labels_df(filename)
        for idx in df_labels.index.values:
            st = df_labels.ix[idx]['start_time']
            et = df_labels.ix[idx]['end_time']
            labels = df_labels.ix[idx]['labels']
            if labels:
                df_traj_all.loc[(df_traj_all['datetime'] >= st) &
                                (df_traj_all['datetime'] <= et), 'labels'] = labels

    filename = OUTPUT_FOLDER + subfolder + '.csv'
    filename_metadata = OUTPUT_FOLDER + subfolder + '_metadata.csv'

    df_traj_all.to_csv(filename)
    df_metadata = retrieve_metadata(df_traj_all)
    df_metadata.to_csv(filename_metadata)