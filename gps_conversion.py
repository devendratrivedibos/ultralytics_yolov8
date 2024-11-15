from osgeo import gdal, osr
import camera
import numpy as np
# from common.models import PcamSet
import matplotlib.pyplot as plt
import math
from pyproj import Proj, transform
import pdb
import pandas as pd
import pickle

class GPS_Cord:
    def __init__(self,points,camera_loc):
        self.points = points

        self.ANGLE_x = 180-80
        self.ANGLE_y = 0
        self.ANGLE_z = 0
        self.HEIGHT_x = 0
        self.HEIGHT_y = 0
        self.HEIGHT_z = 2
        self.FOCAL_LENGTH = 0.008
        self.RES_X = 2464
        self.RES_Y = 2056
        self.SENSOR_PIXEL_SIZE = 3.45e-6        

        # Example geodetic coordinates (latitude, longitude, altitude)
        self.latitude = camera_loc['Latitude']
        self.longitude = camera_loc['Longitude']
        self.altitude = camera_loc['Altitude']

        # self.pitch = camera_loc['Pitch']
        # self.roll = camera_loc['Roll']
        # self.head = camera_loc['Head']

        # Define the projection for ECEF (Earth-Centered, Earth-Fixed)
        self.ecef = Proj(proj='geocent', datum='WGS84')

    def geodetic2ecef(self):
        ## body to ECEF
        # Define the projection for WGS84 (latitude, longitude)
        wgs84 = Proj(proj='latlong', datum='WGS84')
        # Convert geodetic coordinates to ECEF coordinates (x, y, z)
        ecef_x, ecef_y, ecef_z = transform(wgs84, Proj(proj='geocent', datum='WGS84'), self.longitude, self.latitude, self.altitude)

        tLL =  np.array(
            			[
            			[ecef_x],
            			[ecef_y],
            			[ecef_z]
            			]
            			)
        cam = camera.Camera()        

        fov_X = 2 * math.atan(self.RES_X*self.SENSOR_PIXEL_SIZE/2/self.FOCAL_LENGTH)
        fov_Y = 2 * math.atan(self.RES_Y*self.SENSOR_PIXEL_SIZE/2/self.FOCAL_LENGTH)

        K_0 = self.RES_X/2/math.tan(fov_X/2)
        K_1 = self.RES_Y/2/math.tan(fov_Y/2)

        K = np.array(
                [[K_0, 0, self.RES_X/2],
                 [0, K_1, self.RES_Y/2],
                 [0, 0, 1]]
            )

        ## camera extrensic matrix

        tf = np.array(
            [
            [self.HEIGHT_x],
            [self.HEIGHT_y],
            [self.HEIGHT_z]
            ]
        )

        ax = np.radians(self.ANGLE_x)
        ay = np.radians(self.ANGLE_y)
        az = np.radians(self.ANGLE_z)

        ap = np.radians(self.latitude)
        al = np.radians(self.longitude)

        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(ax), -np.sin(ax)],
             [0, np.sin(ax), np.cos(ax)]]
            )
        Ry = np.array(
            [[np.cos(ay), 0, np.sin(ay)],
             [0, 1, 0],
             [-np.sin(ay), 0, np.cos(ay)]]
            )
        Rz = np.array(
            [[np.cos(az), -np.sin(az), 0],
             [np.sin(az), np.cos(az), 0],
             [0, 0, 1]]
            )

        R1 = np.matmul(Rz,Ry)
        Rf = np.matmul(R1,Rx)

        RLL = np.array(
            [[-np.sin(al), -np.sin(ap)*np.cos(al), np.cos(ap)*np.cos(al)],
             [-np.cos(al), -np.sin(ap)*np.sin(al), np.cos(ap)*np.sin(al)],
             [0, np.cos(ap), np.sin(ap)]]
            )

        bx = np.radians(self.pitch)
        by = np.radians(self.roll)
        bz = np.radians(self.head)

        Rbx = np.array(
            [[1, 0, 0],
             [0, np.cos(bx), -np.sin(bx)],
             [0, np.sin(bx), np.cos(bx)]]
            )
        Rby = np.array(
            [[np.cos(by), 0, np.sin(by)],
             [0, 1, 0],
             [-np.sin(by), 0, np.cos(by)]]
            )
        Rbz = np.array(
            [[np.cos(bz), -np.sin(bz), 0],
             [np.sin(bz), np.cos(bz), 0],
             [0, 0, 1]]
            )
        Rb1 = np.matmul(Rbz,Rby)
        Rbf = np.matmul(Rb1,Rbx)
        Rt = np.matmul(Rbf,RLL)

        cam.set_K(K)
        cam.set_t(tf)
        cam.set_R(Rf)
        # pdb.set_trace()


        im = np.array(self.points).T

        imp = camera.e2p(im)
        cc = np.matmul(np.linalg.inv(K),imp)
        bc = cam.image_to_world(im, 0.0)
        wc = np.matmul(Rt.T,bc) + tLL

        # Convert ECEF coordinates to GPS coordinates (latitude, longitude, altitude)
        lon, lat, alt = transform(self.ecef, wgs84, wc[0,:], wc[1,:], wc[2,:])

        df  = pd.DataFrame(columns=['latitude','longitude','altitude'])
        df['latitude'] = lat
        df['longitude'] = lon
        df['altitude'] = alt
        return df

