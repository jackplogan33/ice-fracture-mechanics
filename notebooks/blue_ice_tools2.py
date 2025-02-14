import os
from functools import partial

import numpy as np
import xarray as xr
import rioxarray as rxr
import dask.array as da
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon
from tqdm import tqdm

from scipy.signal import savgol_filter as sg
from scipy.ndimage import median_filter

import matplotlib.pyplot as plt
import imageio.v2 as imageio

##################################################################################

class GlacierDataProcessor:
    """
    GlacierDataProcessor
    =====================
    
    A class for processing and analyzing ITS_LIVE velocity data cubes.
    This class facilitates dataset loading, preprocessing, and computation
    of strain and stress fields for glaciers.
    
    Parameters
    ----------
    shape : gpd.GeoDataFrame, optional
        A GeoDataFrame defining the region of interest.
    points : list or tuple, optional
        A list of (x, y) coordinates for defining specific points of interest.
    epsg : int, default=3031
        EPSG code for the dataset's coordinate reference system.
    dt_delta : float, default=18
        Maximum allowed time difference (in days) for image pairs.
    start_date : str or np.datetime64, default='2018-07-01'
        Start date for filtering the dataset.
    end_date : str or np.datetime64, default='2023-01-31'
        End date for filtering the dataset.
    chunks : str or dict, default='auto'
        Chunk size configuration for dask-based processing.
    
    Attributes
    ----------
    dataset : xr.Dataset
        The loaded and processed xarray dataset.
    dx : int
        Grid spacing in the x-direction (default 120 meters).
    dy : int
        Grid spacing in the y-direction (default 120 meters).
    n : int
        Glen's flow law exponent (default 3).
    A : float
        Flow law coefficient (default 3.5e-25 1/s/Pa^n).
    """
    def __init__(
        self, 
        shape: gpd.GeoDataFrame = None,
        points: list | tuple = None,
        epsg: int = 3031,
        dt_delta: float = 18,
        start_date: str | np.datetime64 = '2018-07-01',
        end_date: str | np.datetime64 = '2023-01-31',
        chunks: str | dict = 'auto'
    ):
        self.shape = shape
        self.points = points
        self.epsg = epsg
        self.urls = self.get_urls()
        self.dt_delta = dt_delta
        self.start_date = start_date
        self.end_date = end_date
        self.chunks = chunks
        self.dataset = None

        # Define stress and strain variables
        self.dx = 120
        self.dy = 120
        self.n = 3
        self.A = 3.5e-25


    def get_urls(self):
        """
        Retrieve URLs from the ITS_LIVE catalog that intersect with a given GeoDataFrame or set of points.
    
        Parameters
        ----------
        shape : gpd.GeoDataFrame, optional
            A GeoDataFrame specifying the region of interest in a given CRS.
        points : list or tuple, optional
            A point (x, y) or a list of points [(x1, y1), (x2, y2), ...].
            Each point must be in the dimensions of the EPSG number passed
        epsg : int, optional
            EPSG code used for filtering or projecting the GeoDataFrame. Default is 3031.
    
        Returns
        -------
        list
            Unique ITS_LIVE data cube URLs intersecting the specified geometry or EPSG code.
    
        Raises
        ------
        TypeError
            If `epsg` is not an integer, or if inputs are not of expected types.
        ValueError
            If the `shape` does not have a defined CRS, or `points` are not properly formatted.
    
        Notes
        -----
        - If both `shape` and `points` are None, the function returns URLs filtered by the EPSG code.
        - Points are expected in (x, y) format.
    
        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> import geopandas as gpd
        >>> polygon = Polygon([(-60, -60), (-60, -61), (-61, -61), (-61, -60), (-60, -60)])
        >>> gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326')
        >>> urls = get_urls(shape=gdf)
        >>> print(urls)
        """
        # preload ITS_LIVE catalog
        catalog_url = 'https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json'
        catalog = gpd.read_file(catalog_url).to_crs(self.epsg)  # Set to CRS of area

        # Empty list for 
        urls = []
        if self.points is not None:
            geometry = [Point(x, y) for x, y in self.points]
            points_gdf = gpd.GeoDataFrame(geometry=geometry, crs=self.epsg)
            catalog_sub = catalog.sjoin(points_gdf, how='inner')
            urls.append(catalog_sub['zarr_url'].drop_duplicates().to_numpy())

        if self.shape is not None:
            self.shape = self.shape.to_crs(self.epsg)
            catalog_sub = catalog.sjoin(self.shape, how='inner')
            urls.append(catalog_sub['zarr_url'].drop_duplicates().to_numpy())
            
        if (self.points is None) and (self.shape is None):
            catalog_sub = catalog[catalog['epsg'] == self.epsg]
            urls.append(catalog_sub['zarr_url'].drop_duplicates().to_numpy())

        return np.concatenate(urls)

    def get_data_cube(
        self,
        urls: str | list | np.ndarray = None,
        engine: str = 'zarr',
        dt_delta: float = 18,
        start_date: str | np.datetime64 = "2018-07-01",
        end_date: str | np.datetime64 = "2023-01-31",
        chunks: str | dict = 'auto',
    ):
        if urls is None:
            urls = self.urls
            
        preprocess = partial(
            _preprocess, 
            shape=self.shape,
            epsg=self.epsg,
            dt_delta=dt_delta,
            start_date=start_date,
            end_date=end_date,
        )

        self.dataset = xr.open_mfdataset(
            urls,
            engine=engine,
            preprocess=preprocess,
            chunks=chunks,
            combine='nested',
            concat_dim='mid_date'
        )

        self.dataset = (
            self.dataset.sortby('mid_date')
            .resample(mid_date='1ME')
            .mean(dim='mid_date', skipna=True, method='cohorts', engine='flox')
        ).chunk({'x':100, 'y':100, 'mid_date':-1})

        self.dx = self.dataset.x.diff('x')[0].item()
        self.dy = self.dataset.y.diff('y')[0].item()
        
        return self.dataset
    
    def compute_strain_stress(
        self, 
        rotate: bool = True, 
        sav_gol: bool = True,
        window_length: int = 11,
        polyorder: int = 2,
        deriv=1,
    ) -> xr.Dataset:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call get_data_cube() first.")

        if sav_gol:
            # Apply savitzy golay filter to calculate gradients
            # For ITS_LIVE dc:
            ## x: axis=-1
            ## y: axis=-2
            L = {
                '11':sg_ufunc(self.dataset.vx, window_length, polyorder, deriv=deriv, axis=-1) / self.dx,
                '12':sg_ufunc(self.dataset.vx, window_length, polyorder, deriv=deriv, axis=-2) / self.dy,
                '21':sg_ufunc(self.dataset.vy, window_length, polyorder, deriv=deriv, axis=-1) / self.dx,
                '22':sg_ufunc(self.dataset.vy, window_length, polyorder, deriv=deriv, axis=-2) / self.dy
            }

        else:
            # Compute strain rates using xr.gradient
            L = {
                '11':self.dataset.vx.differentiate('x'),
                '12':self.dataset.vx.differentiate('y'),
                '21':self.dataset.vy.differentiate('x'),
                '22':self.dataset.vy.differentiate('y')
            }

        # Assign components to strain rate tensor (E)
        E = {
            '11':L['11'],
            '12':0.5 * (L['12'] + L['21']),  # Symmetric part for off-diagonal terms
            '22':L['22']
        }

        if rotate:
            theta = da.arctan2(self.dataset.vy, self.dataset.vx)
            E = rotate_strain_rates(E, theta)

        # Calculate effective strain rate
        E['effective'] = da.sqrt(
            0.5 * ((E['11'] ** 2) + (E['22'] ** 2)) + (E['12'] ** 2)
        )

        S = calculate_stress(
            E['effective'], E['11'], E['22'], A=self.A, n=self.n
        )

        # Assign new variables to dataset
        self.dataset['effective'] = E['effective']
        self.dataset['eps_xx'] = E['11']
        self.dataset['eps_yy'] = E['22']
        self.dataset['von_mises'] = S['VM']
        self.dataset['sigma1'] = S['11']
        self.dataset['sigma2'] = S['22']

        return self.dataset

    def merge_and_compute(
        self, 
        fracture_path='../data/shirase-glacier/shirase-fracture-clipped.nc',
        compute=True
    ):
        frac_ds = xr.open_dataset(fracture_path)
        self.dataset = xr.merge([self.dataset, frac_ds])

        if compute:
            self.dataset = self.dataset.compute()
        return self.dataset

##################################################################################

class LagrangianTracking:
    """
    LagrangianTracking
    ===================
    
    A class for tracking Lagrangian parcels and polygons through a velocity dataset.
    This class enables tracking of specific parcels and polygons over time by moving them
    according to velocity fields from an xarray dataset.
    
    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset containing velocity fields for tracking parcels and polygons.
    epsg : int, optional
        The EPSG code of the dataset projection. Default is 3031.
    
    Attributes
    ----------
    dataset : xr.Dataset
        The xarray dataset used for tracking.
    epsg : int
        The EPSG code specifying the dataset's projection.
    tracked_parcels : dict
        Dictionary to store tracked parcel results.
    tracked_polygons : dict
        Dictionary to store tracked polygon results.
    tracked_change_areas : dict
        Dictionary to store tracked stress change areas.
    """
    def __init__(self, dataset, epsg=3031):
        self.dataset = dataset
        self.epsg = epsg
        self._polygons = {}      # Store objects for multiple polygons
        self._parcels = {}       # Store objects for multiple parcels

    def track_polygon(self, id, polygon, **kwargs):
        """
        Creates and tracks a new polygon
        """
        if id in self.polygons:
            raise ValueError(f"Polygon ID {id} already exists")
            
        polygon = Polygon(id, polygon, self.dataset, **kwargs)
        self.polygons[id] = polygon

    def track_parcel(self, id, x, y, **kwargs):
        """
        Creates and tracks a new parcel.
        If track_change=True, also initializes a ChangeArea linked to this parcel.
        """
        if id in self._parcels:
            raise ValueError(f"Parcel ID {id} already exists")
        
        parcel = Parcel(id, x, y, **kwargs)
        self._parcels[id] = parcel

    def update_tracking(self, objects):
        """
        Updates the tracking for the objects stored in LagrangianTracker
        """
        if type == 'all':
            for parcel in self._parcels.values():
                parcel.update_position(self.dataset)

                if parcel._change_area:
                    parcel._change_area.compute_change(self.dataset)

            for polygon in self._polygons.values():
                polygon.update_bounds(self.dataset)
        
        elif type == 'parcels':
            for parcel in self._parcels.values():
                parcel.update_bounds(self.dataset)
                
                if parcel._change_area:
                    parcel.change_area.compute_change(self.dataset)
                    
        elif type == 'polygons':
            for polygon in self._polygons.values():
                polygon.update_position(self.dataset)

    def get_polygon(self, id):
        return self._polygons[id]

    def get_parcel(self, id):
        return self._parcels[id]

            
##################################################################################

class TrackedObject:
    """Base class for tracked objects."""
    def __init__(self, id: str):
        self.id = id
        self._history = []
        self.__tracked = False

    def add_state(self, state):
        """Record a state in the tracking history."""
        self._history.append(state)

    def mark_tracked(self):
        self.__tracked = True

    def check_tracked(self):
        return self.__tracked

class Polygon(TrackedObject):
    """Class for Polygon tracking through time."""
    def __init__(self, id, polygon):
        super().__init__(id)
        self._polygon = polygon

    def update_bounds(self, ds):
        if self.__tracked:
            print("Polygon fully tracked")
            return
        pass

class Parcel(TrackedObject):
    """Class for Parcel Tracking through time."""
    def __init__(
        self,
        id: str,
        x: float | int, 
        y: float,
        track_change: bool = False,
        radius: float | int = None,
        start_index: int = 0,
        steps_forward: int = None,
        steps_reverse: int = None
    ):
        super().__init__(id)
        self.x = x
        self.y = y
        self._change_area = None

        if track_change and radius:
            self.change_area = ChangeArea(id + '_area', x, y, radius)

    def update_position(self, ds):
        vx = ds.vx
        vy = ds.vy
        

class ChangeArea(Parcel):
    def __init__(self, id, x, y, radius):
        super().__init__(id, x, y)
        self._radius = radius
        self._stress_changes = None

    def calculate_change(self, ds):
        pass
        

##################################################################################

def _preprocess(
    ds: xr.Dataset,
    shape: gpd.GeoDataFrame = None, 
    epsg: int = 3031,
    dt_delta: int = None,
    start_date = None,
    end_date = None
) -> xr.Dataset:
    """
    Preprocess ITS_LIVE velocity data cubes before concatenation.

    Steps:
    - Filter dataset to include specific date range if specified.
    - Filter dataset for time delta (if `dt_delta` is set).
    - Select relevant velocity variables.
    - Clip data to a GeoDataFrame region (if provided).
    - Sort by 'mid_date'.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset to preprocess.
    shape : gpd.GeoDataFrame, optional
        GeoDataFrame for spatial clipping.
    epsg : int, optional
        EPSG code for the CRS. Defaults to 3031.
    dt_delta : int, optional
        Maximum allowed time difference (in days) for image pairs. If None, no filtering is applied.
    start_date : string, optional
        First date to include data in the dataset. Default is None (beginning of record).
    end_date : string, optional
        Last date to include in dataset. Default is None (most recent image)

    Returns
    -------
    xr.Dataset
        Preprocessed xarray Dataset.
    """
    ## TODO: 
    ## figure out single satellite selection

    # Filter by start and end date if specified
    if start_date is not None:
        start_date = np.datetime64(start_date)
        ds = ds.where(ds.mid_date >= start_date, drop=True)
    
    if end_date is not None:
        end_date = np.datetime64(end_date)
        ds = ds.where(ds.mid_date <= end_date, drop=True)

    # Filter by time delta if dt_delta is specified
    if dt_delta is not None:
        time_threshold_ns = np.timedelta64(dt_delta, 'D')
        ds = ds.where(ds.date_dt <= time_threshold_ns)

    # Select relevant velocity variables
    ds = ds[['vx', 'vy']]
    
    # Clip to GeoDataFrame if provided
    if shape is not None:
        ds = ds.rio.write_crs(f"EPSG:{epsg}")
        ds = ds.rio.clip(shape.geometry, shape.crs)

    return ds

##################################################################################

def rotate_strain_rates(E: dict, theta: xr.DataArray) -> dict:
    """
    Rotates the strain rate tensor components by the flow direction theta.
    """
    cos_theta = da.cos(theta)
    sin_theta = da.sin(theta)
    cos2 = cos_theta**2
    sin2 = sin_theta**2
    cos_sin = cos_theta * sin_theta

    E_rot = {
        '11': E['11'] * cos2 + 2 * E['12'] * cos_sin + E['22'] * sin2,
        '22': E['11'] * sin2 - 2 * E['12'] * cos_sin + E['22'] * cos2,
        '12': (E['22'] - E['11']) * cos_sin + E['12'] * (cos2 - sin2),
    }

    return E_rot

##################################################################################

def calculate_stress(effective, exx, eyy, A=3.5e-25, n=3):
    """
    Computes stresses using Glen's Flow Law.
    """
    exp = (1 - n) / n
    A *= 365 * 24 * 3600  # Convert from 1/year to 1/second

    # Deviatoric stress tensor
    T = {
        '11': (A**(-1 / n)) * (effective**exp) * exx,
        '22': (A**(-1 / n)) * (effective**exp) * eyy,
    }

    # Cauchy stress tensor components
    S = {
        '11': (2 * T['11'] + T['22']) / 1000,
        '22': (T['11'] + 2 * T['22']) / 1000,
    }
    # Compute Von Mises stress from Cauchy tensor components
    S['VM'] = da.sqrt((S['11']**2 + S['22']**2 - S['11'] * S['22']))
    
    return S

##################################################################################

def sg_ufunc(arr, window_length, polyorder, deriv, axis):
    """
    Applies Savitzky-Golay filter via xarray's apply_ufunc.
    """
    filt_arr = xr.apply_ufunc(
        sg, 
        arr,
        kwargs={'window_length':window_length, 'polyorder':polyorder, 'deriv':deriv, 'axis':axis},
        dask='parallelized',
        output_dtypes=['float32']
    )
    return filt_arr

##################################################################################

def apply_med_filt(arr, size=3):
    return xr.apply_ufunc(
        median_filter,
        arr,
        kwargs={'size':size}
    )