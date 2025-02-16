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
        self._dataset = dataset
        self.epsg = epsg
        self._polygons = {}      # Store objects for multiple polygons
        self._parcels = {}       # Store objects for multiple parcels
        self._max_index = len(dataset.mid_date)

    def track_polygon(self, id, polygon, **kwargs):
        """
        Creates and tracks a new polygon

        -- Docstring needs LOVE so that it can be initialized properly
        """
        if id in self._polygons:
            raise ValueError(f"Polygon ID {id} already exists")
            
        polygon = Polygon(id, polygon, self._max_index, **kwargs)
        self._polygons[id] = polygon

    def track_parcel(self, id, x, y, **kwargs):
        """
        Creates and tracks a new parcel.
        If track_change=True, also initializes a ChangeArea linked to this parcel.
        
        -- Docstring needs LOVE so that it can be initialized properly
        """
        if id in self._parcels:
            raise ValueError(f"Parcel ID {id} already exists")
        
        parcel = Parcel(id, x, y, self._max_index, **kwargs)
        self._parcels[id] = parcel

    def update_tracking(self, object_type='all'):
        """
        Updates the tracking for the objects stored in LagrangianTracker
        """
        objects = []

        # Get the polygons or parcels depending on desired type
        if object_type == 'all':
            objects = list(self._parcels.values()) + list(self._polygons.values())
        
        elif type == 'parcels':
            objects = list(self._parcels.values())

        elif type == 'polygons':
            objects = list(self._polygons.values())

        # Update position for all tracked objects desired
        for object in objects:
            object.update(self._dataset)

    def get_polygon(self, ids):
        if ids == 'all':
            return self._polygons

        else:
            return self._polygons[ids]

    def get_parcel(self, ids):
        if ids == 'all':
            return self._parcels
        
        else:
            return self._parcels[ids]

            
##################################################################################
"""
TODO: (high to low priority)
- Incorporate change areas into Parcel.update()
- Write merge function
- Add `get_data` method for retreiving final data

- Start visualization class
- Work on docstrings for ease of use
- Rework GlacierDataProcessor to be more modular + clean
"""

class TrackedObject:
    """Base class for tracked objects."""
    def __init__(
        self, 
        id: str,
        max_index: int,
        start_index: int = 0,
        steps_forward: int = None,
        steps_reverse: int = None,
        timeseries_length: int = None
    ):
        self.id = id
        self._backward_history = []  # Save history for reverse tracking
        self._forward_history = []   # Save hsitory for forward tracking
        self.__tracked = False  # Fully Tracked ds flag
        self._dataset = None    # Merged dataset 

        # Store tracking parameters
        self._start_index = start_index
        self._max_index = max_index
        self._steps_forward = steps_forward
        self._steps_reverse = steps_reverse

        self._validate_params()

    def _validate_params(self):
        """
        Assertions and such that will validate the params, raise appropriate type errors
        """
        # Start index validation
        if not isinstance(self._start_index, int) and (self._start_index <= self._max_index) and (self._start_index >= 0):
            raise ValueError(f"`start_index` must be an integer between 0 and {self._max_index}")

        # Reverse Stepping validation
        if not self._steps_reverse:
            self._steps_reverse = 0
        else:
            if not isinstance(self._steps_reverse, int):
                raise TypeError("`steps_reverse` must be an integer")

        # Forward stepping validaiton
        if not self._steps_forward:
            self._steps_forward = (self._max_index - self._start_index)

        else:
            if not isinstance(self.steps_forward, int):
                raise TypeError("`steps_forward` must be an integer value")

            if (self._steps_forward + self._start_index) > self._max_index:
                raise ValueError("`steps_forward` is too large, please pick a lower number")
    
    def update(self, dataset):
        """
        Calls class specific forward and backward tracking
        Merges and sorts the two directional datasets
        Marks object as tracked
        """
        if not self.is_tracked():
            print(f'Tracking {self.id}')
            self._forward(dataset)

            self._backward(dataset)

            self._merge_and_sort()

            self.mark_tracked()

    def _forward(self, dataset):        
        for i in range(self._steps_forward-1):
            self._move_points(dataset, i, direction=1)

    def _backward(self, dataset):
        for i in range(self._steps_reverse):
            self._move_points(dataset, self._start_index-i, direction=-1)
    
    def _move_points(self, dataset, index, direction):
        raise NotImplementedError('Subclasses must implment `_move_points`')

    def mark_tracked(self):
        """Flag that sets object status to fully tracked"""
        self.__tracked = True

    def is_tracked(self):
        """Check if item is fully tracked"""
        return self.__tracked

class Polygon(TrackedObject):
    """Class for Polygon tracking through time."""
    def __init__(
        self, 
        id, 
        polygon,
        max_index,
        start_index,
        steps_forward,
        steps_reverse,
        remove_threshold = None
    ):
        super().__init__(id, max_index, start_index, steps_forward, steps_reverse)
        self._polygon = polygon
        self._remove_threshold = remove_threshold
        self._fractured_pts = None
        
        self._forward_history.append(polygon)  # Add initial polygon
    
    def _move_points(self, ds, index, direction):
        # Fetch closest version of polygon 
        if index == self._start_index:
            polygon = self._polygon
        
        elif direction == 1:
            polygon = self._forward_history[-1]

        else:
            polygon = self._backward_history[-1]

        points = np.array(polygon.exterior.coords)
        
        for i in range(points.shape[0]):
            # Fetch velocities
            vx = ds.vx[index].sel(x=points[i, 0], y=points[i, 1], method='nearest').data
            vy = ds.vy[index].sel(x=points[i, 0], y=points[i, 1], method='nearest').data

            # Set NaN velocities to 0
            if np.isnan(vx): vx = 0
            if np.isnan(vy): vy = 0

            # Change in x- and y-directions
            points[i, 0] += direction * (vx / 12)
            points[i, 1] += direction * (vy / 12)

            if direction == 1:
                self._forward_history.append(Polygon(points))

            else:
                self._backward_history.append(Polygon(points))

class Parcel(TrackedObject):
    """Class for Parcel Tracking through time."""
    def __init__(
        self,
        id: str,
        x: float | int, 
        y: float | int,
        max_index: int,
        track_change: bool = False,
        radius: float | int = None,
        start_index: int = 0,
        steps_forward: int = None,
        steps_reverse: int = None
    ):
        super().__init__(id, max_index, start_index, steps_forward, steps_reverse)
        self.x = x
        self.y = y
        self._change_area = None

        self._forward_history.append((x, y))

        if track_change and radius:
            self._change_area = ChangeArea(id + '_area', x, y, max_index, radius)
    
    def _move_points(self, ds, index, direction):
        if index == self._start_index:
            x, y = self.x, self.y
        
        elif direction == 1:
            x, y = self._forward_history[-1]

        else:
            x, y = self._backward_history[-1]

        vx = ds.vx[index].sel(x=x, y=y, method='nearest').data
        vy = ds.vy[index].sel(x=x, y=y, method='nearest').data

        if np.isnan(vx): vx = 0
        if np.isnan(vy): vy = 0

        x += direction * (vx / 12)
        y += direction * (vy / 12)

        if direction == 1:
            self._forward_history.append((x, y))
        else:
            self._forward_history.append((x, y))

class ChangeArea(Parcel):
    def __init__(self, id, x, y, max_index, radius):
        super().__init__(id, x, y, max_index)
        self._radius = radius
        self._stress_changes = None

    def grab_area(
        self, 
        ds,
        time_index: int  # Need to decide if using .sel or .isel
    ):
        """Extract stress change in a circular area around the parcel."""
        x, y, r = self.x, self.y, self._radius
        ds = ds.sel(x=slice(x-r, x+r), y=slice(y-r, y+r))

        xx, yy = np.meshgrid(ds.x.values, ds.y.values)
        distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

        ds['dist'] = (('mid_date', 'y', 'x'), distances)
        area_ds = ds.where(ds['dist'] < r)

        self._history.append(area_ds)
    
    def compute_change(self):
        if self.__tracked():
            ds = xr.merge(self._history)
            ds = ds.sortby('mid_date')
        return

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