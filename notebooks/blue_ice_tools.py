import os
from functools import partial

import numpy as np
import xarray as xr
import rioxarray as rxr
import dask.array as da
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon as sPolygon
from shapely.geometry import Point as Point
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
        chunks: str | dict = 'auto',
        n: float | int = 3,
        A: float = 3.5e-25
    ):
        self._shape = shape
        self._points = points
        self._epsg = epsg
        self._urls = self.get_urls()
        self._dt_delta = dt_delta
        self._start_date = start_date
        self._end_date = end_date
        self._chunks = chunks
        self._dataset = None

        # Define stress and strain variables
        self._n = n
        self._A = A

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
        catalog = gpd.read_file(catalog_url).to_crs(self._epsg)  # Set to CRS of area

        # Empty list to store urls
        urls = []

        # get urls for any points
        if self._points is not None:
            # convert points 
            geometry = [Point(x, y) for x, y in self._points]
            points_gdf = gpd.GeoDataFrame(geometry=geometry, crs=self._epsg)
            catalog_sub = catalog.sjoin(points_gdf, how='inner')
            urls.append(catalog_sub['zarr_url'].drop_duplicates().to_numpy())

        if self._shape is not None:
            self.shape = self._shape.to_crs(self._epsg)
            catalog_sub = catalog.sjoin(self._shape, how='inner')
            urls.append(catalog_sub['zarr_url'].drop_duplicates().to_numpy())
            
        if (self._points is None) and (self._shape is None):
            catalog_sub = catalog[catalog['epsg'] == self._epsg]
            urls.append(catalog_sub['zarr_url'].drop_duplicates().to_numpy())

        return np.concatenate(urls)

    def load_dataset(
        self,
        urls: str | list | np.ndarray = None,
        engine: str = 'zarr',
        dt_delta: float = 18,
        start_date: str | np.datetime64 = "2018-07-01",
        end_date: str | np.datetime64 = "2023-01-31",
        chunks: str | dict = 'auto',
    ):
        if urls is None:
            urls = self._urls
            
        preprocess = partial(
            _preprocess, 
            shape=self._shape,
            epsg=self._epsg,
            dt_delta=dt_delta,
            start_date=start_date,
            end_date=end_date,
        )

        # Open with xr.open_mfdataset
        self._dataset = xr.open_mfdataset(
            urls,
            engine=engine,
            preprocess=preprocess,
            chunks=chunks,
            combine='nested',
            concat_dim='mid_date',
            parallel=True
        )

        self._monthly_resample()  # resample to monthly timesteps

        self.dx = self._calc_grid_spacing('x')  # get x-spacing
        self.dy = self._calc_grid_spacing('y')  # get x-spacing

        self._dataset = self._dataset.rio.write_crs(f"EPSG:{self.epsg}")
        
        return self._dataset
        
    def calc_strain_stress(
        self, 
        rotate: bool = True, 
        sav_gol: bool = True,
        window_length: int = 11,
        polyorder: int = 2,
    ) -> xr.Dataset:
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        L = self._sav_gol(window_length, polyorder) if sav_gol else self._deriv()

        self._calc_effective(L)  # Calculate effective strain rate

        if rotate: self._rotate_strain()

        self._calc_stress()  # Calculate cauchy stress tensor

        # Rename dataset vars for clarity
        self._E = self._E.rename_vars({'11':'eps_xx', '12':'eps_xy', '22':'eps_yy'})
        self._S = self._S.rename_vars({'11':'sigma1', '22':'sigma2', 'VM':'von_mises'})

        self._dataset = xr.merge([self._dataset, self._E, self._S])  # Merge three datasets
        return self._dataset

    def compute_dataset(self):
        self._dataset = self._dataset.compute()
        return self._dataset

    def get_dataset(self):
        return self._dataset

    def merge_external_dataset(self, path, **kwargs):
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        ds = xr.open_dataset(path, **kwargs)
        self._dataset = xr.merge([self._dataset, ds])

        return self._dataset

    def get_stress_tensor(self):
        return self._S

    def get_strain_tensor(self):
        return self._E

    def get_epsg(self, number):
        self._epsg = number

    def _monthly_resample(self):
        self._dataset = (
            self._dataset.sortby('mid_date')
            .resample(mid_date='1ME')
            .mean(dim='mid_date', skipna=True, method='cohorts', engine='flox')
        ).chunk({'x':100, 'y':100, 'mid_date':-1})
    
    def _calc_grid_spacing(self, dim):
        diff = self._dataset[dim].diff(dim)

        diff = np.unique(diff)

        if diff.size > 1:
            raise ValueError(f'Dataset does not have uniform stepping along {dim}')

        self.__setattr__(self, 'd'+dim, diff[0])

    def _sav_gol(self, window_length, polyorder):
        vx = self._dataset.vx
        vy = self._dataset.vy

        return {
            '11':sg_ufunc(vx, window_length, polyorder, deriv=1, axis=-1) / self.dx,
            '12':sg_ufunc(vx, window_length, polyorder, deriv=1, axis=-2) / self.dy,
            '21':sg_ufunc(vy, window_length, polyorder, deriv=1, axis=-1) / self.dx,
            '22':sg_ufunc(vy, window_length, polyorder, deriv=1, axis=-2) / self.dy
        }

    def _deriv(self, ):
        vx = self._dataset.vx
        vy = self._dataset.vy
        
        return {
            '11':vx.differentiate('x'),
            '12':vx.differentiate('y'),
            '21':vy.differentiate('x'),
            '22':vy.differentiate('y')
        }

    def _calc_effective(self, L):
        E = {
            '11':L['11'],
            '12':0.5 * (L['12'] + L['21']),  # Symmetric part for off-diagonal terms
            '22':L['22'],
            'effective':da.sqrt(0.5 * ((L['11'] ** 2) + (L['22'] ** 2)) + ((0.5 * (L['12'] + L['21']))** 2))
        }

        self._E = E  # Save to attribute

    def _rotate_strain(self):
        # Calculate angle of velocity
        theta = da.arctan2(self._dataset.vy, self._dataset.vx)  

        # Save trig terms as variables for ease
        cos_theta = da.cos(theta)  
        sin_theta = da.sin(theta)
        cos2 = cos_theta**2
        sin2 = sin_theta**2
        cos_sin = cos_theta * sin_theta

        # rotate strain rates following Alley et. al. 2018
        self._E = xr.Dataset({
            '11': self._E['11'] * cos2 + 2 * self._E['12'] * cos_sin + self._E['22'] * sin2,
            '22': self._E['11'] * sin2 - 2 * self._E['12'] * cos_sin + self._E['22'] * cos2,
            '12': (self._E['22'] - self._E['11']) * cos_sin + self._E['12'] * (cos2 - sin2),
            'effective':self._E['effective']
        })

    def _calc_stress(self):
        n = self._n
        exp = (1 - self._n) / self._n
        A = self._A * (365*24*3600)

        # Deviatoric stress tensor
        T = {
            '11':(A ** (-1/n)) * (self._E['effective'] ** exp) * self._E['11'],
            '22':(A ** (-1/n)) * (self._E['effective'] ** exp) * self._E['22']
        }

        # Cauchy stress tensor
        S = {
            '11':(2 * T['11'] + T['22']) / 1000,  # Along flow component
            '22':(T['11'] + 2 * T['22']) / 1000   # Across flow component
        }

        S['VM'] = da.sqrt((S['11']**2 + S['22']**2 - S['11'] * S['22']))  # von Mises stress

        self._S = xr.Dataset(S)  # Convert to dataset, save as attribute        

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
    tracked_parcels : list
        List to store tracked parcel data.
    tracked_polygons : list
        List to store tracked polygon data.
    """
    def __init__(self, dataset, epsg=3031):
        self._dataset = dataset
        self.epsg = epsg
        self._polygons = []      # Store objects for multiple polygons
        self._parcels = []       # Store objects for multiple parcels
        self._max_index = len(dataset.mid_date)

    def track_polygon(self, id, polygon, **kwargs):
        """
        Creates and tracks a new polygon.

        Parameters
        ----------
        id : str
            Unique identifier for the polygon.
        polygon : shapely.geometry.Polygon
            Shapely polygon representing the initial location.
        override : bool, optional
            If True, allows overwriting an existing polygon with the same ID. 
            Default is False.
        **kwargs : optional
            Additional tracking parameters, including:
            - start_index (int): Time index where tracking starts.
            - steps_forward (int): Steps forward to track.
            - steps_reverse (int): Steps backward to track.
            - remove_threshold (float): Fracture confidence threshold for pixel removal.

        Raises
        ------
        ValueError
            If the ID already exists and override is False.
        """
        if any(poly.id == id for poly in self._polygons):
            raise ValueError(f"Polygon ID {id} already exists")
            
        valid_kwargs = {'start_index', 'steps_forward', 'steps_reverse', 'remove_threshold'}
        for key in kwargs.values():
            if key not in valid_kwargs:
                raise ValueError(f"track_polygon() received unexpected keyword argument: {key}")
        
        polygon = Polygon(id, polygon, self.epsg, self._max_index, **kwargs)
        self._polygons.append(polygon)

    def track_parcel(self, id, x, y, override=False, **kwargs):
        """
        Creates and tracks a new parcel.

        Parameters
        ----------
        id : str
            Unique identifier for the parcel.
        x : float
            X-coordinate of the parcel.
        y : float
            Y-coordinate of the parcel.
        override : bool, optional
            If True, allows overwriting an existing parcel with the same ID. 
            Default is False.
        **kwargs : optional
            Additional tracking parameters, including:
            - start_index (int): Time index where tracking starts.
            - steps_forward (int): Steps forward to track.
            - steps_reverse (int): Steps backward to track.
            - radius (float): Radius for tracking change around the parcel.
            - buffer (float): Distance for spatial averaging. Default is 120.

        Raises
        ------
        ValueError
            If the ID already exists and override is False.
        """
        if any(parc.id == id for parc in self._parcels):
            raise ValueError(f"Parcel ID {id} already exists")
        
        valid_kwargs = {'start_index', 'steps_forward', 'steps_reverse', 'radius', 'buffer'}
        for key in kwargs.keys():
            if key not in valid_kwargs:
                raise ValueError(f"track_parcel() received unexpected keyword argument: {key}")        
        
        parcel = Parcel(id, x, y, self._max_index, **kwargs)
        self._parcels.append(parcel)

    def update_tracking(self, object_type='all'):
        """
        Updates tracking for stored objects.

        Parameters
        ----------
        object_type : str, optional
            Specifies which objects to update. One of {'all', 'parcels', 'polygons'}. 
            Default is 'all'.

        Raises
        ------
        ValueError
            If `object_type` is not one of {'all', 'parcels', 'polygons'}.
        """
        valid_objects = {'all', 'parcels', 'polygons'}
        if object_type not in valid_objects:
            raise ValueError(f'Object type {object_type} not one of {valid_objects}')

        objects = []

        # Get all tracked objects
        if object_type == 'all':
            objects = self._parcels + self._polygons

        # Get only parcels
        elif object_type == 'parcels':
            objects = self._parcels

        # get only polygons
        elif object_type == 'polygons':
            objects = self._polygons

        # Update position for all tracked objects desired
        for obj in objects:
            obj.update(self._dataset)
    
    def get_polygon_data(self):
        """
        Retrieves data from tracked polygons.

        Returns
        -------
        dict
            Dictionary with polygon IDs as keys and polygon data as values.
        """
        polygon_data = {}
        for polygon in self._polygons:
            polygon_data[polygon.id] = polygon.get_data()

        return polygon_data

    def get_parcel_data(self):
        """
        Retrieves data from tracked parcels.

        Parameters
        ----------
        data_type : str, optional
            Specifies the type of parcel data to return. One of {'all', 'point', 'area'}. 
            Default is 'all'.

        Returns
        -------
        dict
            Dictionary with parcel IDs as keys and corresponding data as values.

        Raises
        ------
        ValueError
            If `data_type` is not one of {'all', 'point', 'area'}.
        """
        # Raise error if invalid value passed
        if data_type not in valid_types:
            raise ValueError(f"Type {data_type} is not one of {valid_types}")

        parcel_data = {}
        for parcel in self._parcels:
            if parcel.is_tracked():
                parcel_data[parcel.id] = parcel.get_data(data_type)

        return parcel_data

    def plot_tracked_objects(self, index=27, figsize=(10,8)):
        fig, ax = plt.subplots(figsize=figsize)
                
        self._dataset.fracture_conf[index].plot(ax=ax, cbar_kwargs={'label':'Fracture Confidence [0:1]'})
        
        for i, parcel in enumerate(self._parcels):
            x, y = parcel._history[index]        
            ax.plot(x, y, markersize=2, c=f'C{i}', marker='o', ls='')
        
        for polygon in self._polygons:
            gdf = gpd.GeoDataFrame(geometry=[polygon._history[index]], crs=self.epsg)
            gdf.plot(ax=ax, facecolor='none', edgecolor='k', lw=1,)
        
        ax.set_title(f"Tracked Objects on {np.datetime64(self._dataset.mid_date[index].data, 'D')}")
        ax.set_xlabel('x [meters]')
        ax.set_ylabel('y [meters]')
        ax.set_aspect('equal')
        ax.legend()
        
        plt.show()

    def plot_stress_drop(self):
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        ax1, ax2, ax3 = axs

        for i, parcel in enumerate(self._parcels):
            if parcel.get_break_time():
                parcel_t0, index = parcel.get_break_time()
            else:
                continue

            parcel_ds = parcel.get_data('point')
            parcel_ds = parcel_ds.isel(mid_date=slice(index, -1))
            
            parcel_diff = parcel_ds.diff('mid_date')

            t = np.linspace(1, len(parcel_diff.mid_date), len(parcel_diff.mid_date), dtype='int64')

            ax1.plot(t, parcel_diff.von_mises, ls='-', c=f"C{i}")
            ax2.plot(t, parcel_diff.sigma1, ls='-', c=f'C{i}')
            ax3.plot(t, parcel_diff.sigma2, ls='-', c=f'C{i}')

        ax1.set_ylabel('$\Delta \sigma_{vm}$ [kPa]')
        ax2.set_ylabel('$\Delta \sigma_1$ [kPa]')
        ax3.set_ylabel('$\Delta \sigma_2$ [kPa]')
        ax3.set_xlabel('$\Delta t$ [months]')

        for ax in axs:
            ax.set_xlim(1,16)

        plt.suptitle('Change in Stress per Month\nAfter Fracture')

    def plot_prior_stress_change(self):
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        ax1, ax2, ax3 = axs

        for i, parcel in enumerate(self._parcels):
            if parcel.get_break_time():
                parcel_t0, index = parcel.get_break_time()
            else:
                continue

            parcel_ds = parcel.get_data('point')
            parcel_ds = parcel_ds.isel(mid_date=slice(0, index))

            parcel_diff = parcel_ds.diff('mid_date')

            t = np.linspace(-len(parcel_diff.mid_date), 0, len(parcel_diff.mid_date), dtype='int64')
            ax1.plot(t, parcel_diff.von_mises, ls='-', c=f'C{i}')
            ax2.plot(t, parcel_diff.sigma1, ls='-', c=f'C{i}')
            ax3.plot(t, parcel_diff.sigma2, ls='-', c=f'C{i}')

        ax1.set_ylabel('$\Delta \sigma_{vm}$ [kPa]')
        ax2.set_ylabel('$\Delta \sigma_1$ [kPa]')
        ax3.set_ylabel('$\Delta \sigma_2$ [kPa]')
        ax3.set_xlabel('$\Delta t$ [months]')

        for ax in axs:
            ax.set_xlim(-12, 0)
        
        plt.suptitle('Change in Stress per Month Before Fracture')

    def plot_stress_timeseries(self):
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        ax1, ax2, ax3 = axs

        for i, parcel in enumerate(self._parcels):
            if parcel.get_break_time():
                parcel_t0, index = parcel.get_break_time()
            else:
                continue

            parcel_ds = parcel.get_data('point')
            parcel_ds = parcel_ds.isel(mid_date=slice(index-10, index+10))

            t = np.linspace(-10, 10, 20, dtype='int64')
            ax1.plot(t, parcel_ds.von_mises, ls='-', c=f'C{i}')
            ax2.plot(t, parcel_ds.sigma1, ls='-', c=f'C{i}')
            ax3.plot(t, parcel_ds.sigma2, ls='-', c=f'C{i}')

        ax1.set_ylabel('$\Delta \sigma_{vm}$ [kPa]')
        ax2.set_ylabel('$\Delta \sigma_1$ [kPa]')
        ax3.set_ylabel('$\Delta \sigma_2$ [kPa]')
        ax3.set_xlabel('$\Delta t$ [months]')

        for ax in axs:
            ax.set_xlim(-10, 10)
            ax.grid(ls='--', c='gray', lw='0.5', alpha=.5)
        
        plt.suptitle('Change in Stress per Month Before Fracture')

    def area_change_gif(self, id, gif_path, delete_pngs=True, figsize=(5,8), dpi=150):
        for parcel in self._parcels:
            if id == parcel.id:
                ds = parcel.get_data('area')
    
        img_filenames = []
        levels = np.array([.65])
        vmax = np.max([np.abs(ds.von_mises.max().data), np.abs(ds.von_mises.min().data)])
        
        for i in range(1, 54):
            fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi, constrained_layout=True)
        
            ds.von_mises[i].plot(ax=ax, cmap='coolwarm', vmax=vmax, vmin=-vmax, cbar_kwargs={'label':'$\Delta \sigma_{vm} [kPa]$'})
            ds.fracture_conf[i].plot.contour(ax=ax, levels=levels, cmap='Greys')
        
            ax.set_title(None)
            ax.set_xlabel('x [meters]')
            ax.set_ylabel('y [meters]')
            
            ax.set_aspect('equal')

            contour_line = mlines.Line2D([], [], color='k', linewidth=2, label='65% Fracture Confidence')
            ax.legend(handles=[contour_line])
        
            if i - 1 >= 0:
                date1 = np.datetime64(ds.mid_date[i-1].data, 'D')
                date2 = np.datetime64(ds.mid_date[i].data, 'D')
                date_label = f"Change in von Mises Stress\nfrom {date1} to {date2}"
            else:
                date_label = f"{np.datetime64(ds.mid_date[i].data, 'D')}"  # Last frame has no next timestep

            plt.suptitle(date_label, weight='bold')
    
            filepath = gif_path+f'-{i}.png'
            plt.savefig(filepath)
            img_filenames.append(filepath)
            plt.close()
            
        with imageio.get_writer(gif_path, duration=1000, loop=0, palettesize=32) as writer:
            for filename in img_filenames:
                writer.append_data(imageio.imread(filename))
        
        if delete_pngs:
            for filename in img_filenames:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    print(f"Warning: {filename} not found, skipping deletion.")

##################################################################################
"""
TODO: (high to low priority)
- Comment code (started)
- Work on docstrings for ease of use
- Removing fractured points in Polygon class
"""

class TrackedObject:
    """Base class for tracked objects."""
    def __init__(
        self, 
        id: str,
        max_index: int,
        **kwargs
    ):
        self.id = id
        self._history = None         # Placeholder for in-order history
        self._backward_history = []  # Save history for reverse tracking
        self._forward_history = []   # Save hsitory for forward tracking
        self._dataset = []           # Merged dataset 
        self.__tracked = False       # Fully Tracked ds flag

        # Store tracking parameters
        self._max_index = max_index

        # default kwarg values
        defaults = {'start_index':0, 'steps_forward':None, 'steps_reverse':None}
        defaults.update(kwargs)
        
        for key, val in defaults.items():
            self.__setattr__('_'+key, val)

        self._validate_params()

    def _validate_params(self):
        """
        Input sanitization function
        """
        # Start index validation
        if not isinstance(self._start_index, int):
            raise TypeError(f'`start_index` must be an integer')

        if not (self._start_index <= self._max_index) and (self._start_index >= 0)
            raise ValueError(f"`start_index` must be an integer between 0 and {self._max_index}")

        # Reverse Stepping validation
        if not self._steps_reverse:
            self._steps_reverse = 0
        else:
            if not isinstance(self._steps_reverse, int):
                raise TypeError("`steps_reverse` must be an integer")

        # Forward stepping validaiton
        if not self._steps_forward:
            self._steps_forward = (self._max_index - self._start_index) - 1

        else:
            if not isinstance(self._steps_forward, int):
                raise TypeError("`steps_forward` must be an integer value")

            if (self._steps_forward + self._start_index) > self._max_index:
                raise ValueError("`steps_forward` is too large, please pick a lower number")
    
    def update(self, dataset):
        """
        Calls class specific forward and backward tracking
        Merges and sorts the two directional datasets
        Marks object as tracked
        """
        # Only update if not tracked
        if not self.is_tracked():
            self._forward(dataset)    # Track object forward
            self._backward(dataset)   # Track object in reverse
            self._merge_and_sort()    # Merge and timesort
            self._merge_history()     # Merge histories 
            self.mark_tracked()       # Mark object as tracked
    
    def _move_points(self, ds, x, y, index, direction):
        """Internal method for moving xy-points in a velocity field"""
        # Point x- and y-velocites
        vx = ds.vx[index].sel(x=x, y=y, method='nearest').item()
        vy = ds.vy[index].sel(x=x, y=y, method='nearest').item()

        # Set NaN to 0
        if np.isnan(vx): vx = 0
        if np.isnan(vy): vy = 0

        # Compute distnace travellen in 1 month
        # multiply by direction factor
        x += direction * (vx / 12)
        y += direction * (vy / 12)

        return [x, y]

    def _merge_history(self):
        self._history = self._backward_history[::-1] + self._forward_history
    
    def mark_tracked(self):
        """Flag that sets object status to fully tracked"""
        self.__tracked = True

    def is_tracked(self):
        """Check if item is fully tracked"""
        return self.__tracked

    def _forward(self, dataset):
        raise NotImplementedError('Subclasses must implement `_forward`')
        
    def _backward(self, dataset):
        raise NotImplementedError('Subclasses must implement `_backward`')

    def _merge_and_sort(self):
        raise NotImplementedError('Subclasses must implement `move_points`')

    def get_data(self):
        raise NotImplementedError('Subclasses must implement `get_data`')

class Polygon(TrackedObject):
    """Class for Polygon tracking through time."""
    def __init__(
        self, 
        id, 
        polygon,
        epsg,
        max_index,
        **kwargs
    ):  
        defaults = {'remove_threshold':None}
        defaults.update(kwargs)
        
        super().__init__(id, max_index, **kwargs)
        self._polygon = polygon
        self._epsg = epsg
        self._fractured_pts = None
        
    def _validate_kwargs(self):
        pass
    
    def _forward(self, ds):
        frame = ds.isel(mid_date=self._start_index).rio.clip([self._polygon], f"EPSG:{self._epsg}", all_touched=True)
        self._dataset.append(frame)

        polygon = self._polygon
        for i in range(self._steps_forward):
            index = self._start_index + i
            points = np.array(polygon.exterior.coords)
            for i in range(points.shape[0]):
                points[i, :] = self._move_points(ds, points[i,0], points[i,1], index, direction=1)
            polygon = sPolygon(points)
            frame = ds.isel(mid_date=index+1).rio.clip([polygon], f"EPSG:{self._epsg}", all_touched=True)

            self._forward_history.append(polygon)
            self._dataset.append(frame)

    def _backward(self, ds):
        polygon = self._polygon
        for i in range(self._steps_reverse):
            index = self._start_index - i
            points = np.array(polygon.exterior.coords)
            
            for i in range(points.shape[0]):
                points[i,:] = self._move_points(ds, points[i,0], points[i,1], index, direction=-1)
            polygon = sPolygon(points)
            frame = ds.isel(mid_date=index-1).rio.clip([polygon], f"EPSG:{self._epsg}", all_touched=True)
            
            self._backward_history.append(polygon)
            self._dataset.append(frame)

    def _merge_and_sort(self):
        self._dataset = xr.concat(self._dataset, dim='mid_date')
        self._dataset = self._dataset.sortby('mid_date')

    def get_data(self):
        return self._dataset

class Parcel(TrackedObject):
    """Class for Parcel Tracking through time."""
    def __init__(
        self,
        id: str,
        x: float | int, 
        y: float | int,
        max_index: int,
        **kwargs
    ):
        # Creates default values for kwargs
        defaults = {'radius':None, 'buffer':None}
        defaults.update(kwargs)

        super().__init__(id, max_index, **defaults)
        self.x = x
        self.y = y
        self._break_time = None

        if radius:
            self._change_areas = []
            self._change_dataset = None
        
    def _forward(self, ds):
        # Save initial (x, y) as variables
        x, y = self.x, self.y

        # Select inital frame, save to attribute
        pt_data = self._point_data(ds, x, y, self._start_index)
        self._dataset.append(pt_data)

        # Selected radius around if passed
        if self._radius: self._areas_data(ds, self._start_index, x, y)

        # Iterate through number of steps forward
        for i in range(self._steps_forward):
            t = self._start_index + i  # current time index

            # Move (x, y), select point data at *next* time index
            x, y = self._move_points(ds, x, y, t, direction=1)
            pt_data = self._point_data(ds, x, y, t+1)

            # Select area around if radius passed
            if self._radius: self._areas_data(ds, t+1, x, y)
                
            # Add states to dataset and forward history
            self._dataset.append(pt_data)
            self._forward_history.append((x, y))

    def _backward(self, ds):
        # Save initial (x, y) as variables
        x, y = self.x, self.y

        # Iterate through number of steps backward
        for i in range(self._steps_reverse):
            t = self._start_index - i  # Current time index

            # Move (x, y), select point data at *previous* time index
            x, y = self._move_points(ds, x, y, t, direction=-1)
            pt_data = self._point_data(ds, x, y, t-1)

            # Select area around redius if passed
            if self._radius: self._areas_data(ds, t-1, x, y)

            # Add states to dataset and forward history
            self._dataset.append(pt_data)
            self._backward_history.append((x, y))

    def _point_data(self, ds, x, y, t):
        ds = ds.isel(mid_date=t)     # select time index

        if self._buffer:
            # Select range around point, take the spatial mean
            ds_clip = ds.sel(
                x=slice(x-self._buffer, x+self._buffer), 
                y=slice(y-self._buffer, y+self._buffer)
            ).mean(['x', 'y'], skipna=True)

        else:
            # Select nearest point
            ds_clip = ds.sel(x=x, y=y, method='nearest')

        return ds_clip

    def _areas_data(self, ds, t, x, y):
        """Extract a circular area around the parcel."""
        r = self._radius

        # Grab bounding box
        ds = ds.isel(mid_date=t).sel(x=slice(x-r, x+r), y=slice(y-r, y+r))

        # Calculate the distances from (x, y)
        xx, yy = np.meshgrid(ds.x.values, ds.y.values)
        distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

        ds['dist'] = (('y', 'x'), distances)  # New variable in ds
        area_ds = ds.where(ds['dist'] < r)    # Get only distances < r

        self._change_areas.append(area_ds)    # add state to change areas

    def _merge_and_sort(self):
        self._dataset = xr.concat(self._dataset, dim='mid_date')  # Concatenate point dataset
        self._dataset = self._dataset.sortby('mid_date')          # Sort by time

        # Compute change between timesteps if radius tracked
        if self._radius: self._compute_change()

        self.find_break_time()

    def _compute_change(self):
        # Concatenate change areas, sort by time
        area_ds = xr.concat(self._change_areas, dim='mid_date').sortby('mid_date')

        # For Xarray: set up coords and data variables
        coords = {'mid_date':[],'y':(('mid_date','y'),[]), 'x':(('mid_date','x'),[])}
        dims = ('mid_date', 'y', 'x')
        change_data = {'d_'+var:(dims, []) for var in area_ds.data_vars}
        
        for i in range(1, len(area_ds.mid_date)):  # iterate timeseries
            for var, values in area_ds.items():    # iterate variables
                # send arrays to numpy, compute difference
                var_change = values[i].to_numpy() - values[i-1].to_numpy()

                # add difference to list for respective variable
                change_data['d_'+var][1].append(var_change)

            # Update coords
            coords['mid_date'].append(values.mid_date[i].data)
            coords['y'][1].append(values.y)
            coords['x'][1].append(values.x)
                
        # Create xarray dataset for difference, save as attribute
        self._change_dataset = xr.Dataset(data_vars=change_data, coords=coords).sortby('mid_date')

    def get_data(self, type):
        if type == 'all':
            return self._dataset, self._change_dataset

        elif type == 'point':
            return self._dataset

        elif type == 'area':
            return self._change_dataset
            
    def find_break_time(self, thresh=0.65):
        point_data = self._dataset
        for i in range(len(point_data.mid_date)):
            if point_data.fracture_conf[i] > thresh:
                self._break_time = (point_data.isel(mid_date=i), i)
                break

    def get_break_time(self):
        return self._break_time

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