import os
from functools import partial

import numpy as np
import xarray as xr
import rioxarray as rxr
import dask.array as da
import pands as pd
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
        self.dataset = self.get_data_cube()

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
        dt_delta: float = 18,
        start_date: str | np.datetime64 = "2018-07-01",
        end_date: str | np.datetime64 = "2023-01-31",
        chunks: str | dict = 'auto',
    ):
        preprocess = partial(
            _preprocess, 
            shape=self.shape,
            epsg=self.epsg,
            dt_delta=dt_delta,
            start_date=start_date,
            end_date=end_date,
        )

        self.dataset = xr.open_mfdataset(
            self.urls,
            engine='zarr',
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
        self.tracked_parcels = {}       # Store results for multiple parcels
        self.tracked_polygons = {}      # Store results for multiple polygons
        self.tracked_change_areas = {}  # Store results for multiple circular regions

    def move_points(
        self,
        points, 
        index,
        reverse_direction=False,
    ):
        """
        Move points based on velocity fields at a given time index.

        Parameters
        ----------
        points : list
            A list of (x, y) coordinates representing points to be moved.
        index : int
            The time index in the dataset at which the velocity is applied.
        reverse_direction : bool, optional
            If True, moves points in the reverse direction of the velocity field. Default is False.

        Returns
        -------
        list
            Updated list of moved (x, y) points.
        """
        ds = self.dataset
        updated_points = points.copy()
    
        for i, (x, y) in enumerate(updated_points):
            # Fetch velocities at the nearest grid point
            vx = ds.vx[index].sel(x=x, y=y, method="nearest").data
            vy = ds.vy[index].sel(x=x, y=y, method="nearest").data
    
            # Handle NaN velocities by setting to zero
            if np.isnan(vx): vx = 0
            if np.isnan(vy): vy = 0
    
            # Apply velocity move; reverse direction if specified
            factor = -1 if reverse_direction else 1
            x += factor * (vx / 12)
            y += factor * (vy / 12)
    
            updated_points[i] = [x, y]
    
        return updated_points

    def track_polygon(self, polygon_id, polygon, **kwargs):
        """
        Tracks a moving polygonal feature over time within a spatial dataset, leveraging velocity data to predict changes 
        in the feature's geometry.
    
        The function iteratively updates a given polygonal geometry to reflect its movement across time steps within an 
        `xarray.Dataset`. The dataset is expected to contain velocity components (`vx` and `vy`) which guide the movement 
        of the polygon. Additional parameters enable filtering of the dataset, handling of fracture points, and temporal 
        movement both forward and backward.
    
        Parameters:
        -----------
        polygon : shapely.geometry.Polygon
            A polygonal shape representing the feature of interest, used as the initial geometry for tracking.
        start_index : int, optional
            The time index to start the tracking process (default is 0, the first time step).
        steps_forward : int, optional
            The number of time steps to track forward in time. If `None`, all available time steps are processed.
        steps_reverse : int, optional
            The number of time steps to move backward from `start_index` to adjust the initial geometry. Defaults to `None`.
        filtersize : int, optional
            Size of the median filter applied to smooth the final output dataset (default is `None`, meaning no filtering).
        remove_threshold : float, optional
            A threshold value for fracture confidence. Points with fracture confidence below this threshold are removed.
            If provided, it must be a float between 0 and 1 (default is `None`).
    
        Returns:
        --------
        xr.Dataset
            A new dataset clipped to the geometry of the moving polygon across the specified time steps. The resulting
            dataset may optionally include smoothed data if a `filtersize` is provided.

        Notes:
        ------
        - The function employs a reverse tracking option (`steps_reverse`) to determine the starting position of the polygon.
        - Polygon movement is guided by velocity fields in the dataset, and the new position is used for iterative clipping.
        - If `remove_threshold` is provided, the function tracks and removes fracture points, ensuring they do not influence
          subsequent frames.
        - The dataset may include spatial attributes like `fracture_conf` for more nuanced operations during tracking.
        """
        result = self._track_polygon(polygon, **kwargs)
        self.tracked_polygons[polygon_id] = result
        return result
    
    def _track_polygon(
        self, 
        polygon: Polygon,
        start_index: int = 0,
        steps_forward: int = None,
        steps_reverse: int = None,
        filtersize: float = None,
        remove_threshold: int = None
    ) -> xr.Dataset:
        """
        Internal method to track a polygon by moving its boundary points over time.
        """
        if not isinstance(geometry, Polygon):
            raise TypeError("Input geometry must be a shapely Polygon")
        if steps_forward is not None and not isinstance(steps_forward, int):
            raise TypeError("steps_forward must be None or an integer")
        if steps_reverse is not None and not isinstance(steps_reverse, int):
            raise TypeError("steps_reverse must be None or an integer")
        if remove_threshold is not None:
            if not isinstance(remove_threshold, float) or not (0 <= remove_threshold <= 1):
                raise ValueError("remove_threshold must be a float between 0 and 1")
        if filtersize is not None and not isinstance(filtersize, int):
            raise TypeError("filtersize must be None or an integer")

        ds = self.dataset

        # Determine end index
        end_index = (
            len(ds.mid_date) if steps_forward is None 
            else min(start_index + steps_forward, len(ds.mid_date))
        )
        
        # Extract coordinates from the input polygon
        points = np.array(geometry.exterior.coords)
        
        # Reverse steps (optional)
        if steps_reverse:
            for i in tqdm(range(steps_reverse), desc="Reversing Polygon Position"):
                points = self.move_points(points, start_index - i, reverse_direction=True)
            start_index -= steps_reverse
        
        # Clip original area
        gdf = gpd.GeoDataFrame(geometry=[Polygon(points)], crs=f"EPSG:{self.epsg}")
        crs = gdf.crs
        first_frame = ds.isel(mid_date=start_index).rio.clip(gdf.geometry, crs, all_touched=True)
        
        # Initalize list of frames
        clipped_frames = [first_frame]
        
        # Efficient handling of fracture points (vectorized approach)
        if remove_threshold is not None:
            fracture_points = (
                first_frame.fracture_conf
                .where(first_frame.fracture_conf > remove_threshold)
                .stack(points=('x', 'y'))
                .dropna(dim='points')
            )
            fracture_points_coords = (
                fracture_points.points.data.tolist() 
                if fracture_points.sizes['points'] > 0 
                else np.empty((0,2))
            )
        
        # Iterate through time steps
        for t in tqdm(range(start_index, end_index - 1), desc="Tracking Polygon Over Time"):
            # Move points
            points = self.move_points(points, t)
            
            # Clip next frame
            geometry = Polygon(points)
            next_frame = ds.isel(mid_date=(t + 1)).rio.clip([geometry], crs, all_touched=True)
            
            # Efficient fracture points removal (masking)
            if remove_threshold is not None:
                # Move fractured coords
                fracture_points_coords = self.move_points(fracture_points_coords, t)
            
                # Create a mask for fracture points
                fracture_mask = np.zeros_like(next_frame['fracture_conf'], dtype=bool)
            
                for x, y in fracture_points_coords:
                    # Find the closest point in the grid (use nearest if necessary)
                    x_idx = np.argmin(np.abs(next_frame.x.values - x))
                    y_idx = np.argmin(np.abs(next_frame.y.values - y))
            
                    # Set the corresponding position in the mask to True
                    fracture_mask[y_idx, x_idx] = True
            
                # Apply the mask: set all values to NaN where the mask is True
                next_frame = next_frame.where(~fracture_mask, np.nan)
        
                # Use vectorized approach to collect new fratured ice
                new_fracture_points = (
                    next_frame.fracture_conf
                    .where(next_frame.fracture_conf > remove_threshold)
                    .stack(points=('x', 'y'))
                    .dropna(dim='points')
                )
                new_fracture_points_coords =(
                    new_fracture_points.points.data.tolist() 
                    if new_fracture_points.sizes['points'] > 0 
                    else np.empty((0,2))
                )
        
                # Stack new fractures on old fractures
                fracture_points_coords = np.vstack((fracture_points_coords, new_fracture_points_coords))
        
            # Append frame to list
            clipped_frames.append(next_frame)
        
        # Concatenate into one dataset
        lagrange_frame = xr.concat(clipped_frames, dim='mid_date')
        return apply_med_filt(lagrange_frame, size=filtersize) if filtersize is not None else lagrange_frame

    def track_parcel(self, parcel_id, x, y, **kwargs):
        """
        Computes the time-series values for a parcel moving across a domain over time. 
        The function integrates strain rate to compute the total strain experienced by the parcel 
        and appends these values as new variables to the dataset.
    
        Parameters
        ----------
        x : int or float
            The x-coordinate of the initial point of interest.
        y : int or float
            The y-coordinate of the initial point of interest.
        buffer : int or float, optional
            Distance for spatial averaging around the point of interest. Default is 120.
        start_index : int, optional
            The time index corresponding to the given (x, y) coordinates. If None, defaults to 0.
        steps_forward : int, optional
            The number of timesteps to move forward in time. Defaults to as many timesteps 
            as available from the start index.
        steps_reverse : int, optional
            The number of timesteps to move backward in time. Defaults to 0.
        filtersize : int, optional
            The size of the median filter applied to smooth the output dataset. Default is None, meaning no filtering.
    
        Returns
        -------
        parcel : xr.Dataset
            A dataset with the time dimension containing the original variables and new variables:
            - `effective_strain`: Cumulative effective strain.
            - `e_xx`: Cumulative strain in the x-direction.
            - `e_yy`: Cumulative strain in the y-direction.
        points : np.ndarray
            An array of shape (N, 2) where N is the total number of timesteps. Each row contains 
            the x and y coordinates of the parcel at each timestep.
    
        Raises
        ------
        TypeError
            If input types are not as expected.
        ValueError
            If `start_index` is out of bounds, if `steps_reverse` or `steps_forward` exceed the dataset limits, 
            or if any parameters are inconsistent with the dataset dimensions.
    
        Notes
        -----
        - The function assumes that the dataset's `mid_date` dimension corresponds to the time axis.
        - The strain rate is integrated cumulatively over time for each timestep.
        - The `move_points` function is used to determine the parcel's location at each timestep.
        - If `filtersize` is specified, a median filter is applied using the `apply_med_filt` function.
        """
        result = self._track_parcel(x, y, **kwargs)
        self.tracked_parcels[parcel_id] = result
        return result
    
    def _track_parcel(
        self,
        x: int | float,
        y: int | float,
        buffer: int | float = 120,
        start_index: int = None,
        steps_forward: int = None,
        steps_reverse: int = None,
        filtersize: int = None
    ) -> xr.Dataset:
        """
        Internal method to track a parcel by moving it over time.
        """
        if not ((type(x) == type(y)) & (isinstance(x, (int, float)))):
            raise ValueError('x- and y-points must be same type, int or float')
        if start_index is not None and not isinstance(start_index, int):
            raise TypeError('Starting index must be an int or None')
        if steps_forward is not None and not isinstance(steps_forward, int):
            raise TypeError("steps_forward must be None or an integer")
        if steps_reverse is not None and not isinstance(steps_reverse, int):
            raise TypeError("steps_reverse must be None or an integer")
        if not isinstance(buffer, (int, float)):
            raise TypeError("buffer must be inter or float")
        if filtersize is not None and not isinstance(filtersize, int):
            raise TypeError("filtersize must be None or an integer")

        ds = self.dataset

        # Initialize variable, ensure indexing works
        # Grab length of time index
        length = len(ds.mid_date)
        
        # If start index not given, 
        if start_index is None:
            start_index = 0
            steps_reverse = 0
        else:
            # If start index given, check that the index is less than number of timesteps
            if start_index >= length:
                raise ValueError("Starting index must be at least 1 less than the mid_date length")
        
        # If steps forward not given, go as many steps forward as possible
        if steps_forward is None:
            steps_forward = (length - start_index)
        
        # If no steps reversed given, set to 0
        if steps_reverse is None:
            steps_reverse = 0
        
        # Define start and end index
        first_step = start_index - steps_reverse
        final_step = start_index + steps_forward
        
        if first_step < 0:
            raise ValueError("Too many steps reverse. Decrease the number of timesteps")
        
        if final_step > len(ds.mid_date):
            raise ValueError("Too many steps forward. Decrease the number of timesteps")
        
        # Define points array for move_points func
        point = np.c_[x, y]
        
        points_f = point.copy()
        
        # Add initial pt to forward step
        for i in range(steps_forward-1):
            point = self.move_points(point, (start_index+i))
            points_f = np.vstack((points_f, point))
        
        point = np.c_[x, y]
        points_rev = np.empty((0,2))
        for i in range(0, steps_reverse):
            point = self.move_points(point, (start_index-i), reverse_direction=True)
            points_rev = np.vstack((points_rev, point))
        
        points = np.vstack((points_rev[::-1], points_f))
        
        # Initialize list for selection of dfs for each point
        point_vals = []    
        for i, time_index in enumerate(range(first_step, final_step)):
            x, y = points[i]  # Grab point at time index
        
            parcel = ds.sel(x=slice(x-buffer, x+buffer), y=slice(y-buffer, y+buffer)).mean(['x','y'], skipna=True)
            point_vals.append(parcel.isel(mid_date=time_index))
        
        # concat  list of points
        point_srs = xr.concat(point_vals, dim='mid_date')
        
        # Compute total strain using cumulative sum
        strain = point_srs[['effective', 'eps_xx', 'eps_yy']].cumsum(dim='mid_date')
        strain = strain.rename_vars({'effective':'effective_strain','eps_xx':'e_xx', 'eps_yy':'e_yy'})
        
        # Add strain vars to dataset
        parcel = xr.merge([point_srs, strain])
        
        # Apply median filter
        if filtersize:
            parcel = apply_med_filt(parcel, filtersize)
        
        # Add x- and y-coordinates to dataset at function of mid_date
        parcel[['x', 'y']] = [(('mid_date'), points.T[0]), (('mid_date'), points.T[1])]
    
        return parcel

    def change_around_parcel(self, parcel_id, x, y, r, **kwargs):
        """
        Compute stress changes over time within a circular region.
    
        This function iterates through all time steps in `parcel_ds`, extracting a 
        circular region of radius `radius` centered on `(x, y)` at each time step. 
        It computes the differences in stress metrics (`sigma1`, `sigma2`, `von_mises stress`) 
        between consecutive time steps and returns them in a new `xarray.Dataset`.
    
        Parameters
        ----------
        parcel_ds : xarray.Dataset
            Dataset containing parcel location coordinates (`x`, `y`) indexed by `mid_date`.
        ds : xarray.Dataset
            Dataset containing stress values (`sigma1`, `sigma2`, `von_mises`, `fracture_conf`), 
            indexed by `mid_date`, `x`, and `y`.
        radius : float
            The radius of the circular region for stress analysis (in dataset coordinate units).
    
        Returns
        -------
        xarray.Dataset
            A dataset containing:
            
            - `delta_P1` : (mid_date, y, x) Change in Principal Stress 1 (`sigma1`) over time.
            - `delta_P2` : (mid_date, y, x) Change in Principal Stress 2 (`sigma2`) over time.
            - `delta_VM` : (mid_date, y, x) Change in von Mises stress over time.
            - `fracture_conf` : (mid_date, y, x) Fracture confidence values at each time step.
    
            The dataset is indexed by `mid_date`, `y`, and `x`, with coordinates for 
            `x` and `y` varying per time step.
    
        Notes
        -----
        - The function requires `extract_circular_region` to define the extraction method.
        - The first time step does not have a corresponding `delta_*` value since differences 
          are computed between consecutive frames.
        """
        result = self._change_around_parcel(x, y, r, **kwargs)
        self.tracked_change_areas[parcel_id] = result
        return result
    
    def _change_around_parcel(
        self, 
        x: int | float,
        y: int | float,
        radius: int | float,
        start_index: int = None,
        steps_forward: int = None,
        steps_reverse: int = None,
        filtersize: int = None
    ) -> xr.Dataset:
        
        parcel_ds = self._track_parcel(
            x, y, 
            start_index=start_index, 
            steps_forward=steps_forward, 
            steps_reverse=steps_reverse
        )

        p_i = None  # initialize previous step as empty
        
        # Prepare dictionaries for data storage
        var_names = ['delta_P1', 'delta_P2', 'delta_VM', 'fracture_conf']
        parcel_dict = {var:(('mid_date','y','x'),[]) for var in var_names}
        
        coords = {'mid_date':[], 'x':(('mid_date', 'x'),[]), 'y':(('mid_date', 'y'),[])}
        for t in parcel_ds.mid_date:
            # Grab circular region at t
            p_n, x_coords, y_coords = self.extract_circular_region(parcel_ds, t, radius)
    
            # If previous (x, y) exists:
            if p_i:
                # Calculcate change in stressed, store in dict
                parcel_dict['delta_P1'][1].append(p_n.sigma1.to_numpy() - p_i.sigma1.to_numpy())
                parcel_dict['delta_P2'][1].append(p_n.sigma2.to_numpy() - p_i.sigma2.to_numpy())
                parcel_dict['delta_VM'][1].append(p_n.von_mises.to_numpy() - p_i.von_mises.to_numpy())
                parcel_dict['fracture_conf'][1].append(p_n.fracture_conf.to_numpy())
    
                # Store coordinate data
                coords['mid_date'].append(t.data)
                coords['x'][1].append(x_coords)
                coords['y'][1].append(y_coords)
            
            # Save copy of current frame as previous frame
            p_i = p_n.copy()
    
        return xr.Dataset(data_vars=parcel_dict, coords=coords)

    def extract_circular_region(self, parcel_ds, t, r):
        """
        Extract a circular region of radius `r` centered at (x, y) from a dataset at a given time `t`.
    
        This function first extracts a square bounding box around (x, y) using `slice()`, 
        then applies a circular mask to retain only points within the specified radius.
    
        Parameters
        ----------
        parcel_ds : xarray.Dataset
            Dataset containing parcel location coordinates (`x`, `y`) indexed by `mid_date`.
        t : datetime-like or str
            The timestamp at which to extract the circular region.
        r : float
            The radius of the circular region (in dataset coordinate units).
    
        Returns
        -------
        tuple
            - **xarray.Dataset** : The subset of `ds` within the circular region.
            - **numpy.ndarray** : The x-coordinates of the extracted region.
            - **numpy.ndarray** : The y-coordinates of the extracted region.
    
        Notes
        -----
        - Uses a square bounding box (`x ± r`, `y ± r`) for an initial selection before applying the circular mask.
        - The function assumes that `x` and `y` are scalar values and extracts them using `.item()`.
        """
        ds = self.dataset
        
        # grab (x, y) center
        x = parcel_ds.x.sel(mid_date=t).item()
        y = parcel_ds.y.sel(mid_date=t).item()
        
        # Grab square region around (x, y)
        ds = ds.sel(mid_date=t, x=slice(x - r, x + r), y=slice(y - r, y + r))
        
        # Compute distances from center
        distances = np.sqrt((ds.x.values[:, np.newaxis] - x)**2 + (ds.y.values[np.newaxis, :] - y)**2)
    
        # Save distance from (x, y) as variable in dataset
        ds['radii'] = (('y', 'x'), distances)
        ds = ds.where(ds.radii <= r)
        
        return ds, ds.x.values, ds.y.values

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