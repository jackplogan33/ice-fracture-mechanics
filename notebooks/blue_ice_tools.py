import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from functools import partial

from scipy.signal import savgol_filter as sg
from tqdm import tqdm

from shapely.geometry import Polygon
from shapely.geometry import Point
from scipy import ndimage
import shapely.geometry

import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})

##################################################################################

def get_urls(
    shape: gpd.GeoDataFrame = None,
    points: list | tuple = None,
    epsg: int = 3031
) -> list:
    """
    Retrieve URLs from the ITS_LIVE catalog that intersect with a given GeoDataFrame or set of points.

    Parameters
    ----------
    shape : gpd.GeoDataFrame, optional
        A GeoDataFrame specifying the region of interest in a given CRS.
    points : list or tuple, optional
        A point (lon, lat) or a list of points [(lon, lat), (lon, lat), ...].
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
    - Points are expected in (lon, lat) format.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> import geopandas as gpd
    >>> polygon = Polygon([(-60, -60), (-60, -61), (-61, -61), (-61, -60), (-60, -60)])
    >>> gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326')
    >>> urls = get_urls(shape=gdf)
    >>> print(urls)
    """
    # Input validation
    if not isinstance(epsg, int):
        raise TypeError("epsg must be an integer.")

    if shape is not None:
        if not isinstance(shape, gpd.GeoDataFrame):
            raise TypeError("Shape must be a GeoDataFrame.")
        if shape.crs is None:
            raise ValueError("Shape GeoDataFrame must have a CRS defined.")
    elif points is not None:
        if isinstance(points, (list, tuple)):
            # Handle single point
            if all(isinstance(coord, (int, float)) for coord in points) and len(points) == 2:
                points = [points]
            # Handle multiple points
            elif all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in points):
                pass  # Already a list of points
            else:
                raise ValueError("Points must be in the format [(lon, lat), (lon, lat), ...] or (lon, lat).")
        else:
            raise TypeError("Points must be a list, tuple, or None.")
    else:
        shape, points = None, None  # Default case

    # Load ITS_LIVE catalog
    catalog_url = 'https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json'
    catalog = gpd.read_file(catalog_url)

    # Logic for points input
    if points:
        # Convert points to a GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in points]
        points_gdf = gpd.GeoDataFrame(geometry=geometry, crs='EPSG:4326')

        # Perform spatial join
        catalog_sub = catalog.sjoin(points_gdf, how='inner')

    # Logic for GeoDataFrame input
    elif shape is not None:
        shape = shape.to_crs(epsg)  # Reproject to specified EPSG
        catalog = catalog.to_crs(epsg)
        catalog_sub = gpd.sjoin(shape, catalog, how='inner')

    # Default case: return all URLs for the given EPSG
    else:
        catalog_sub = catalog[catalog['epsg'] == epsg]

    # Extract unique URLs
    urls = catalog_sub['zarr_url'].drop_duplicates().tolist()
    return urls

##################################################################################

def get_data_cube(
    shape: gpd.GeoDataFrame = None, 
    urls: list | tuple | np.ndarray = None, 
    epsg: int = 3031,
    dt_delta: int = None,
    start_date: str = None,
    end_date: str = None,
    engine: str = 'zarr'
) -> xr.Dataset:
    """
    Download and process ITS_LIVE velocity data cubes, optionally clipping them to a specified geometry.

    Parameters
    ----------
    shape : gpd.GeoDataFrame, optional
        GeoDataFrame defining the region of interest, with a defined CRS.
    urls : list, tuple, or np.ndarray, optional
        URLs of ITS_LIVE data cubes. If not provided, `get_urls` is used with the `shape` and `epsg` parameters.
    epsg : int, optional
        EPSG code for the CRS. Defaults to 3031.
    dt_delta : int, optional
        Maximum allowed time difference (in days) for image pairs. Default is None (no filtering).
    start_date : string, optional
        First date to include data in the dataset. Default is None (beginning of record).
    end_date : string, optional
        Last date to include in dataset. Default is None (most recent image)
    engine : str, optional
        Engine used for reading Zarr files. Defaults to 'zarr'.

    Returns
    -------
    xr.Dataset
        Concatenated and resampled xarray Dataset representing the velocity data.

    Raises
    ------
    TypeError
        If inputs are not of the expected types.
    ValueError
        If `shape` does not have a defined CRS.

    Notes
    -----
    - Data cubes are resampled to monthly intervals using the mean.
    - Input URLs must correspond to valid ITS_LIVE data cubes.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> import geopandas as gpd
    >>> polygon = Polygon([(-60, -60), (-60, -61), (-61, -61), (-61, -60), (-60, -60)])
    >>> gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326')
    >>> dataset = get_data_cube(shape=gdf)
    >>> print(dataset)
    """
    # Validate inputs
    if not isinstance(epsg, int):
        raise TypeError("EPSG must be an integer.")
    if dt_delta is not None and not isinstance(dt_delta, int):
        raise TypeError("dt_delta must be an integer or None.")
    if start_date is not None and not isinstance(start_date, str):
        raise TypeError("start_date must be a string or None")
    if end_date is not None and not isinstance(end_date, str):
        raise TypeError("start_date must be a string or None")
    if not isinstance(engine, str):
        raise TypeError("engine must be a string.")
    if shape is not None:
        if not isinstance(shape, gpd.GeoDataFrame):
            raise TypeError("Shape must be a GeoDataFrame or None.")
        if shape.crs is None:
            raise ValueError("Shape GeoDataFrame must have a CRS defined.")

    # Get URLs if not provided
    if urls is None:
        urls = get_urls(shape, epsg=epsg)
    else:
        if not isinstance(urls, (list, tuple, np.ndarray)):
            raise TypeError("URLs must be a list, tuple, or numpy array.")
        if not all(isinstance(url, str) for url in urls):
            raise ValueError("All items in URLs must be strings.")

    # Set chunking and preprocessing parameters
    chunks = {'mid_date': -1, 'x': 'auto', 'y': 'auto'}
    preprocess = partial(
        _preprocess, 
        shape=shape, 
        epsg=epsg, 
        dt_delta=dt_delta, 
        start_date=start_date,
        end_date=end_date
    )

    # Open datasets and process
    dc = xr.open_mfdataset(
        urls,
        engine=engine,
        preprocess=preprocess,
        chunks=chunks,
        combine='nested',
        concat_dim='mid_date'
    )

    # Sort and resample data
    dc = (
        dc.sortby('mid_date')
        .resample(mid_date='1ME')  # '1ME' means month-end frequency
        .mean(dim='mid_date', skipna=True)
        .chunk('auto')
    )

    return dc

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
    - Filter dataset to include specific satellites if specified.
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
    ds = ds[['vx', 'vy', 'v']]
    
    # Clip to GeoDataFrame if provided
    if shape is not None:
        ds = ds.rio.write_crs(f"EPSG:{epsg}")
        ds = ds.rio.clip(shape.geometry, shape.crs)

    # Sort by mid_date
    return ds.sortby('mid_date')
    
##################################################################################

def compute_strain_stress(
    ds: xr.Dataset,
    rotate: bool = False,
    sav_gol: bool = False,
    dx: int = 120,
    dy: int = 120,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 1,
    n: float = 3,
    A: float = 3.5e-25
) -> xr.DataArray | np.ndarray:
    """
    Computes strain rates and stresses from velocity fields in an xarray Dataset.

    This function calculates the strain rate tensor and derived stress fields based on
    the velocity components (`vx`, `vy`) in the input dataset. It supports two methods
    for calculating gradients: Savitzky-Golay filtering or finite differences. The user
    can optionally apply tensor rotation to align results with a local coordinate system.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing velocity components `vx` and `vy`.
    sav_gol : bool, optional
        If True, use Savitzky-Golay filtering for gradient computation. 
        If False, use finite differences via xarray's `.differentiate` (default is False).
    dx : int, optional
        Spatial resolution in the x-direction (default is 120). Only relevant if `sav_gol` is True.
    dy : int, optional
        Spatial resolution in the y-direction (default is 120). Only relevant if `sav_gol` is True.
    window_length : int, optional
        Length of the Savitzky-Golay filter window (default is 11). Only relevant if `sav_gol` is True.
    polyorder : int, optional
        Polynomial order for the Savitzky-Golay filter (default is 2). Only relevant if `sav_gol` is True.
    deriv : int, optional
        Derivative order for the Savitzky-Golay filter (default is 1). Only relevant if `sav_gol` is True.
    rotate : bool, optional
        If True, rotates the strain rate tensor to align with local flow direction (default is False).
    n : float, optional
        Glen's flow law exponent (default is 3).
    A : float, optional
        Temperature-dependent flow law constant (default is 3.5e-25).

    Returns
    -------
    xr.Dataset
        The input dataset with additional variables:
        - `effective`: Effective strain rate.
        - `eps_xx`: Strain rate tensor component xx.
        - `eps_yy`: Strain rate tensor component yy.
        - `von_mises`: Von Mises stress.
        - `sigma1`: Principal stress component 1.
        - `sigma2`: Principal stress component 2.

    Notes
    -----
    - The Savitzky-Golay filter is applied along the specified axes with constant window
      length and polynomial order. Ensure that `vx` and `vy` are on the appropriate axes.
    - The strain rate tensor is symmetric, with off-diagonal components averaged appropriately.
    - Stress calculations follow Glen's Flow Law and are scaled to kilopascals (kPa).
    - Rotational calculations are based on the arctangent of velocity components to derive
      the local flow direction.

    Examples
    --------
    Using finite differences to compute strain and stress:
    >>> result = compute_strain_stress(ds, dx=100, dy=100, sav_gol=False)

    Using Savitzky-Golay filtering for gradient computation:
    >>> result = compute_strain_stress(ds, sav_gol=True, window_length=13, polyorder=3)

    Rotating strain rates:
    >>> result = compute_strain_stress(ds, rotate=True)
    """
    # Initialize gradients dict (L)
    L = {}
    progress_bar = tqdm(total=4, desc=f'{"Computing gradients":<25}', position=0)
    
    # Compute velocty components using method of choice
    if sav_gol:
        # Apply savitzy golay filter to calculate gradients
        # For ITS_LIVE dc:
        ## x: axis=-1
        ## y: axis=-2
        L['11'] = sg_ufunc(ds.vx, window_length, polyorder, deriv=deriv, axis=-1) / dx
        progress_bar.update(1)
        L['12'] = sg_ufunc(ds.vx, window_length, polyorder, deriv=deriv, axis=-2) / dy
        progress_bar.update(1)
        L['21'] = sg_ufunc(ds.vy, window_length, polyorder, deriv=deriv, axis=-1) / dx
        progress_bar.update(1)
        L['22'] = sg_ufunc(ds.vy, window_length, polyorder, deriv=deriv, axis=-2) / dy
    
    else:
        # Compute strain rates using xr.gradient
        L['11'] = ds.vx.differentiate('x')
        progress_bar.update(1)
        L['12'] = ds.vx.differentiate('y')
        progress_bar.update(1)
        L['21'] = ds.vy.differentiate('x')
        progress_bar.update(1)
        L['22'] = ds.vy.differentiate('y')

    progress_bar.update(1)
    progress_bar.close()
    
    # Initialize strain rate tensor (E)
    E = {}

    # Assign components to tensor
    E['11'] = L['11']
    E['12'] = 0.5 * (L['12'] + L['21'])  # Symmetric part for off-diagonal terms
    E['22'] = L['22']
    
    # Rotate Strain Rates
    if rotate:
        progress_bar = tqdm(total=1, desc=f'{"Rotating Strain Rates":<25}', position=0)
        theta = np.arctan2(ds.vy, ds.vx)
        E = rotate_strain_rates(E, theta)
        
        progress_bar.update(1)
        progress_bar.close()
    
    # Calculate effective strain rate
    E['effective'] = np.sqrt(
        0.5 * ((E['11'] ** 2) + (E['22'] ** 2)) + (E['12'] ** 2)
    )
    
    progress_bar = tqdm(total=1, desc=f'{"Computing stresses":<25}', position=0)
    # Stress Computation
    S = _strain_stress(
        E['effective'], E['11'], E['22'], A=A, n=n
    )

    progress_bar.update(1)
    progress_bar.close()

    # Assign new variables to dataset
    ds['effective'] = E['effective']
    ds['eps_xx'] = E['11']
    ds['eps_yy'] = E['22']
    ds['von_mises'] = S['VM']
    ds['sigma1'] = S['11']
    ds['sigma2'] = S['22']

    return ds

##################################################################################

def rotate_strain_rates(E: dict, theta: xr.DataArray) -> dict:
    """
    Rotates the strain rate tensor components by the flow direction theta.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
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

def _strain_stress(effective, exx, eyy, A=3.5e-25, n=3):
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
    S['VM'] = np.sqrt((S['11']**2 + S['22']**2 - S['11'] * S['22']))
    
    return S

##################################################################################

def sg_ufunc(arr, window_length, polyorder, deriv, axis):
    """
    Applies Savitzky-Golay filter via xarray's apply_ufunc.
    """
    filt_arr = xr.apply_ufunc(
        sg, arr,
        kwargs={'window_length':window_length, 'polyorder':polyorder, 'deriv':deriv, 'axis':axis}
    )
    return filt_arr

##################################################################################

def lagrangian_frame(
    ds: xr.Dataset, 
    geometry: Polygon, 
    start_index: int = 0,
    steps_forward: int = None,
    steps_reverse: int = None,
    epsg: int = 3031,
    filtersize: int = None,
    remove_threshold : float = None
) -> xr.Dataset:
    """
    Tracks a moving polygonal feature over time within a spatial dataset, leveraging velocity data to predict changes 
    in the feature's geometry.

    The function iteratively updates a given polygonal geometry to reflect its movement across time steps within an 
    `xarray.Dataset`. The dataset is expected to contain velocity components (`vx` and `vy`) which guide the movement 
    of the polygon. Additional parameters enable filtering of the dataset, handling of fracture points, and temporal 
    movement both forward and backward.

    Parameters:
    -----------
    ds : xr.Dataset
        The dataset containing velocity fields `vx` and `vy`, and a temporal dimension `mid_date`. 
        It must also include spatial coordinates and optionally a fracture confidence field (`fracture_conf`).
    geometry : shapely.geometry.Polygon
        A polygonal shape representing the feature of interest, used as the initial geometry for tracking.
    start_index : int, optional
        The time index to start the tracking process (default is 0, the first time step).
    steps_forward : int, optional
        The number of time steps to track forward in time. If `None`, all available time steps are processed.
    steps_reverse : int, optional
        The number of time steps to move backward from `start_index` to adjust the initial geometry. Defaults to `None`.
    epsg : int, optional
        EPSG code defining the coordinate reference system of the dataset (default is 3031, commonly used for polar data).
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

    Raises:
    -------
    TypeError
        If inputs are of invalid types, e.g., `ds` is not an `xarray.Dataset` or `geometry` is not a `shapely.Polygon`.
    ValueError
        If `remove_threshold` is not a float or is outside the range [0, 1].

    Notes:
    ------
    - The function employs a reverse tracking option (`steps_reverse`) to refine the starting position of the polygon.
    - Polygon movement is guided by velocity fields in the dataset, and the new position is used for iterative clipping.
    - If `remove_threshold` is provided, the function tracks and removes fracture points, ensuring they do not influence
      subsequent frames.
    - The dataset may include spatial attributes like `fracture_conf` for more nuanced operations during tracking.

    Examples:
    ---------
    >>> from shapely.geometry import Polygon
    >>> import xarray as xr
    >>> dataset = xr.open_dataset("velocity_data.nc")
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> tracked_ds = lagrangian_frame(
    ...     ds=dataset, 
    ...     geometry=polygon, 
    ...     start_index=0, 
    ...     steps_forward=10, 
    ...     filtersize=3
    ... )
    >>> tracked_ds.to_netcdf("tracked_output.nc")
    """
    # Defensive programming
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input ds must be an xarray.Dataset")
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
            points = move_points(points, start_index - i, ds, reverse_direction=True)
        start_index -= steps_reverse
    
    # Clip original area
    gdf = gpd.GeoDataFrame(geometry=[Polygon(points)], crs=f"EPSG:{epsg}")
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
        points = move_points(points, t, ds)
        
        # Clip next frame
        geometry = Polygon(points)
        next_frame = ds.isel(mid_date=(t + 1)).rio.clip([geometry], crs, all_touched=True)
        
        # Efficient fracture points removal (masking)
        if remove_threshold is not None:
            # Move fractured coords
            fracture_points_coords = move_points(fracture_points_coords, t, ds)
        
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

##################################################################################

def parcel_strain_stress(
    ds: xr.Dataset,
    x: (int | float),
    y: (int | float),
    buffer: (int | float) = 0,
    start_index: int = None,
    steps_forward: int = None,
    steps_reverse: int = None,
    filtersize: int = None
) -> xr.Dataset:
    """
    Computes the time-series values for a parcel moving across a domain over time. 
    The function integrates strain rate to compute the total strain experienced by the parcel 
    and appends these values as new variables to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray dataset. Must contain `x` and `y` velocities and strain rates 
        (`effective`, `eps_xx`, `eps_yy`) as variables.
    x : int or float
        The x-coordinate of the initial point of interest.
    y : int or float
        The y-coordinate of the initial point of interest.
    buffer : int or float, optional
        Distance for spatial averaging around the point of interest. Default is 0 (no averaging).
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

    Examples
    --------
    >>> parcel, points = parcel_strain_stress(
    ...     ds=my_dataset, x=100.0, y=50.0, buffer=10.0, 
    ...     start_index=0, steps_forward=10, steps_reverse=5, filtersize=3
    ... )
    >>> print(parcel)
    >>> print(points)
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError('Input ds must be an xarray.Dataset')
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
        steps_forward = length - start_index
    
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
    for i in range(steps_forward):
        point = move_points(point, (start_index+i), ds)
        points_f = np.vstack((points_f, point))
    
    point = np.c_[x, y]
    points_rev = np.empty((0,2))
    for i in range(steps_reverse):
        point = move_points(point, (start_index-i), ds, reverse_direction=True)
        points_rev = np.vstack((points_rev, point))
    
    points = np.vstack((points_rev[::-1], points_f))
    
    # Initialize list for selection of dfs for each point
    point_vals = []
    progress_bar = tqdm(range(first_step, final_step), desc="Tracking Parcel")
    
    for i, time_index in enumerate(range(first_step, final_step)):
        x, y = points[i]
        
        parcel = ds.sel(x=slice(x-buffer, x+buffer), y=slice(y-buffer, y+buffer)).mean(['x','y'], skipna=True)
        point_vals.append(parcel.isel(mid_date=time_index))
        progress_bar.update(1)

    progress_bar.close()
    point_srs = xr.concat(point_vals, dim='mid_date')
    strain = point_srs[['effective', 'eps_xx', 'eps_yy']].cumsum(dim='mid_date')
    strain = strain.rename_vars({'effective':'effective_strain','eps_xx':'e_xx', 'eps_yy':'e_yy'})
    
    parcel = xr.merge([point_srs, strain])

    if filtersize:
        parcel = apply_med_filt(parcel, filtersize)

    return parcel, points

##################################################################################

def move_points(points, index, ds, reverse_direction=False, snap=False):
    """
    Adjusts the positions of coordinate points based on velocity fields.

    This function moves a set of (x, y) coordinate pairs by applying velocity components (`vx`, `vy`) 
    from a dataset at a specific time step. Optionally, points can be snapped to the nearest grid 
    locations, and duplicates removed if snapping is enabled.

    Parameters
    ----------
    points : np.ndarray
        A 2D array of shape (n, 2) where each row represents an (x, y) coordinate pair.
    index : int
        Time step index to use for velocity data.
    ds : xr.Dataset
        The dataset containing `vx` and `vy` variables.
    reverse_direction : bool, optional
        If True, moves points in the reverse direction by negating the velocities. Default is False.
    snap : bool, optional
        If True, snaps points to the nearest grid center based on the dataset's resolution and removes duplicates. 
        Default is False.

    Returns
    -------
    np.ndarray
        Updated array of (x, y) coordinate pairs after applying the velocity adjustments.

    Notes
    -----
    - Velocities are scaled by a factor of 1/12 to approximate monthly motion.
    - Snapping aligns points to the dataset's grid spacing and origin.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> points = np.array([[100, 200], [150, 250]])
    >>> dataset = xr.open_dataset("velocity_data.nc")
    >>> updated_points = move_points(points, index=0, ds=dataset, reverse_direction=True, snap=True)
    >>> print(updated_points)
    """
    # Determine x and y grid spacing
    x_spacing = np.abs(np.diff(ds.x.values).mean())
    y_spacing = np.abs(np.diff(ds.y.values).mean())

    # Determine x and y offsets (origin alignment)
    x_offset = ds.x.values.min() % x_spacing
    y_offset = ds.y.values.min() % y_spacing

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

        # Snap points to the nearest grid center if snapping is enabled
        if snap:
            x = round((x - x_offset) / x_spacing) * x_spacing + x_offset
            y = round((y - y_offset) / y_spacing) * y_spacing + y_offset

        updated_points[i] = [x, y]

    # Remove duplicates if snapping is enabled
    # return np.unique(updated_points, axis=0) if snap else updated_points
    return updated_points

##################################################################################

def apply_med_filt(arr, size=3):
    return xr.apply_ufunc(
        ndimage.median_filter,
        arr,
        kwargs={'size':size}
    )

##################################################################################

def plot_arrows(xs, ys, ax):
    '''
    Plotting function that plots the direction arrows for the parcel location
    produced by bit.parcel_strain_stress().
    '''
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Sample points for arrows (e.g., every 10th point)
    arrow_indices = np.arange(2, len(xs)-1, 5)  # Avoid including the last index
    x_arrows = xs[arrow_indices]
    y_arrows = ys[arrow_indices]
    # Compute direction vectors for arrows (using np.diff to calculate directional vectors)
    dx = np.diff(xs)  # Differences in x
    dy = np.diff(ys)  # Differences in y
    directions = np.sqrt(dx**2 + dy**2)  # Magnitudes of direction vectors
    
    # Normalize the direction vectors
    dx = dx / directions
    dy = dy / directions
    
    # Align direction vectors with sampled arrow positions
    dx_arrows = dx[arrow_indices]  # Aligning with arrow positions
    dy_arrows = dy[arrow_indices]  # Aligning with arrow positions
    
    # Plot the line
    ax.plot(xs, ys, color='white', label='Parcel path')
    
    # Add arrows using quiver
    ax.quiver(x_arrows, y_arrows, dx_arrows, dy_arrows, 
               angles='uv', scale_units='xy', width=.05, color='white')

##################################################################################

def rewrite_nc(
    ds: xr.Dataset, 
    gdf, 
    fname
):
    '''
    Rewrites the tiles from trystan to follow the mid_date, y, x 
    format of the ITSLIVE data
    Parameters:
    -----------
    ds:
        opened netCDF file from trys. open using xr.open_dataset

    gdf:
        geodataframe of the area to clip maps to. 

    fname:
        filename and pathway to save as a new .nc file
    '''
    
    ds = ds.assign_coords({'crs':'crs'})
    
    arrays = []
    dates = []
    for varname, da in ds.data_vars.items():
        arrays.append(da.values)
        var = varname.split('_')[-1]
        date = var[0:4] + '-' + var[4:6] + '-' + var[6:]
        dates.append(date)
    
    new_array = np.array(arrays)
    mid_date = np.array(dates, 'datetime64[ns]')
    
    x = ds.x.values
    y = ds.y.values
    
    data_vars = {'fracture_conf':(('mid_date', 'y', 'x'), new_array)}
    coords = {'mid_date':mid_date, 'y':y, 'x':x}
    
    new_ds = xr.Dataset(
        data_vars = data_vars,
        coords = coords
    )
    new_ds = new_ds.sortby('mid_date')
    
    new_ds.rio.write_crs('EPSG:3031', inplace=True)
    new_ds_clip = new_ds.rio.clip(gdf.geometry, gdf.crs)
    new_ds_clip.to_netcdf(fname)

##################################################################################
# end of file