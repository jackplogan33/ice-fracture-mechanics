import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from functools import partial

from shapely.geometry import Polygon
from scipy import ndimage
import shapely.geometry

import matplotlib.pyplot as plt
from types import NoneType
import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})

import itslivetools

##################################################################################

def get_urls(
    area: gpd.GeoDataFrame | tuple | list = None,  
    epsg: int=3031
) -> list:
    '''
    Given a GDF area or a point in EPSG:4326, return URL or list of URLs
    that cover the area or point of areas.

    Parameters
    ----------
    area (gpd.GeoDataFrame | tuple | list, optional):
        GeoDataFrame of the reference area, a singular point as a list or tuple,
        or a list or tuple of points. Points should be passed as (lon, lat).
        If none are passed, all URLs from the specified EPSG number will be returned
    epsg (int, optional):
        If area is GeoDataFrame, value must be same as the CRS of the gdf. 
        If area is none, returns all URLs from the EPSG number passed.
        Default: 3031

    Returns
    -------
    urls:
        list of data cube URLs from ITS_LIVE catalog
    '''
    # sanitize inputs, raise errors: 
    assert isinstance(area, (tuple, list, gpd.GeoDataFrame, NoneType)), "area must be a tuple, list, or GeoDataFrame"
    assert isinstance(epsg, int), "epsg must be and integer value"
    
    if isinstance(area, gpd.GeoDataFrame):
        # If the area is a geodataframe, point var is false
        point = False
        gdf = True
        # ensures area is in proper CRS
        area = area.to_crs(epsg)

    elif isinstance(area, (list, tuple)):
        # if the area is a list or tuple, point var is true
        point = True

        # Check for multiple points by  seeing if first entry is number or tuple/list
        # also check for proper length of points        
        if all(isinstance(entry, (int, float)) for entry in area) & (len(area) == 2):
            # singular point, set multi as false
            multi = False
        
        elif all(isinstance(entry, (list, tuple)) for entry in area):
            assert all((len(entry) == 2) for entry in area), \
                "To pass multiple points, format area such that ((lon,lat), (lon,lat), ...)"
            
            # multiple points, set multi as true
            multi = True
        else:
            # point passed is not valid, raise error
            raise ValueError(
                "Point not in correct format. Must be passed as (lon, lat), or ((lon,lat),(lon,lat), ...)"
            )
    else:
        gdf = False
        point = False

    # if the area passed is a point
    if point:
        # 
        if multi:
            urls = [itslivetools.find_granule_by_point(pt) for pt in area]

        else:
            # use itslivetools 
            urls = [itslivetools.find_granule_by_point(area)]

        return urls
    
    # is area passed is a GeoDataFrame
    elif gdf:
        # Read in ITSLIVE json catalog, and write to crs
        catalog = gpd.read_file('https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json')
        catalog = catalog.to_crs(f'EPSG:{epsg}')

        # Use a spatial join to match URLs of the catalog to the shapes in the shapefile
        catalog_sub = gpd.sjoin(area, catalog, how='inner')
    
        # Get list of unique datacube urls
        urls = catalog_sub.drop_duplicates('zarr_url').zarr_url.values.tolist()
        return urls

    else:
        catalog = gpd.read_file('https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json')
        catalog_sub = catalog[catalog['epsg'] == epsg]
        urls = catalog_sub.drop_duplicates('zarr_url').zarr_url.values.tolist()
        return urls

##################################################################################

def get_data_cube(
    shape: gpd.GeoDataFrame = None, 
    urls: tuple | list | np.ndarray = None, 
    epsg: int = 3031,
    dt_delta: int = 19,
    engine: str = 'zarr'
):
    '''
    Given a geometry vector, will call the get_urls function to download
    ITSLIVE velocity zarr cubes into xarray, clip each cube to the passed geometry,
    concatenate into one datacube, and resample by month.
    
    If given a list of urls, get_urls() is not called.

    Parameters:
    -----------
    shape (gpd.GeoDataFrame, optional):
        GeoDataFrame of shape to clip the datacubes. If not passed, datacubes will be returned in full
    urls (tuple | list | np.ndarray, optional):
        If none, calls get_urls function for the shapefile/EPSG. If urls are passed, 
        skips function and opens files from given list. Must all be of same engine.
        Defaults to None
    epsg (int, optional):
        epsg number of the CRS used in shape
    dt_delta (int, optional) :
        Maximum time difference, in days, between image pairs. Equivalent to the time widget on ITS_LIVE.
        Default to 19
    Returns:
    --------
    dc ()
    '''
    # Type errors
    assert isinstance(shape, (gpd.GeoDataFrame, NoneType)), "Shape must be type gpd.GeoDataFrame or be None"
    assert isinstance(epsg, int), "EPSG must be of type int"
    assert isinstance(dt_delta, int), "dt_delta must be an int value"
    assert isinstance(engine, str), "engine must be of type str"
    
    if isinstance(shape, gpd.GeoDataFrame):
        assert shape.crs, "shape must have a CRS attribute"
        gdf = True

    else:
        gdf = False
    
    # if not given urls, get urls
    if urls == None:
        urls = get_urls(shape, epsg=epsg)
        
    # if given urls, check type
    else:
        assert isinstance(urls, (tuple, list, np.ndarray)), "URLs must be passed as list, tuple, or array"
        assert all(isinstance(url, str) for url in urls), "Each item in urls must be a string"
        
    # chunks to pass to
    chunks = {'mid_date':-1, 'x':'auto', 'y':'auto'}
    
    # Use partial function to pass shape to the preprocessing function
    # From xr.open_mfdataset documentation
    preprocess = partial(_preprocess, shape=shape, epsg=epsg, gdf=gdf, dt_delta=dt_delta)

    # opens multiple datasets. Chunks time=-1 to make sortby easier
    dc = xr.open_mfdataset(urls, engine=engine, preprocess=preprocess, 
                           chunks=chunks, combine='nested', concat_dim='mid_date').chunk(mid_date=-1)
    
    # sort by mid_date, resample by month, taking mean per month
    dc = dc.sortby('mid_date').resample(mid_date='1ME').mean(dim='mid_date', skipna=True).chunk('auto')
    return dc

##################################################################################

def _preprocess(
    ds: xr.Dataset,
    shape: gpd.GeoDataFrame, 
    epsg: str, 
    gdf: bool, 
    dt_delta: int
):
    '''
    Preprocessing function for xr.open_mfdataset is as follows:
        - Refines datasets to only the x- and y-velocities, and their corresponding errors.
        - Shortens timespan to 2015-present, where coverage is more consistent
        - Clip data to blue ice regions using rioxarray
        - Sorts datasets by mid_date because the ITSLIVE data cubes are out of order, which
        allows open_mfdataset to concatenate along time dimension. 
    '''
    # Convert time in days to nanoseconds
    ns = dt_delta * 86400000000000
    
    # Data variables to be used in blue ice zone calculations
    T = np.array(ns, dtype='timedelta64[ns]')
    
    # clip to images that have a time delta less than 19 days
    ds = ds[['vx','vy','v']].where(ds.date_dt <= T)

    if gdf:
        # Write CRS to data and clip to blue ice regions geometry
        ds = ds.rio.write_crs(f'EPSG:{epsg}')
        ds = ds.rio.clip(shape.geometry, shape.crs)

    # Sort by time variable
    ds = ds.sortby('mid_date')
    return ds

##################################################################################

def compute_strain_stress(
    vx: xr.DataArray | np.ndarray,
    vy: xr.DataArray | np.ndarray,
    dx: int = 120,
    dy: int = 120,
    axis: tuple = (1,2),
    rotate: bool = False,
) -> xr.DataArray | np.ndarray:
    '''
    Calculate the strain rate and Von Mises stress of the given velocities. 

    Parameters:
    -----------
    vx (xr.DataArray | np.ndarray):
        velocity in the x-direction
    vy (xr.DataArray | np.ndarray):
        velocity in the y-direction
    dx (int, optional):
        pixel length in m. Only needs to be passed if using numpy
        Default to 120m
    dy (int, optional):
        pixel length in m. Only needs to be passed if using numpy
        Default to 120m
    axis (tuple, optional):
        specifies the axis to take the gradient on. Defaults to (1,2), meaning (x, y)
    rotate (bool, optional):
        rotate the strain rate components to along flow, accross flow, and shear. Default to False

    Returns:
    --------
    tuple of np.ndarray | xr.DataArray:
        returns dataarray or numpy arrays of computed variables 
        Returns in order: ('eps_eff', 'eps_xx', 'eps_yy', 'sigma_vm', 'sigma1', 'sigma2')
    '''
    # Sanitize inputs, raise value errors
    assert (type(vx) == type(vy)), "Input velocities must be same data type"
    assert isinstance(vx, (xr.DataArray, np.ndarray)), "Input velocities must be numpy arrays or xarray DataArrays"
    assert isinstance(dx, int) & isinstance(dy, int), "dx and dy must be integers"
    assert isinstance(axis, (tuple)), 'axis must be a tuple'
    assert isinstance(rotate, bool), 'rotate must be a boolean'

    xarray = False
    if isinstance(vx, xr.DataArray):
        xarray = True
    
    # use xarray to compute gradients
    if xarray:
        # xarray takes the coord size into account 
        # no need to include dy | dx
        du_dx = vx.differentiate('x')
        du_dy = vx.differentiate('y')
        dv_dx = vy.differentiate('x')
        dv_dy = vy.differentiate('y')

    # use numpy to compute gradients
    else:
        # Calculate the gradients of the x and y velocities
        du_dx, du_dy = np.gradient(vx, dx, dy, axis=axis)
        dv_dx, dv_dy = np.gradient(vy, dx, dy, axis=axis)

    # strain rate tensor
    eps_xy = 0.5 * (dv_dx + du_dy)
    eps_xx = du_dx
    eps_yy = dv_dy

    # rotate the directions if passed
    if rotate:
        # Angle of rotation
        theta = np.arctan2(vy, vx)
        
        # use alley et al equations to rotate strain rates
        # longitudinal, transverse, shear
        eps_lon = (
            (eps_xx * (np.cos(theta) ** 2)) + 
            (2 * eps_xy * np.cos(theta) * np.sin(theta)) + 
            (eps_yy * (np.sin(theta) ** 2))
        )
        eps_trn = (
            (eps_xx * (np.sin(theta) ** 2)) - 
            (2 * eps_xy * np.cos(theta) * np.sin(theta)) + 
            (eps_yy * (np.cos(theta) ** 2))
        )
        eps_shr = (
            (eps_yy - eps_xx) * np.cos(theta) * np.sin(theta) + 
            eps_xy * ((np.cos(theta) ** 2) - (np.sin(theta) ** 2))
        )

        # Save into corresponding variables
        eps_xx = eps_lon
        eps_yy = eps_trn
        eps_xy = eps_shr
    # Calculate effective strain rate
    eps_eff = np.sqrt(0.5 * (eps_xx ** 2 + eps_yy ** 2) + eps_xy ** 2)

    # Call fcn that computes stress tensor
    sigma_vm, sigma1, sigma2 = _strain_stress(eps_eff, eps_xx, eps_yy)

    return eps_eff, eps_xx, eps_yy, sigma_vm, sigma1, sigma2

##################################################################################

def _strain_stress(
    e_eff: np.ndarray | xr.DataArray,
    e_xx: np.ndarray | xr.DataArray,
    e_yy: np.ndarray | xr.DataArray, 
    A: float = 3.5e-25
) -> np.ndarray | xr.DataArray:
    '''
    Returns Von Mises stress given x and y stress 
    values following Vaughan 1993.
    
    Parameters:
    -----------
    e_eff (np.ndarray | xr.DataArray):
        Effective strain rate
    e_xx (np.ndarray | xr.DataArray):
        Array of strain rate in x direction (longitudinal) 
    e_yy (np.ndarray | xr.DataArray):
        Array of strain rate in y direction (transverse)
    A (float):
        Coefficient 'A' for Glen's Flow Law. Defaults to 3.5e-25

    Returns:
    --------
    sigma_vm (np.ndarray | xr.DataArray):
        The Von Mises tensile stress in kPa
    sigma1 (np.ndarray | xr.DataArray):
        Principle stress 1 in kPa
    sigma2 (np.ndarray | xr.DataArray):
        Principle stress 2 in kPa
    '''
    # Canonical exponent of 3
    n = 3

    # Use Glens Flow law to relate strain rate to stress (following Grinsted)    
    # exponent variable
    exp = ((1 - n) / n)
    A *= (365 * 24 * 3600)    # Units converter from 1/yr to 1/s
    
    tau_xx = (A ** (-1 / n)) * (e_eff ** exp) * e_xx
    tau_yy = (A ** (-1 / n)) * (e_eff ** exp) * e_yy

    # Convert from tau to sigma, principle stresses
    sigma1 = (2 * tau_xx) + tau_yy
    sigma2 = tau_xx + (2 * tau_yy)

    # Compute Von Mises Stress following Vaughan et al
    sigma_vm = np.sqrt((sigma1 ** 2) + (sigma2 ** 2) - (sigma1 * sigma2))

    return sigma_vm / 1000, sigma1 / 1000, sigma2 / 1000

##################################################################################

def lagrangian_frame(
    ds: xr.Dataset, 
    geometry: shapely.geometry.polygon.Polygon, 
    start_index: int = 0,
    steps_forward: int = None,
    steps_reverse: int = None,
    epsg: int = 3031,
    filtersize: int = 2,
    remove_threshold = None
) -> xr.Dataset:
    '''
    Given an xr.Dataset with velocity components and a shapely Polygon, 
    returns clipped dataset tracking the feature as it moves through time. 

    Parameters:
    -----------
    ds (xr.Dataset):
        dataset containing velocity components
    geometry (shapely Polygon):
        polgyon outline of the feature to be tracked
    start (int, optional):
        Optional parameter of the starting index. If passed, start from that index.
        Default: 0
    reversed (bool):
        calculates original starting place of polygon, then steps forward
    steps (int, optional):
        number of time frames to track the feature. If none, 
        covers all time steps in dataset
    filtersize (int, optional):
        size of window in median filter
        Defaults to 2
    remove_threshold (float, optional):
        if you would like to remove values based on the fracture confidence variable,
        specify the threshold to remove from the next frame

    Returns:
    --------
    lagrange_frame (xr.Dataset):
        dataset that follows on the passed polygon as it moves through time
    '''
    # defensive programming
    assert isinstance(ds, xr.Dataset), 'ds must be xr.Dataset'
    assert isinstance(geometry, shapely.geometry.polygon.Polygon), \
        'geometry must be a shapely polygon'
    assert isinstance(epsg, int), 'EPSG must be an integer'
    assert isinstance(start_index, int), 'Start index must be an Integer'
    assert isinstance(steps_forward, (NoneType, int)), \
        'Number of steps forward must be None or integer'
    assert isinstance(steps_reverse, (NoneType, int)), \
        'Number of steps in reverse must be None or int'
    assert isinstance(filtersize, int), 'filter size must be an integer'
    assert isinstance(remove_threshold, (float, NoneType)), \
        'remove_threshold must be float or None'
    
    # variable initialization
    # If steps not passed, set end index to last timestep
    if isinstance(steps_forward, NoneType):
        end = len(ds.mid_date)
        
    # If steps passed, set end index to starting index + num of steps
    else:      
        end = start_index + steps_forward
    
    if end > len(ds.mid_date):
        end = len(ds.mid_date)
    
    # get list of points from polygon
    xs, ys = np.array(geometry.exterior.coords).T
        
    # If reversed steps given, move polygon backwards to beginning timestep
    if isinstance(steps_reverse, int): 
        for i in range(steps_reverse):
            move_points(xs, ys, (start_index-i), ds, reversed=True)
            
        # Create original polygon, subtract reversed indexes from starting index
        geometry = Polygon(np.array([xs, ys]).T)
        start_index -= steps_reverse
    
    # Set remove to False if None, True if value passed
    remove = False if isinstance(remove_threshold, NoneType) else True
    
    # Clip original area from first time slice, apply median filter
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=f'EPSG:{epsg}')
    first_frame = ds.isel(mid_date=start_index).rio.clip(gdf.geometry, gdf.crs, all_touched=True)
    first_frame = apply_med_filt(first_frame, filtersize)
    
    # initialize list of frames, add first frame to that
    clipped_frames = [first_frame]
    
    # If removing values, get all points above threshold
    if remove:
        fracture_clip = first_frame.where(first_frame.fracture_conf > remove_threshold)
        
        # extract points to x and y arrays
        nan_pts = fracture_clip.fracture_conf.stack(points=['x','y'])
        xs_f, ys_f = nan_pts[nan_pts.notnull()].x.data, nan_pts[nan_pts.notnull()].y.data
    
    # iterate from start to end index, stepping in the direction specified
    for i in range(start_index, end):
        # call function to move each point in polygon
        move_points(xs, ys, i, ds)
        
        # Make a shapely Polgyon of the new points
        geometry = Polygon(np.array([xs, ys]).T)
        # make GDF of the polgyon
        gdf = gpd.GeoDataFrame(geometry=[geometry], crs=f'EPSG:{epsg}')
        
        # clip the next time slice to the moved polygon
        next_frame = ds.isel(mid_date=(i+1)).rio.clip(gdf.geometry, gdf.crs, all_touched=True)
        next_frame = apply_med_filt(next_frame, filtersize)
            
        if remove:
            # call function to move points of fracture
            move_points(xs_f, ys_f, i, ds)
            
            # get all points over and 85% fracture confidence from new frame
            # must be done before masking with NaN's, or values get lost
            fracture_clip = next_frame.where(next_frame.fracture_conf > remove_threshold)
        
            for xval, yval in zip(xs_f, ys_f):
                nearest_point = next_frame.sel(x=xval, y=yval, method='nearest')
                next_frame.loc[{'x': nearest_point.x.data, 'y': nearest_point.y.data}] = np.nan
        
            # extract fractured points to a list
            nan_pts = fracture_clip.fracture_conf.stack(points=['x','y'])
            xs_f, ys_f = nan_pts[nan_pts.notnull()].x.data, nan_pts[nan_pts.notnull()].y.data
    
        clipped_frames.append(next_frame)
        
    lagrange_frame = xr.concat(clipped_frames, dim='mid_date')
    return lagrange_frame

##################################################################################

def parcel_strain_stress(
    ds: xr.Dataset, 
    x: (int | float),
    y: (int | float),
    buffer: (int | float) = 0,
    start_index: '(int, optional)' = None,
    steps_forward: '(int, optional)' = None, 
    steps_reverse: '(int, optional)' = None
) -> xr.Dataset:
    '''
    Returns the time series values of a single point as it moves
    across the domain with time. It integrates the strain rate to
    calculate the strain of a parcel and adds as new variables to the dataset

    Parameters:
    -----------
    ds : xr.Dataset
        xarray dataset to be passed to move_points function. Must have x- and y-velocity,
        effective, x, and y strain rates.
    x : (int, float)
        Integer or float value of the x-coordinate of the point of interest
    y : (int, float)
        integer or float value of the x-coordinate of the point of intetest
    buffer : (int, float, optional)
        integer or float amount of distance to average around the point of interest
    start_index : (int, optional)
        Time index of the x- and y-coordinates given. If none passed, start index of 0 is assumed.
        Gives flexibility of where and when a point can be tracked
    steps_forward : (int, optional)
        Number of timesteps to take in the forward time direction. If none passed, will step as many indexes
        are available in the time index
    steps_reverse : (int, optional)
        Number of timesteps to take in the reverse time direction. If non passed, 0 is assumed.
        Allows the tracking of points identfied as crevassed in the middle of a time series before they broke.

    Returns:
    --------
    parcel (ds) :
        xarray dataset with time variable remaining. Appends total effective, x-, and y-direction strains
    xs (list) :
        list of x-coordinates in order from the first to last timestep
    ys (list) : 
        list of y-coordinates in order from the first to last timestep
    '''
    assert isinstance(ds, xr.Dataset), 'da must be a DataArray'
    assert type(x) == type(y), 'x and y must be the same type'
    assert isinstance(x, (int, float)), 'x- and y-points must be int or float'
    assert isinstance(start_index, (int, NoneType)), 'Starting index must be an int or None'
    assert isinstance(steps_forward, (int, NoneType)), 'Steps forward must be an int or None'
    assert isinstance(steps_forward, (int, NoneType)), 'Steps reverse must be an int or None'
    assert isinstance(buffer, int)
    
    # Initialize variable, ensure indexing works
    # Grab length of time index
    length = len(ds.mid_date)
    
    # If start index not given, 
    if isinstance(start_index, NoneType):
        start_index = 0
        steps_reverse = 0
    
    else:
        # If start index given, check that the index is less than number of timesteps
        assert start_index < length, \
            "Starting index must be at least 1 less than the mid_date length"
    
    # If steps forward not given, go as many steps forward as possible
    if isinstance(steps_forward, NoneType):
        steps_forward = length - start_index
        
    else:
        final_step = steps_forward + start_index
        # If steps forward given
        assert final_step <= len(ds.mid_date), \
            "Too many steps forward. Decrease the number of timesteps"
    
    # If no steps reversed given, set to 0
    if isinstance(steps_reverse, NoneType):
        steps_reverse = 0
    
    else:
        first_step = start_index - steps_reverse
        assert first_step >= 0, "Too many steps reverse. Decrease the number of timesteps"
    
    # Define start and end index
    first_step = start_index - steps_reverse
    final_step = start_index + steps_forward
    
    # Define x- and y-arrays for move_points func
    move_x = [x]
    move_y = [y]
    
    # Add initial pt to forward step
    x_forward = [move_x[0]]
    y_forward = [move_y[0]]
    for i in range(steps_forward):
        move_points(move_x, move_y, (start_index+i), ds)
        x_forward.append(move_x[0])
        y_forward.append(move_y[0])
    
    move_x = [x]
    move_y = [y]
    
    # Initialize list for points in reverse
    x_reverse = []
    y_reverse = []
    for i in range(steps_reverse):
        move_points(move_x, move_y, (start_index-i), ds, reversed=True)
        x_reverse.append(move_x[0])
        y_reverse.append(move_y[0])
    
    # Make full list of x values in increasing order
    xs = x_reverse[::-1] + x_forward
    ys = y_reverse[::-1] + y_forward
    
    # Initialize list for selection of dfs for each point
    point_vals = []
    
    for i, time_index in enumerate(range(first_step, final_step)):
        xx = xs[i]
        yy = ys[i]
        
        parcel = ds.sel(x=slice(xx-buffer, xx+buffer), y=slice(yy-buffer, yy+buffer)).mean(['x','y'], skipna=True)
        point_vals.append(parcel.isel(mid_date=time_index))
    
    point_srs = xr.concat(point_vals, dim='mid_date')
    strain = point_srs[['eps_eff', 'eps_xx', 'eps_yy']].cumsum(dim='mid_date')
    strain = strain.rename_vars({'eps_eff':'e_eff', 'eps_xx':'e_xx', 'eps_yy':'e_yy'})
        
    parcel = xr.merge([point_srs, strain])

    return parcel, xs, ys

##################################################################################

def move_points(
    xs: (list | np.ndarray),
    ys: (list | np.ndarray), 
    index: (int), 
    ds: xr.Dataset,
    reversed: bool = False
):
    '''
    Given x- and y- coordinates, a time index, and a dataset with
    x- and y-velocities moves the points to the location they would be
    after one month. Velocity in m/yr, coordinates in meters.
    '''
    assert type(xs) == type(ys), 'xs and ys must have the same type'
    assert isinstance(xs, (list, np.ndarray)), 'xs and ys must be a list or numpy array'
    assert len(xs) == len(ys), 'Number of x and y coordinates should be equal'
    assert all([isinstance(x, (int, float)) for x in xs]), 'all xs must be int or float values'
    assert all([isinstance(y, (int, float)) for y in ys]), 'all ys must be int or float values'
    assert isinstance(index, int), 'index must be and integer'
    assert isinstance(ds, xr.Dataset), 'ds must be an xarray Dataarray'
    assert isinstance(reversed, bool), 'reversed must be a boolean'
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        # if REVERSE:
        # Get velocities from previous timestep
        # make negative
        if reversed:
            vx = ds.vx[index-1].sel(x=x, y=y, method='nearest').data
            vy = ds.vy[index-1].sel(x=x, y=y, method='nearest').data
            vx = vx * -1
            vy = vy * -1
        
        # Get vels from current timestep
        else:
            # get x and y velocity for the coordinate pt
            vx = ds.vx[index].sel(x=x, y=y, method='nearest').data
            vy = ds.vy[index].sel(x=x, y=y, method='nearest').data
            
        if np.isnan(vx):
            vx = 0
        if np.isnan(vy):
            vy = 0
        
        # add amt of distance travelled over the month
        xs[i] += (vx / 12)
        ys[i] += (vy / 12)

##################################################################################

def apply_med_filt(arr, size=3):
    return xr.apply_ufunc(
        ndimage.median_filter,
        arr,
        kwargs={'size':size}
    )

##################################################################################

def plotting_stress(
    ds: xr.Dataset,
    data_vars: list | tuple,
    name: str = None, 
    area: list | tuple = None,
    selected: list | tuple = None,
    figsize: tuple = (20,9),
    vmax: int | float = None
):
    '''
    Takes Xarray datacube and the gpd vector and plots the stress and strain rates of the glacier
    Has an outline of the body 

    Parameters:
    -----------
    ds (xarray.Dataset):
        xarray dataset.

    data_vars (list | tuple):
        names of the variables to plot a raster of, side by side

    name (str, optional):
        Name of the region being mapped. Defaults to None
        
    area (list | tuple, optional):
        A point or rectangle to plot a time series over. 
        If a single point is wanted: (x, y)
        Is a rectangle is wanted: (x1, x2, y1, y2)

    selected (list | tuple, optional):
        If an area is passed, variable or pair of variables to plot over time.
        Area and selected must be passed in tandem.
    
    figsize
        passed to plt.subplots as the figsize for the plots.
    '''
    
    # sanitize inputs, defensive programming
    assert isinstance(ds, xr.Dataset), "ds must be of type xr.Dataset"

    assert isinstance(data_vars, (tuple, list)), "data_vars must be passed in a tuple or list"

    assert set(data_vars).issubset(ds), "data_vars must be variables in dataset" 

    ncols = len(data_vars)
    
    time_series = False
    selection = None
    if isinstance(area, (tuple, list)):
        ncols += 1
        time_series = True
        assert (len(area) == 2) | (len(area) == 2), "Area must be (x, y) or (x1, x2, y1, y2)"
        assert isinstance(selected, (tuple, list)), "If an area is passed, 'selected' variables must be passed, too"
        assert set(selected).issubset(ds), "Selected variables must be in the dataset"
        assert ((len(selected) == 1) | (len(selected) == 2)), "Can only select 1 or 2 variables"
        
        if len(area) == 4:
            box = True
            x1, x2, y1, y2 = area
            x = [x1, x2]
            y = [y1, y2]
            ds_sub = ds.sel(x=x, y=y, method='nearest').mean(dim=['x', 'y'], skipna=True)
        else:
            box = False
            x, y = area
            ds_sub = ds.sel(x=x, y=y, method='nearest')

        selection = ds_sub[selected]

    time_avg = ds[data_vars].mean(dim='mid_date', skipna=True)
    
    fig, axs = _fixed(time_avg, figsize, ncols, name, data_vars, time_series, selection=selection, vmax=vmax)

    if time_series:
        axs = axs[1:]
        if box:
            for ax in axs:
                ax.fill([x1,x2,x2,x1], [y1,y1,y2,y2], facecolor='red', edgecolor='k', alpha=.5)
                ax.fill([x1,x2,x2,x1], [y1,y1,y2,y2], facecolor='red', edgecolor='k', alpha=.5)
            plt.show()
        else:
            for ax in axs:
                ax.axvline(x=x, c='blue')
                ax.axhline(y=y, c='blue')

##################################################################################

def _fixed(time_avg, figsize, ncols, name, data_vars, time_series, selection=None, vmax=None):
    # time_avg = ds[data_vars].mean(dim='mid_date', skipna=True, keep_attrs=True)
    
    if isinstance(name, str):
        name = ' on ' + name
    
    else:
        name = ''
    
    fig, axs = plt.subplots(ncols=ncols, figsize=figsize, layout='constrained')
    
    if time_series:
        vars = list(selection.data_vars)
        if len(vars) == 1:
            selection[f'{vars[0]}'].plot(ax=axs[0])
            axs[0].set_title('Tensile Stress at Point')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Tensile Strength (kPa)')

        else:
            ax1 = axs[0]
            selection[f'{vars[0]}'].plot(ax=ax1, x='mid_date')
            # Plots on same axis
            ax2 = ax1.twinx()
            selection[f'{vars[1]}'].plot(ax=ax2, color='red')
    
    for i, variable in enumerate(data_vars):
        ax = axs[-(i+1)]
    
        a = time_avg[variable].plot(ax=ax)
        a.colorbar.set_label(f"{time_avg[variable].attrs['units']}")
    
        ax.set_aspect('equal')
    
    # Polishing the figure
    fig.suptitle(f'Tensile Stress and Strain Rate{name}', fontsize=20)
    return fig, axs

##################################################################################

def plot_monthly_failure_tau(ds, figsize, fname=None):
    for i in range(len(ds.mid_date)):
        fig, axs = plt.subplots(ncols=4, figsize=figsize, layout='constrained')
        
        mu = 0.03
        x = ((2 * ds.tau_xx[i]) + ds.tau_yy[i]) * (1 - mu)
        y = ((2 * ds.tau_yy[i]) + ds.tau_xx[i]) * (1 + mu)
        c = ds.fracture_conf[i]
        
        axs[0].scatter(x=x, y=y, c=c, cmap='viridis', vmax=1)
        axs[0].set_xlim([-1500,1500])
        axs[0].set_xlabel('Principle stress 1 [kPa]')
        axs[0].set_ylim([-1500,1500])
        axs[0].set_ylabel('Principle stress 2 [kPa]')
        axs[0].hlines(0, xmin=-1500, xmax=1500, color='black')
        axs[0].vlines(0, ymin=-1500, ymax=1500, color='black')
        axs[0].grid()
        
        ds.tau_xx[i].plot(ax=axs[1], vmin=-750, vmax=750, cmap='RdBu_r')
        ds.tau_yy[i].plot(ax=axs[2], vmin=-750, vmax=750, cmap='RdBu_r')
        ds.fracture_conf[i].plot(ax=axs[3], vmax=1)
        
        titles = ['Failure Map', 'Principle Stress 1', 'Principle Stress 2', 'Fracture Confidence']
        for ax, title in zip(axs, titles):
            ax.set_aspect('equal')
            ax.set_title(title)

        plt.suptitle(f'{ds.mid_date[i].data}')

        if isinstance(fname, str):
            plt.savefig(f'../figures/gif-pngs/{fname}-{i}.png', bbox_inches='tight')
        
        plt.show()

    if isinstance(fname, str):
        images = []
        for i in range(len(ds.mid_date)):
            image = Image.open(f'../figures/gif-pngs/{fname}-{i}.png')
            images.append(image)
        # Save as GIF with each frame
        imageio.mimsave(f'../figures/{fname}.gif', images, duration=750)

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

def plot_vel_arrows(ds, figsize=None):
    # Get absolute velocity
    mean = ds.mean(dim='mid_date', skipna=True)
    
    vx = mean.vx
    vy = mean.vy
    vv = np.sqrt(vx**2 + vy**2)
    # Plot velocity with imshow
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # Plot velocity field colourmap
    vv.plot(ax=ax, cmap="turbo", cbar_kwargs={'label':'Velocity [m a$^{-1}$]'})
    
    # # Account for 'strictly increasing' requirement for streamplots in newer matplotlibs:
    # # flip if Y is decreasing rather than increasing. This is only necessary for plotting
    # streamplots, not quiverplots.
    if vv.y.values[1] - vv.y.values[0] < 0:
        vv_flip = vv.reindex(y = vv.y[::-1])
        vx_flip = vx.reindex(y = vx.y[::-1])
        vy_flip = vy.reindex(y = vy.y[::-1])
    else: 
        vv_flip = vv
        vx_flip = vx
        vy_flip = vy
    
    
    # # Plot flow direction using streamplot
    plt.streamplot(
        vv_flip.x.values, 
        vv_flip.y.values, 
        vx_flip.values, 
        vy_flip.values,
        color='white',
        linewidth=0.6,
        density=1.2,
    )
    
    ax.ticklabel_format(scilimits=(6,6))
    ax.set_title(None)
    ax.set_aspect('equal')

##################################################################################
# end of file