import os
import subprocess
# use the LibraryRequirements.txt file (pip install -r LibraryRequirements.txt )
import h5py
import s3fs
import xarray as xr
import cfgrib  # Import the cfgrib engine
import numpy as np
import pandas as pd
from datetime import datetime
import cv2  # Used for image resizing
from tqdm import tqdm
import urllib.request  # Import the built-in library
import sys  # Import sys to allow exiting
import hashlib  # For caching HRRR filenames

# =============================================================================
# --- 1. CONFIGURATION: SET YOUR PATHS HERE ---
# =============================================================================

BASE_DRIVE_PATH = 'D:/'

# --- Project Paths ---
PROJECT_PATH = os.path.join(BASE_DRIVE_PATH, 'WeatherProject')
RAW_SEVIR_PATH = os.path.join(PROJECT_PATH, 'RawSEVIRData')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'ProcessedData')
CATALOG_PATH = os.path.join(PROJECT_PATH, 'CATALOG.csv')

# --- HRRR CACHE (reused files for speed) ---
HRRR_CACHE = os.path.join(PROJECT_PATH, 'HRRR_CACHE')


# =============================================================================
# --- Utilities ---
# =============================================================================

def download_catalog():
    """Downloads the SEVIR catalog if it doesn't exist."""
    if os.path.exists(CATALOG_PATH):
        if os.path.getsize(CATALOG_PATH) < 500000:  # <0.5MB is probably broken
            os.remove(CATALOG_PATH)
        else:
            print("CATALOG already exists.")
            return True

    print("Downloading SEVIR CATALOG.csv...")
    try:
        url = 'https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv'
        urllib.request.urlretrieve(url, CATALOG_PATH)
        print("Catalog downloaded.")
        return True
    except Exception as e:
        print("Error downloading CATALOG:", e)
        return False


def download_sevir_data_type(data_type):
    print(f"--- Downloading SEVIR {data_type} ---")
    s3_source_path = f's3://sevir/data/{data_type}'
    local_destination_path = os.path.join(RAW_SEVIR_PATH, data_type)
    os.makedirs(local_destination_path, exist_ok=True)

    command = [
        'aws', 's3', 'sync',
        s3_source_path,
        local_destination_path,
        '--no-sign-request'
    ]

    try:
        subprocess.run(command)
    except Exception as e:
        print("AWS S3 Sync Error:", e)


def build_hrrr_url(timestamp):
    dt = pd.to_datetime(timestamp)
    return (
        f"s3://noaa-hrrr-bdp-pds/hrrr.{dt:%Y%m%d}/conus/"
        f"hrrr.t{dt:%H}z.wrfsfcf00.grib2"
    )


def get_event_data(event_id, catalog):
    """Returns rows where VIL + IR107 timestamps match."""
    try:
        rows = catalog[catalog['event_id'] == event_id].copy()
        vil = rows[rows['img_type'] == 'vil'].copy()
        ir107 = rows[rows['img_type'] == 'ir107'].copy()

        if len(vil) == 0 or len(ir107) == 0:
            return None

        vil['time_utc'] = pd.to_datetime(vil['time_utc'])
        ir107['time_utc'] = pd.to_datetime(ir107['time_utc'])

        merged = pd.merge(
            vil,
            ir107,
            on='time_utc',
            suffixes=('_vil', '_ir107')
        ).sort_values('time_utc')

        if len(merged) == 0:
            return None

        return merged
    except:
        return None


# =============================================================================
# --- HRRR CACHE SYSTEM ---
# =============================================================================

def get_local_hrrr_path(hrrr_url, s3):
    """Downloads an HRRR file once and reuses it."""
    os.makedirs(HRRR_CACHE, exist_ok=True)

    # deterministic filename
    file_hash = hashlib.md5(hrrr_url.encode()).hexdigest()
    local_path = os.path.join(HRRR_CACHE, f"{file_hash}.grib2")

    if not os.path.exists(local_path):
        print(f"Downloading HRRR to cache: {hrrr_url}")
        try:
            s3.get(hrrr_url, local_path)
        except Exception as e:
            print("HRRR download failed:", e)
            return None

    return local_path


# =============================================================================
# --- HRRR VARIABLE HANDLING (PRMSL OPTIONAL) ---
# =============================================================================

def _get_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return ds[c]
    raise KeyError(f"No matching vars from {candidates}")


# =============================================================================
# --- Processing ---
# =============================================================================

def process_event(event_id, catalog, s3):
    output_path = os.path.join(OUTPUT_PATH, f"storm_{event_id}.npy")

    if os.path.exists(output_path):
        return

    merged = get_event_data(event_id, catalog)
    if merged is None:
        return

    # storm center comes from VIL box
    row0 = merged.iloc[0]
    center_lat = (row0['llcrnrlat_vil'] + row0['urcrnrlat_vil']) / 2
    center_lon = (row0['llcrnrlon_vil'] + row0['urcrnrlon_vil']) / 2

    # slices (HRRR lat is descending)
    lat_slice = slice(center_lat + 1.5, center_lat - 1.5)
    lon_slice = slice(center_lon - 1.5, center_lon + 1.5)

    event_frames = []

    for idx, row in merged.iterrows():

        # SEVIR paths
        path_vil = os.path.join(RAW_SEVIR_PATH, row['file_name_vil'])
        path_ir = os.path.join(RAW_SEVIR_PATH, row['file_name_ir107'])

        if not os.path.exists(path_vil) or not os.path.exists(path_ir):
            continue

        # load SEVIR data
        try:
            with h5py.File(path_vil, "r") as f:
                vil = np.squeeze(f['vil'][row['file_index_vil']])
            with h5py.File(path_ir, "r") as f:
                ir = np.squeeze(f['ir107'][row['file_index_ir107']])
        except:
            continue

        # --- HRRR ---
        hrrr_url = build_hrrr_url(row['time_utc'])
        local_hrrr = get_local_hrrr_path(hrrr_url, s3)
        if local_hrrr is None:
            continue

        try:
            ds_list = cfgrib.open_datasets(local_hrrr, backend_kwargs={'indexpath': ''})
        except Exception as e:
            print("cfgrib could not parse HRRR file:", e)
            continue

        # candidates
        t2m_c = ['t2m', '2t']
        cape_c = ['cape', 'capesfc']
        prmsl_c = ['prmsl', 'msl', 'mslet', 'MSLET_P0_L101_GLC0']

        ds_t2m = None
        ds_cape = None
        ds_prmsl = None

        for ds in ds_list:
            vars = ds.variables
            if any(v in vars for v in t2m_c):
                ds_t2m = ds
            elif any(v in vars for v in cape_c):
                ds_cape = ds
            elif any(v in vars for v in prmsl_c):
                ds_prmsl = ds  # optional

        # required vars
        if ds_t2m is None or ds_cape is None:
            continue

        # coords
        ds_t2m = ds_t2m.set_coords(['latitude', 'longitude'])
        ds_cape = ds_cape.set_coords(['latitude', 'longitude'])
        if ds_prmsl:
            ds_prmsl = ds_prmsl.set_coords(['latitude', 'longitude'])

        # extract T2M
        try:
            slc = ds_t2m.sel(latitude=lat_slice, longitude=lon_slice, heightAboveGround=2)
            t2m = np.squeeze(_get_var(slc, t2m_c).values)
        except:
            continue

        # extract CAPE
        try:
            slc = ds_cape.sel(latitude=lat_slice, longitude=lon_slice)
            cape = np.squeeze(_get_var(slc, cape_c).values)
        except:
            continue

        # extract PRMSL (optional)
        if ds_prmsl:
            try:
                slc = ds_prmsl.sel(latitude=lat_slice, longitude=lon_slice)
                prmsl = np.squeeze(_get_var(slc, prmsl_c).values)
            except:
                prmsl = np.zeros_like(cape)
        else:
            prmsl = np.zeros_like(cape)

        # resize HRRR → 384×384
        size = (384, 384)
        t2m = cv2.resize(t2m, size)
        cape = cv2.resize(cape, size)
        prmsl = cv2.resize(prmsl, size)

        # resize IR107
        ir = cv2.resize(ir, size)

        # stack
        frame = np.stack([vil, ir, t2m, prmsl, cape], axis=-1)
        event_frames.append(frame)

        # close GRIB datasets
        for ds in ds_list:
            ds.close()

    if len(event_frames) > 0:
        np.save(output_path, np.array(event_frames))


# =============================================================================
# --- MAIN ---
# =============================================================================

def main():
    print("Initializing paths...")
    os.makedirs(RAW_SEVIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(HRRR_CACHE, exist_ok=True)

    # Catalog
    if not download_catalog():
        print("Catalog download failed.")
        return

    catalog = pd.read_csv(CATALOG_PATH)
    required = ['event_id', 'img_type', 'time_utc',
                'file_name', 'file_index',
                'llcrnrlat', 'llcrnrlon', 'urcrnrlat', 'urcrnrlon']

    for r in required:
        if r not in catalog.columns:
            print("Missing required column:", r)
            return

    # Processing
    vil_events = set(catalog[catalog['img_type'] == 'vil']['event_id'])
    ir_events = set(catalog[catalog['img_type'] == 'ir107']['event_id'])
    events = sorted(list(vil_events.intersection(ir_events)))

    print(f"Processing {len(events)} events.")

    # s3 client
    s3 = s3fs.S3FileSystem(anon=True)

    for eid in tqdm(events, desc="Processing Events"):
        process_event(eid, catalog, s3)

    print("All events processed.")


if __name__ == "__main__":
    main()
