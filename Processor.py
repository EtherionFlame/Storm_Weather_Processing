# FULL FIXED VERSION OF YOUR SCRIPT
# Includes working HRRR GRIB loader using cfgrib group-merging

import os
import subprocess
import h5py
import s3fs
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from tqdm import tqdm
import urllib.request
import sys

xr.set_options(use_new_combine_kwarg_defaults=True)

BASE_DRIVE_PATH = 'D:/'
PROJECT_PATH = os.path.join(BASE_DRIVE_PATH, 'WeatherProject')
RAW_SEVIR_PATH = os.path.join(PROJECT_PATH, 'RawSEVIRData')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'ProcessedData')
CATALOG_PATH = os.path.join(PROJECT_PATH, 'CATALOG.csv')
TEMP_HRRR_PATH = os.path.join(PROJECT_PATH, 'temp_hrrr')

# ---------------- HRRR GRIB FIX ---------------- #
def open_hrrr_grib(path):
    groups = ["surface", "isobaricInhPa", "pressure", None]
    datasets = []

    for grp in groups:
        try:
            kwargs = {"indexpath": ""}
            if grp is not None:
                kwargs["filter_by_keys"] = {"typeOfLevel": grp}

            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs=kwargs)
            datasets.append(ds)
        except Exception:
            pass

    if not datasets:
        raise RuntimeError(f"No valid GRIB groups found in {path}")

    return xr.merge(datasets, compat="override")

# ------------------------------------------------ #

def download_catalog():
    if os.path.exists(CATALOG_PATH):
        if os.path.getsize(CATALOG_PATH) < 1000000:
            try:
                os.remove(CATALOG_PATH)
            except Exception as e:
                print(f"Could not delete file: {e}")
                return False
        else:
            return True

    try:
        url = 'https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv'
        urllib.request.urlretrieve(url, CATALOG_PATH)
        return True
    except Exception:
        return False

def build_hrrr_url(timestamp):
    dt = pd.to_datetime(timestamp)
    return f"s3://noaa-hrrr-bdp-pds/hrrr.{dt:%Y%m%d}/conus/hrrr.t{dt:%H}z.wrfsfcf00.grib2"

def get_event_data(event_id, catalog):
    try:
        event_rows = catalog[catalog['event_id'] == event_id].copy()
        vil_frames = event_rows[event_rows['img_type'] == 'vil'].copy()
        ir107_frames = event_rows[event_rows['img_type'] == 'ir107'].copy()

        if len(vil_frames) == 0 or len(ir107_frames) == 0:
            return None

        vil_frames['time_utc'] = pd.to_datetime(vil_frames['time_utc'])
        ir107_frames['time_utc'] = pd.to_datetime(ir107_frames['time_utc'])

        merged = pd.merge(
            vil_frames,
            ir107_frames,
            on='time_utc',
            suffixes=('_vil', '_ir107')
        ).sort_values('time_utc')

        return merged if len(merged) else None
    except Exception:
        return None

def _get_var(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"Missing variables {candidates}")


def process_event(event_id, catalog, s3, debug=False):
    output_filepath = os.path.join(OUTPUT_PATH, f'storm_{event_id}.npy')
    if os.path.exists(output_filepath):
        return

    merged_frames = get_event_data(event_id, catalog)
    if merged_frames is None:
        return

    first = merged_frames.iloc[0]
    center_lat = (first['llcrnrlat_vil'] + first['urcrnrlat_vil']) / 2
    center_lon = (first['llcrnrlon_vil'] + first['urcrnrlon_vil']) / 2

    event_sequence = []

    for index, row in merged_frames.iterrows():
        local_vil = os.path.join(RAW_SEVIR_PATH, row['file_name_vil'])
        local_ir = os.path.join(RAW_SEVIR_PATH, row['file_name_ir107'])

        if not os.path.exists(local_vil) or not os.path.exists(local_ir):
            continue

        timestamp = row['time_utc']
        hrrr_url = build_hrrr_url(timestamp)
        temp_file = os.path.join(TEMP_HRRR_PATH, f"temp_{event_id}_{index}.grib2")

        ds_all = None

        try:
            s3.get(hrrr_url, temp_file)

            with h5py.File(local_vil, 'r') as hf:
                vil = np.squeeze(hf['vil'][row['file_index_vil']])
            with h5py.File(local_ir, 'r') as hf:
                ir107 = np.squeeze(hf['ir107'][row['file_index_ir107']])

            lat_slice = slice(center_lat + 1.5, center_lat - 1.5)
            lon_slice = slice(center_lon - 1.5, center_lon + 1.5)

            ds_all = open_hrrr_grib(temp_file)
            ds_all = ds_all.set_coords(['latitude', 'longitude'])

            t2m = _get_var(ds_all.sel(latitude=lat_slice, longitude=lon_slice, heightAboveGround=2), ['t2m', '2t'])
            cape = _get_var(ds_all.sel(latitude=lat_slice, longitude=lon_slice), ['cape', 'capesfc'])

            try:
                prmsl = _get_var(ds_all.sel(latitude=lat_slice, longitude=lon_slice), ['prmsl', 'msl', 'mslet'])
                prmsl = np.squeeze(prmsl)
            except Exception:
                prmsl = np.zeros((vil.shape[0], vil.shape[1]))

            target = (384, 384)
            ir_resize = cv2.resize(ir107, target)
            t2m_resize = cv2.resize(np.squeeze(t2m), target)
            cape_resize = cv2.resize(np.squeeze(cape), target)
            prmsl_resize = cv2.resize(prmsl, target)

            combined = np.stack([
                vil,
                ir_resize,
                t2m_resize,
                prmsl_resize,
                cape_resize
            ], axis=-1)

            event_sequence.append(combined)

        except Exception:
            continue

        finally:
            if ds_all is not None:
                try:
                    ds_all.close()
                except:
                    pass
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    if event_sequence:
        np.save(output_filepath, np.array(event_sequence))


def main():
    print(f"Project Path: {PROJECT_PATH}")

    for p in [PROJECT_PATH, RAW_SEVIR_PATH, OUTPUT_PATH, TEMP_HRRR_PATH]:
        os.makedirs(p, exist_ok=True)

    if not download_catalog():
        print("Catalog download failed.")
        sys.exit()

    catalog = pd.read_csv(CATALOG_PATH)

    vil_events = set(catalog[catalog['img_type']=='vil']['event_id'])
    ir_events = set(catalog[catalog['img_type']=='ir107']['event_id'])
    events = sorted(list(vil_events & ir_events))

    s3 = s3fs.S3FileSystem(anon=True)

    for e in tqdm(events[:5], desc="Processing HRRR/SEVIR Events"):
        process_event(e, catalog, s3, debug=True)


if __name__ == '__main__':
    main()
