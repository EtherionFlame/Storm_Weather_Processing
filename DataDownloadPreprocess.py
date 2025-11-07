import os
import subprocess
#use the LibraryRequirements.txt file (pip install -r LibraryRequirements.txt )
import h5py
import s3fs
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import cv2  # Used for image resizing
from tqdm import tqdm


# =============================================================================
# --- 1. CONFIGURATION: SET YOUR PATHS HERE ---
# =============================================================================

#file path for drive
BASE_DRIVE_PATH = 'E:/'

# --- Project Paths (Script will create these) ---
PROJECT_PATH = os.path.join(BASE_DRIVE_PATH, 'WeatherProject')
RAW_SEVIR_PATH = os.path.join(PROJECT_PATH, 'RawSEVIRData')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'ProcessedData')
CATALOG_PATH = os.path.join(PROJECT_PATH, 'CATALOG.csv')


# =============================================================================


def download_catalog():
    """Downloads the SEVIR catalog if it doesn't exist."""
    if os.path.exists(CATALOG_PATH):
        print(f"Catalog already exists at {CATALOG_PATH}")
        return

    print("Downloading SEVIR CATALOG.csv...")
    try:
        import requests
        url = 'https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv'
        r = requests.get(url, allow_redirects=True)
        open(CATALOG_PATH, 'wb').write(r.content)
        print("Catalog download complete.")
    except Exception as e:
        print(f"Error downloading catalog: {e}")
        print("Please download it manually and place it at: {CATALOG_PATH}")


def download_all_sevir_data():
    """
    Downloads the entire SEVIR dataset (all data types) to your hard drive.
    This will take a very long time and use ~600GB of space.
    """
    print("--- Starting SEVIR Raw Data Download ---")
    print(f"Source: s3://sevir/data")
    print(f"Destination: {RAW_SEVIR_PATH}")
    print("This will take many hours or even days. The script will show progress.")

    # We use 'aws s3 sync' as it's the most robust multi-threaded tool
    # --no-sign-request is for public S3 buckets
    command = [
        'aws', 's3', 'sync',
        's3://sevir/data',
        RAW_SEVIR_PATH,
        '--no-sign-request'
    ]

    try:
        # Run the command. This will block until the download is complete.
        subprocess.run(command, check=True)
        print("--- SEVIR Raw Data Download Complete ---")
    except subprocess.CalledProcessError as e:
        print(f"Error during AWS sync: {e}")
        print("Please ensure AWS CLI is installed and in your system's PATH.")
    except FileNotFoundError:
        print("Error: 'aws' command not found.")
        print("Please install the AWS CLI and ensure it is in your system's PATH.")


def build_hrrr_url(timestamp):
    """Constructs the S3 URL for an HRRR file given a timestamp."""
    dt = pd.to_datetime(timestamp)
    date_str = dt.strftime('%Y%m%d')
    hour_str = dt.strftime('%H')
    return f's3://noaa-hrrr-bdp-pds/hrrr.{date_str}/conus/hrrr.t{hour_str}z.wrfsfcf00.grib2'


def get_event_file_paths(event_id, catalog):
    """
    Finds the file paths and indices for VIL and IR107 data for a given event_id.
    """
    event_rows = catalog[catalog['id'] == event_id]
    vil_row = event_rows[event_rows['data_type'] == 'vil'].iloc[0]
    ir107_row = event_rows[event_rows['data_type'] == 'ir107'].iloc[0]
    vil_path_info = (vil_row['file_name'], vil_row['file_index'])
    ir107_path_info = (ir107_row['file_name'], ir107_row['file_index'])
    return vil_row, vil_path_info, ir107_path_info


def process_event(event_id, catalog, fs_s3):
    """
    Processes a single storm event. Assumes raw SEVIR files are already
    downloaded in RAW_SEVIR_PATH. Streams HRRR data from the cloud.
    Saves the final processed .npy file to OUTPUT_PATH.
    """
    output_filepath = os.path.join(OUTPUT_PATH, f'storm_{event_id}.npy')
    if os.path.exists(output_filepath):
        return  # Event already processed

    try:
        primary_row, vil_info, ir107_info = get_event_file_paths(event_id, catalog)
        vil_file_name, vil_index = vil_info
        ir107_file_name, ir107_index = ir107_info

        # Build local paths to the raw data on your 8TB drive
        local_vil_path = os.path.join(RAW_SEVIR_PATH, vil_file_name)
        local_ir107_path = os.path.join(RAW_SEVIR_PATH, ir107_file_name)

        with h5py.File(local_vil_path, 'r') as hf:
            vil_data = hf['vil'][vil_index]
        with h5py.File(local_ir107_path, 'r') as hf:
            ir107_data = hf['ir107'][ir107_index]

        if vil_data.ndim == 4: vil_data = np.squeeze(vil_data, axis=-1)
        if ir107_data.ndim == 4: ir107_data = np.squeeze(ir107_data, axis=-1)

        timestamps_str = primary_row['time_utc'].strip('[]').replace("'", "").split(', ')
        timestamps = [pd.to_datetime(ts) for ts in timestamps_str]
        center_lat = primary_row['ctr_lat']
        center_lon = primary_row['ctr_lon']

        event_sequence = []
        for i in range(vil_data.shape[0]):  # Loop through all 49 frames
            timestamp = timestamps[i]
            hrrr_url = build_hrrr_url(timestamp)

            try:
                with fs_s3.open(hrrr_url) as hrrr_file_stream:
                    hrrr_ds = xr.open_dataset(hrrr_file_stream, engine='cfgrib',
                                              backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})

                lat_slice = slice(center_lat + 1.5, center_lat - 1.5)
                lon_slice = slice(center_lon - 1.5, center_lon + 1.5)

                hrrr_t2m = hrr_ds['t2m'].sel(latitude=lat_slice, longitude=lon_slice).values
                hrrr_prmsl = hrr_ds['prmsl'].sel(latitude=lat_slice, longitude=lon_slice).values
                hrrr_cape = hrr_ds['cape'].sel(latitude=lat_slice, longitude=lon_slice).values

                target_size = (384, 384)  # VIL's size
                hrrr_t2m_resized = cv2.resize(hrrr_t2m, target_size, interpolation=cv2.INTER_LINEAR)
                hrrr_prmsl_resized = cv2.resize(hrrr_prmsl, target_size, interpolation=cv2.INTER_LINEAR)
                hrrr_cape_resized = cv2.resize(hrrr_cape, target_size, interpolation=cv2.INTER_LINEAR)

                sevir_vil_frame = vil_data[i]  # 384x384
                sevir_ir107_frame = ir107_data[i]  # 192x192
                sevir_ir107_resized = cv2.resize(sevir_ir107_frame, target_size, interpolation=cv2.INTER_LINEAR)

                combined_frame = np.stack([
                    sevir_vil_frame,
                    sevir_ir107_resized,
                    hrrr_t2m_resized,
                    hrrr_prmsl_resized,
                    hrrr_cape_resized
                ], axis=-1)

                event_sequence.append(combined_frame)

            except FileNotFoundError:
                # Log this error but continue processing the event
                print(f"Warning: HRRR file not found for event {event_id}, timestamp {timestamp}. Skipping frame.")
            except Exception as e:
                print(f"Warning: Error processing frame {i} for event {event_id}: {e}. Skipping frame.")

        if event_sequence:
            final_sequence = np.array(event_sequence)
            np.save(output_filepath, final_sequence)

    except FileNotFoundError as e:
        print(f"Error: Raw SEVIR file not found for event {event_id}. {e}")
        print("Did you run the download_all_sevir_data() function first?")
    except Exception as e:
        print(f"CRITICAL Error processing event {event_id}: {e}. Skipping entire event.")


def main():
    """Main function to run the entire pipeline."""

    # --- 1. Setup ---
    print(f"Project Path: {PROJECT_PATH}")
    print(f"Raw SEVIR Data Path: {RAW_SEVIR_PATH}")
    print(f"Processed Data Path: {OUTPUT_PATH}")
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(RAW_SEVIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # --- 2. Download Catalog ---
    download_catalog()
    try:
        catalog = pd.read_csv(CATALOG_PATH)
    except FileNotFoundError:
        print("Catalog not found. Exiting.")
        return

    # --- 3. Download Raw SEVIR Data ---
    # WARNING: This will download ~600GB of data and take a very long time.
    # If you have already downloaded the data, you can comment out this line.
    # download_all_sevir_data()

    # --- 4. Process All Data ---
    print("\n--- Starting Data Processing ---")
    print(f"Processed files will be saved to {OUTPUT_PATH}")

    # Find all unique event IDs with both VIL and IR107 data
    vil_ids = set(catalog[catalog['data_type'] == 'vil']['id'])
    ir107_ids = set(catalog[catalog['data_type'] == 'ir107']['id'])
    event_ids_to_process = sorted(list(vil_ids.intersection(ir107_ids)))

    print(f"Found {len(event_ids_to_process)} total events to process.")

    # Initialize the S3 filesystem one time for streaming HRRR data
    fs_s3 = s3fs.S3FileSystem(anon=True)

    # Run the processing loop with a progress bar
    for event_id in tqdm(event_ids_to_process, desc="Processing All Storms"):
        process_event(event_id, catalog, fs_s3)

    print("--- Data Processing Complete ---")
    print(f"Your final, processed .npy files are located in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()