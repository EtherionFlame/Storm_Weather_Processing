import os
import subprocess
# use the LibraryRequirements.txt file (pip install -r LibraryRequirements.txt )
import h5py
import s3fs
import xarray as xr
# import cfgrib  <-- No longer need this import
import numpy as np
import pandas as pd
from datetime import datetime
import cv2  # Used for image resizing
from tqdm import tqdm
import urllib.request  # Import the built-in library
import sys  # Import sys to allow exiting

# Fix the FutureWarning from xarray
xr.set_options(use_new_combine_kwarg_defaults=True)

# =============================================================================
# --- 1. CONFIGURATION: SET YOUR PATHS HERE ---
# =============================================================================

# file path for drive
BASE_DRIVE_PATH = 'D:/'

# --- Project Paths (Script will create these) ---
PROJECT_PATH = os.path.join(BASE_DRIVE_PATH, 'WeatherProject')
RAW_SEVIR_PATH = os.path.join(PROJECT_PATH, 'RawSEVIRData')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'ProcessedData')
CATALOG_PATH = os.path.join(PROJECT_PATH, 'CATALOG.csv')

# --- NEW: Temporary folder for downloading HRRR files ---
TEMP_HRRR_PATH = os.path.join(PROJECT_PATH, 'temp_hrrr')


# =============================================================================


def download_catalog():
    """Downloads the SEVIR catalog if it doesn't exist."""
    if os.path.exists(CATALOG_PATH):
        # Check file size. If it's tiny, it's a bad download.
        if os.path.getsize(CATALOG_PATH) < 1000000:
            print("Catalog file exists but is too small. Deleting and re-downloading.")
            try:
                os.remove(CATALOG_PATH)
            except Exception as e:
                print(f"Could not delete file: {e}. Please delete it manually.")
                return False
        else:
            print(f"Catalog already exists at {CATALOG_PATH}")
            return True  # Catalog exists and is large enough

    print("Downloading SEVIR CATALOG.csv...")
    try:
        url = 'https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv'
        # Use urllib to download the file, removing the 'requests' dependency
        urllib.request.urlretrieve(url, CATALOG_PATH)
        print("Catalog download complete.")
        return True
    except Exception as e:
        print(f"Error downloading catalog: {e}")
        print("Please check internet connection or permissions.")
        return False


def download_sevir_data_type(data_type):
    """
    Downloads a single data type (e.g., 'vil', 'ir107') from the SEVIR dataset.
    This function is RESUMABLE.
    """
    print(f"--- Starting SEVIR '{data_type}' Data Download ---")
    s3_source_path = f's3://sevir/data/{data_type}'
    local_destination_path = os.path.join(RAW_SEVIR_PATH, data_type)
    print(f"Source: {s3_source_path}")
    print(f"Destination: {local_destination_path}")
    print("This will take a long time. AWS CLI will show live progress in your terminal.")
    print("If you stop (Ctrl+C), just run the script again to resume.")
    os.makedirs(local_destination_path, exist_ok=True)
    command = [
        'aws', 's3', 'sync',
        s3_source_path,
        local_destination_path,
        '--no-sign-request'
    ]
    try:
        subprocess.run(command)
        print(f"--- '{data_type}' Data Download Complete ---")
    except FileNotFoundError:
        print("Error: 'aws' command not found.")
        print("Please install the AWS CLI and ensure it is in your system's PATH.")
    except Exception as e:
        print(f"An error occurred during the download: {e}")


def build_hrrr_url(timestamp):
    """Constructs the S3 URL for an HRRR file given a timestamp."""
    dt = pd.to_datetime(timestamp)
    date_str = dt.strftime('%Y%m%d')
    hour_str = dt.strftime('%H')
    return f's3://noaa-hrrr-bdp-pds/hrrr.{date_str}/conus/hrrr.t{hour_str}z.wrfsfcf00.grib2'


def get_event_data(event_id, catalog):
    """
    Gets all frames for a given event_id from the catalog.
    Merges VIL and IR107 dataframes on 'time_utc' to ensure perfect alignment.
    """
    try:
        event_rows = catalog[catalog['event_id'] == event_id].copy()

        vil_frames = event_rows[event_rows['img_type'] == 'vil'].copy()
        ir107_frames = event_rows[event_rows['img_type'] == 'ir107'].copy()

        if len(vil_frames) == 0 or len(ir107_frames) == 0:
            # print(f"Info: Event {event_id} missing VIL or IR107 data. Skipping.")
            return None

        # Convert to datetime *before* merging
        vil_frames['time_utc'] = pd.to_datetime(vil_frames['time_utc'])
        ir107_frames['time_utc'] = pd.to_datetime(ir107_frames['time_utc'])

        merged_frames = pd.merge(
            vil_frames,
            ir107_frames,
            on='time_utc',
            suffixes=('_vil', '_ir107')
        )

        merged_frames = merged_frames.sort_values('time_utc')

        if len(merged_frames) == 0:
            # print(f"Info: Event {event_id} has no matching VIL/IR107 timestamps. Skipping.")
            return None

        return merged_frames

    except Exception as e:
        print(f"Error getting event data for {event_id}: {e}")
        return None


def _get_var(ds, candidates):
    """
    Helper function to robustly get a variable from an xarray.Dataset
    by trying a list of possible candidate names.
    """
    for name in candidates:
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"None of the variables found: {candidates}. Available: {list(ds.variables)}")


def process_event(event_id, catalog, s3, debug=False):
    """
    Processes a single storm event. Assumes raw SEVIR files are already
    downloaded in RAW_SEVIR_PATH. Downloads HRRR data to a temp folder,
    processes it, and then deletes it.
    This function is resumable.
    """
    output_filepath = os.path.join(OUTPUT_PATH, f'storm_{event_id}.npy')
    if os.path.exists(output_filepath):
        if debug:
            print(f"Event {event_id}: Already processed, skipping")
        return  # Event already processed, skip

    try:
        # Get the merged and time-aligned dataframe for this event
        merged_frames = get_event_data(event_id, catalog)

        if merged_frames is None:
            if debug:
                print(f"Event {event_id}: No merged frames (missing VIL/IR107 or no timestamp matches)")
            return  # Skip if we can't get event data

        if debug:
            print(f"Event {event_id}: Found {len(merged_frames)} aligned frames")

        # Get center coordinates from first VIL frame
        first_row = merged_frames.iloc[0]
        center_lat = (first_row['llcrnrlat_vil'] + first_row['urcrnrlat_vil']) / 2
        center_lon = (first_row['llcrnrlon_vil'] + first_row['urcrnrlon_vil']) / 2

        event_sequence = []

        # Process each aligned frame
        for index, row in merged_frames.iterrows():

            # Get file paths from the merged row
            local_vil_path = os.path.join(RAW_SEVIR_PATH, row['file_name_vil'])
            local_ir107_path = os.path.join(RAW_SEVIR_PATH, row['file_name_ir107'])

            if not os.path.exists(local_vil_path) or not os.path.exists(local_ir107_path):
                if debug:
                    print(f"Event {event_id}: Missing SEVIR files")
                    print(f"  VIL path: {local_vil_path} (exists: {os.path.exists(local_vil_path)})")
                    print(f"  IR107 path: {local_ir107_path} (exists: {os.path.exists(local_ir107_path)})")
                continue  # Skip this frame if data is missing

            timestamp = row['time_utc']  # Already a datetime object
            hrrr_url = build_hrrr_url(timestamp)
            temp_hrrr_file = os.path.join(TEMP_HRRR_PATH, f"temp_{event_id}_{index}.grib2")

            # --- Define datasets for the 'finally' block ---
            ds_all = None  # We only need one dataset object

            try:
                # 1. Download the temp file using s3fs
                s3.get(hrrr_url, temp_hrrr_file)

                # Read SEVIR data
                with h5py.File(local_vil_path, 'r') as hf:
                    vil_data = hf['vil'][row['file_index_vil']]

                with h5py.File(local_ir107_path, 'r') as hf:
                    ir107_data = hf['ir107'][row['file_index_ir107']]

                # Squeeze data to 2D
                vil_data = np.squeeze(vil_data)
                ir107_data = np.squeeze(ir107_data)

                lat_slice = slice(center_lat + 1.5, center_lat - 1.5)
                lon_slice = slice(center_lon - 1.5, center_lon + 1.5)

                # --- START OF NEW FIX ---
                # Use xr.open_dataset (singular) but pass join='outer'
                # This tells the cfgrib engine to merge the internal
                # datasets (surface, isobaricInhPa, etc.) correctly.
                ds_all = xr.open_dataset(
                    temp_hrrr_file,
                    engine='cfgrib',
                    join='outer',  # <-- THIS IS THE CRITICAL FIX
                    backend_kwargs={'indexpath': ''}
                )
                # --- END OF NEW FIX ---

                # --- Check for variables in the single merged dataset ---
                t2m_candidates = ['t2m', '2t']
                prmsl_candidates = ['prmsl', 'msl', 'mslet']
                cape_candidates = ['cape', 'capesfc']

                t2m_found = any(var in ds_all.variables for var in t2m_candidates)
                cape_found = any(var in ds_all.variables for var in cape_candidates)
                prmsl_found = any(var in ds_all.variables for var in prmsl_candidates)

                # Check if we found required variables
                if not t2m_found or not cape_found:
                    missing = []
                    if not t2m_found: missing.append('t2m')
                    if not cape_found: missing.append('cape')
                    if debug:
                        print(
                            f"Warning: Could not find required variables {missing} in GRIB file for {event_id}. Skipping frame.")
                    continue

                # prmsl is optional - log if missing but continue processing
                if not prmsl_found and debug:
                    print(f"Info: PRMSL not found for {event_id}, will use zeros.")

                # Now, set coordinates ONCE on the merged dataset
                # This should now work
                ds_all = ds_all.set_coords(['latitude', 'longitude'])

                # A. Get t2m
                hrrr_t2m_sliced = ds_all.sel(latitude=lat_slice, longitude=lon_slice, heightAboveGround=2)
                hrrr_t2m = _get_var(hrrr_t2m_sliced, t2m_candidates).values

                # B. Get prmsl (optional - use zeros if not available)
                if prmsl_found:
                    try:
                        hrrr_prmsl_sliced = ds_all.sel(latitude=lat_slice, longitude=lon_slice)
                        hrrr_prmsl = _get_var(hrrr_prmsl_sliced, prmsl_candidates).values
                        hrrr_prmsl = np.squeeze(hrrr_prmsl)
                    except Exception as e:
                        if debug: print(f"Warning: Error extracting prmsl: {e}")
                        hrrr_prmsl = None  # Fallback
                else:
                    hrrr_prmsl = None

                # C. Get cape
                hrrr_cape_sliced = ds_all.sel(latitude=lat_slice, longitude=lon_slice)
                hrrr_cape = _get_var(hrrr_cape_sliced, cape_candidates).values

                # Squeeze data
                hrrr_t2m = np.squeeze(hrrr_t2m)
                hrrr_cape = np.squeeze(hrrr_cape)

                # Resize all data
                target_size = (384, 384)
                hrrr_t2m_resized = cv2.resize(hrrr_t2m, target_size, interpolation=cv2.INTER_LINEAR)
                hrrr_cape_resized = cv2.resize(hrrr_cape, target_size, interpolation=cv2.INTER_LINEAR)

                # Handle prmsl - use zeros if not available
                if hrrr_prmsl is not None:
                    hrrr_prmsl_resized = cv2.resize(hrrr_prmsl, target_size, interpolation=cv2.INTER_LINEAR)
                else:
                    hrrr_prmsl_resized = np.zeros(target_size, dtype=np.float32)
                sevir_ir107_resized = cv2.resize(ir107_data, target_size, interpolation=cv2.INTER_LINEAR)

                # Stack the 5 channels
                combined_frame = np.stack([
                    vil_data,
                    sevir_ir107_resized,
                    hrrr_t2m_resized,
                    hrrr_prmsl_resized,
                    hrrr_cape_resized
                ], axis=-1)

                event_sequence.append(combined_frame)

            except FileNotFoundError:
                print(f"Warning: HRRR file not found for event {event_id}, timestamp {timestamp}. Skipping frame.")
            except KeyError as e:
                # This will catch "no index found", "None of the variables found", and the "latitude" key error
                print(f"Warning: Missing data for event {event_id}: {e}. Skipping frame.")
            except Exception as e:
                print(f"Warning: Error processing frame for event {event_id}: {e}. Skipping frame.")
            finally:
                # 3. ALWAYS close all datasets and delete the temp file

                # Close the single merged dataset
                if ds_all is not None:
                    try:
                        ds_all.close()
                    except:
                        pass

                # Try to delete the temp file, with retries if it's locked
                if os.path.exists(temp_hrrr_file):
                    import time
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            os.remove(temp_hrrr_file)
                            break
                        except PermissionError:
                            if attempt < max_attempts - 1:
                                time.sleep(0.5)  # Wait before retry
                            else:
                                if debug:
                                    print(
                                        f"Warning: Could not delete temp file {temp_hrrr_file} after {max_attempts} attempts")

        if len(event_sequence) > 0:  # Only save if we got at least 1 valid frames
            final_sequence = np.array(event_sequence)
            np.save(output_filepath, final_sequence)
            if debug:
                print(f"Event {event_id}: Successfully saved {len(event_sequence)} frames to {output_filepath}")
        else:
            if debug:
                # Changed minimum to 1 to match the save condition
                print(f"Event {event_id}: Only got {len(event_sequence)} frames, need at least 1. Skipping.")

    except Exception as e:
        print(f"CRITICAL Error processing event {event_id}: {e}. Skipping entire event.")


def main():
    """Main function to run the entire pipeline."""
    # --- 1. Setup ---
    print(f"Project Path: {PROJECT_PATH}")
    print(f"Raw SEVIR Data Path: {RAW_SEVIR_PATH}")
    print(f"Processed Data Path: {OUTPUT_PATH}")
    print(f"Temporary HRRR Path: {TEMP_HRRR_PATH}")  # New path

    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(RAW_SEVIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(TEMP_HRRR_PATH, exist_ok=True)  # Create the temp folder

    # --- 2. Download & VERIFY Catalog ---
    try:
        if not download_catalog():
            raise Exception("Failed to download catalog. Please check permissions or download manually.")

        print("--- Verifying Catalog File ---")
        file_size = os.path.getsize(CATALOG_PATH)
        print(f"File size on disk: {file_size} bytes")

        catalog = pd.read_csv(CATALOG_PATH, low_memory=False)
        print("\nColumns found in catalog:")
        print(list(catalog.columns))

        # Check for required columns
        required_cols = ['event_id', 'img_type', 'time_utc', 'file_name', 'file_index',
                         'llcrnrlat', 'llcrnrlon', 'urcrnrlat', 'urcrnrlon']
        missing_cols = [col for col in required_cols if col not in catalog.columns]

        if missing_cols:
            raise Exception(f"Verification FAILED: Missing columns: {missing_cols}")

        print("\n--- Catalog Verification Successful ---")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load or verify the CATALOG.csv file.")
        print(f"Error: {e}")
        sys.exit()

    # --- 3. Download Raw SEVIR Data (In Chunks) ---
    print("\n--- CONFIGURE YOUR DOWNLOAD ---")
    DOWNLOAD_VIL = False  # Set to True to download VIL data
    DOWNLOAD_IR107 = False  # Set to True to download IR107 data

    if DOWNLOAD_VIL:
        download_sevir_data_type('vil')

    if DOWNLOAD_IR107:
        download_sevir_data_type('ir107')

    # --- 4. Process All Data ---
    print("\n--- CONFIGURE YOUR PROCESSING ---")
    RUN_PROCESSING = True
    FORCE_REPROCESS = True  # Set to True to delete and reprocess all data
    DEBUG_MODE = True  # Set to True to see detailed processing info for first 5 events

    if RUN_PROCESSING:
        print("\n--- Starting Data Processing ---")

        # Delete existing processed data if FORCE_REPROCESS is enabled
        if FORCE_REPROCESS and os.path.exists(OUTPUT_PATH):
            print(f"FORCE_REPROCESS enabled. Deleting existing processed data in {OUTPUT_PATH}...")
            import shutil
            shutil.rmtree(OUTPUT_PATH)
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            print("Existing data deleted. Starting fresh processing...")

        print(f"Processed files will be saved to {OUTPUT_PATH}")

        # Get unique event IDs that have both VIL and IR107 data
        vil_events = set(catalog[catalog['img_type'] == 'vil']['event_id'])
        ir107_events = set(catalog[catalog['img_type'] == 'ir107']['event_id'])
        event_ids_to_process = sorted(list(vil_events.intersection(ir107_events)))

        print(f"\nFound {len(event_ids_to_process)} events with both VIL and IR107 data.")

        # Initialize the S3 filesystem ONCE here for efficiency.
        # 'anon=True' means we're accessing a public bucket (no login needed).
        print("Initializing S3 file system for HRRR data...")
        s3 = s3fs.S3FileSystem(anon=True)

        # Run the processing loop with a progress bar
        print("\nProcessing storms:")

        if DEBUG_MODE:
            print("\n=== DEBUG MODE: Processing first 5 events with detailed output ===")
            for i, event_id in enumerate(event_ids_to_process[:5]):
                print(f"\n--- Processing Event {i + 1}/5: {event_id} ---")
                process_event(event_id, catalog, s3, debug=True)
            print("\n=== DEBUG MODE COMPLETE ===\n")

            # Ask user if they want to continue
            response = input("Continue processing all remaining events? (y/n): ")
            if response.lower() != 'y':
                print("Processing cancelled by user.")
                return

        for event_id in tqdm(event_ids_to_process, desc="Processing All Storms"):
            # Pass the s3 object to the function
            process_event(event_id, catalog, s3, debug=False)

        print("\n--- Data Processing Complete ---")
        print(f"Your final, processed .npy files are located in: {OUTPUT_PATH}")
    else:
        print("Skipping processing step as RUN_PROCESSING is set to False.")


if __name__ == "__main__":
    main()