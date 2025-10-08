"""
Interaktiver Panorama Downloader für Google Street View & Apple Look Around
Unterstützt Bounding Box, Polygon und Koordinatenlisten
"""

import os
import sys
import math
import re
import random
import threading
import logging
import gc
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from streetlevel import streetview, lookaround
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np

register_heif_opener()

def setup_logging(session_dir):
    """Configure logging for the current session."""
    log_file = os.path.join(session_dir, "panorama_downloader.log")
    logger = logging.getLogger('panorama_downloader')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = None


def latlon_to_tile(lat, lon, zoom=17):
    """Converts Lat/Lon to tile coordinates (Slippy Map XYZ format)."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def get_tiles_in_bbox(lat1, lon1, lat2, lon2, zoom=17):
    """Calculates all tiles within a bounding box."""
    min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
    min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
    
    x1, y1 = latlon_to_tile(max_lat, min_lon, zoom)
    x2, y2 = latlon_to_tile(min_lat, max_lon, zoom)
    
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    
    tiles = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tiles.append((x, y))
    
    return tiles


def parse_dms_to_decimal(dms_str):
    """
    Converts Degrees/Minutes/Seconds format to decimal.
    Example: 52°30'57.6"N 13°20'50.7"E -> (52.516, 13.347416...)
    """
    pattern = r'(\d+)°(\d+)\'([\d.]+)"([NSWE])\s+(\d+)°(\d+)\'([\d.]+)"([NSWE])'
    match = re.match(pattern, dms_str.strip())
    
    if not match:
        return None
    
    lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = match.groups()
    
    # Calculate latitude
    lat = float(lat_deg) + float(lat_min) / 60 + float(lat_sec) / 3600
    if lat_dir == 'S':
        lat = -lat
    
    # Calculate longitude
    lon = float(lon_deg) + float(lon_min) / 60 + float(lon_sec) / 3600
    if lon_dir == 'W':
        lon = -lon
    
    return lat, lon


def parse_coordinate(coord_str):
    """
    Parses a coordinate in various formats:
    - Decimal: 52.520008,13.404954
    - DMS: 52°30'57.6"N 13°20'50.7"E
    """
    coord_str = coord_str.strip()
    
    if '°' in coord_str:
        result = parse_dms_to_decimal(coord_str)
        if result:
            return result
        raise ValueError("Invalid DMS format")
    
    try:
        parts = coord_str.split(',')
        if len(parts) == 2:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            return lat, lon
    except:
        pass
    
    raise ValueError("Invalid coordinate format")


def generate_random_berlin_coords():
    """Generates random coordinates in Berlin DE."""
    lat1, lon1 = 52.5127, 13.3422
    lat2, lon2 = 52.5160, 13.3656
    
    coord1_lat = random.uniform(lat1, lat2)
    coord1_lon = random.uniform(lon1, lon2)
    
    coord2_lat = coord1_lat + 0.0001
    coord2_lon = coord1_lon + 0.0003
    
    return [(coord1_lat, coord1_lon), (coord2_lat, coord2_lon)]


def streetview_task(progress, task_id, tiles, output_dir, filename_template):
    """Worker function to download Street View panoramas."""
    global logger

    downloaded_ids = set()
    errors = 0

    progress.update(task_id, total=None, description=f"[cyan]Street View [Scanning {len(tiles)} tiles...]")
    logger.info(f"Street View: Scanning {len(tiles)} tiles to count panoramas...")
    
    all_panos = []
    tiles_with_panos = 0
    for tile_idx, (tile_x, tile_y) in enumerate(tiles):
        try:
            panos = streetview.get_coverage_tile(tile_x, tile_y)
            if panos:
                panos_in_tile = 0
                logger.debug(f"Street View: Found {len(panos)} panoramas in tile {tile_x}, {tile_y}")
                for pano in panos:
                    if pano.id not in downloaded_ids:
                        downloaded_ids.add(pano.id)
                        all_panos.append(pano)
                        panos_in_tile += 1
                if panos_in_tile > 0:
                    tiles_with_panos += 1
        except Exception as e:
            logger.error(f"Street View: Error scanning tile {tile_x}, {tile_y}: {str(e)}")
    
    total_expected_panos = len(all_panos)
    logger.info(f"Street View: Found {total_expected_panos} unique panoramas in {tiles_with_panos}/{len(tiles)} tiles")
    
    if total_expected_panos == 0:
        progress.update(task_id, total=1, completed=1, description="[bold green]✓ Street View (0 panoramas)")
        return
    
    progress.update(task_id, total=total_expected_panos, completed=0, description="[cyan]Street View")
    
    total_downloaded = 0
    
    for pano_idx, pano in enumerate(all_panos):
        logger.info(f"Street View: Downloading panorama {pano_idx+1}/{total_expected_panos}: {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}")

        try:
            full_pano = streetview.find_panorama_by_id(pano.id)

            if full_pano is None:
                errors += 1
                logger.error(f"Street View: Could not find full panorama data for {pano.id}")
                continue

            if full_pano.date:
                try:
                    date_str = full_pano.date.strftime("%Y%m%d")
                except AttributeError:
                    date_str = str(full_pano.date).replace("-", "").split()[0]
            else:
                date_str = "unknown"

            adapted_template = adapt_filename_template(filename_template, 'streetview')

            filename = adapted_template.format(
                id=full_pano.id,
                lat=full_pano.lat,
                lon=full_pano.lon,
                date=date_str,
                heading=full_pano.heading,
                pitch=getattr(full_pano, 'pitch', 0),
                roll=getattr(full_pano, 'roll', 0),
                elevation=getattr(full_pano, 'elevation', 0),
                country_code=getattr(full_pano, 'country_code', ''),
                source='Google'
            )
            full_path = os.path.join(output_dir, filename)

            logger.debug(f"Street View: Saving panorama {pano.id} to {filename}")
            streetview.download_panorama(full_pano, full_path)
            total_downloaded += 1
            logger.info(f"Street View: Successfully downloaded panorama {pano.id}")
            
            progress.update(task_id, completed=total_downloaded, description="[cyan]Street View")

        except Exception as e:
            errors += 1
            logger.error(f"Street View: Error downloading panorama {pano.id}: {str(e)}")
            try:
                with open(os.path.join(output_dir, "streetview_errors.log"), "a", encoding="utf-8") as f:
                    f.write(f"Error with Pano {pano.id}: {str(e)}\n")
            except:
                pass

    result_text = f"[bold green]✓ Street View ({total_downloaded} panoramas"
    if errors > 0:
        result_text += f", {errors} errors"
    result_text += ")"
    progress.update(task_id, completed=total_expected_panos, description=result_text)


def lookaround_task(progress, task_id, tiles, output_dir, filename_template, zoom=0):
    """Worker-Funktion zum Download von Look Around-Panoramen mit Pipeline-Optimierung."""
    global logger

    total_tiles = len(tiles)
    completed = 0
    total_panos = 0
    processed_panos = 0
    downloaded_ids = set()
    errors = 0
    equirect_failures = 0

    auth = lookaround.Authenticator()

    progress.update(task_id, total=None, description=f"[magenta]Look Around [Scanning {len(tiles)} tiles...]")
    logger.info(f"Look Around: Scanning {len(tiles)} tiles to count panoramas...")
    
    all_panos = []
    tiles_with_panos = 0
    for tile_idx, (tile_x, tile_y) in enumerate(tiles):
        try:
            tile = lookaround.get_coverage_tile(tile_x, tile_y)
            if tile and tile.panos:
                panos_in_tile = 0
                for pano in tile.panos:
                    if pano.id not in downloaded_ids:
                        downloaded_ids.add(pano.id)
                        all_panos.append(pano)
                        panos_in_tile += 1
                if panos_in_tile > 0:
                    tiles_with_panos += 1
                    logger.debug(f"Look Around: Tile {tile_x},{tile_y} has {panos_in_tile} unique panoramas")
        except Exception as e:
            logger.error(f"Look Around: Error scanning tile {tile_x}, {tile_y}: {str(e)}")
    
    total_expected_panos = len(all_panos)
    logger.info(f"Look Around: Found {total_expected_panos} unique panoramas in {tiles_with_panos}/{len(tiles)} tiles")
    
    if total_expected_panos > 10000:
        logger.warning(f"Look Around: {total_expected_panos} panoramas found - this seems unusually high!")
        logger.warning(f"Look Around: Average {total_expected_panos/len(tiles):.1f} panoramas per tile")
    
    if total_expected_panos == 0:
        progress.update(task_id, total=1, completed=1, description="[bold green]✓ Look Around (0 Panoramen)")
        return
    
    progress.update(task_id, total=total_expected_panos, completed=0, description="[magenta]Look Around")

    queue_size = 1 if zoom <= 2 else 10
    download_queue = Queue(maxsize=queue_size)
    processing_done = threading.Event()
    
    progress_lock = threading.Lock()
    
    logger.info(f"Look Around: Using queue size {queue_size} for zoom level {zoom}")

    def download_worker():
        """Producer: Downloads all 6 faces and puts them in the queue."""
        nonlocal errors, processed_panos
        try:
            for pano_idx, pano in enumerate(all_panos):
                logger.info(f"Look Around: Downloading panorama {pano_idx+1}/{total_expected_panos}: {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}")

                try:
                    if pano.date:
                        try:
                            date_str = pano.date.strftime("%Y%m%d")
                        except AttributeError:
                            date_str = str(pano.date).replace("-", "").split()[0]
                    else:
                        date_str = "unknown"

                    adapted_template = adapt_filename_template(filename_template, 'lookaround')

                    filename = adapted_template.format(
                        id=pano.id,
                        lat=pano.lat,
                        lon=pano.lon,
                        date=date_str,
                        heading=pano._heading,
                        pitch=pano._pitch,
                        roll=pano._roll,
                        elevation=pano._elevation,
                        country_code='',
                        source='Apple',
                        build_id=pano.build_id,
                        coverage_type=str(pano.coverage_type).split('.')[1] if pano.coverage_type else '',
                        has_blurs='true' if pano.has_blurs else 'false'
                    )
                    full_path = os.path.join(output_dir, filename)

                    logger.debug(f"Look Around: Downloading 6 faces for panorama {pano.id}")

                    faces = [None] * 6
                    temp_files = [None] * 6
                    face_errors = [False] * 6

                    def download_face(face_idx):
                        try:
                            temp_path = os.path.join(output_dir, f"temp_{pano.id}_face{face_idx}.heic")
                            temp_files[face_idx] = temp_path
                            logger.debug(f"Look Around: Downloading face {face_idx} for panorama {pano.id}")
                            lookaround.download_panorama_face(pano, temp_path, face_idx, zoom, auth)
                            face_img = Image.open(temp_path)
                            faces[face_idx] = face_img
                        except Exception as e:
                            face_errors[face_idx] = True
                            logger.error(f"Look Around: Error downloading face {face_idx} for panorama {pano.id}: {str(e)}")

                    max_workers = 3 if zoom <= 2 else 6
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        executor.map(download_face, range(6))
                    if any(face_errors) or None in faces:
                        failed_faces = [i for i, err in enumerate(face_errors) if err]
                        logger.error(f"Look Around: Failed to download faces {failed_faces} for panorama {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}")
                        
                        try:
                            error_log_path = os.path.join(output_dir, "lookaround_errors.log")
                            with open(error_log_path, "a", encoding="utf-8") as f:
                                f.write(f"DOWNLOAD_ERROR|{pano.id}|{pano.lat:.6f}|{pano.lon:.6f}|failed_faces={failed_faces}\n")
                        except Exception as log_err:
                            logger.warning(f"Look Around: Could not write to error log: {log_err}")
                        
                        for i, face in enumerate(faces):
                            if face is not None:
                                try:
                                    face.close()
                                except:
                                    pass
                            if temp_files[i] and os.path.exists(temp_files[i]):
                                try:
                                    os.remove(temp_files[i])
                                except:
                                    pass
                        
                        with progress_lock:
                            errors += 1
                            processed_panos += 1
                        continue

                    logger.info(f"Look Around: Downloaded all 6 faces for panorama {pano.id}")

                    download_queue.put({
                        'pano': pano,
                        'faces': faces,
                        'temp_files': temp_files,
                        'full_path': full_path
                    })

                except Exception as e:
                    logger.error(f"Look Around: Error downloading faces for {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}: {str(e)}")
                    
                    try:
                        error_log_path = os.path.join(output_dir, "lookaround_errors.log")
                        with open(error_log_path, "a", encoding="utf-8") as f:
                            f.write(f"EXCEPTION|{pano.id}|{pano.lat:.6f}|{pano.lon:.6f}|{str(e)}\n")
                    except Exception as log_err:
                        logger.warning(f"Look Around: Could not write to error log: {log_err}")
                    
                    with progress_lock:
                        errors += 1
                        processed_panos += 1
                    continue

        finally:
            download_queue.put(None)
            logger.debug("Look Around: Download worker finished")

    def processing_worker():
        """Consumer: Processes downloaded faces into an equirectangular image."""
        nonlocal total_panos, processed_panos, errors, equirect_failures, completed
        
        try:
            while True:
                try:
                    item = download_queue.get(timeout=1)
                    
                    if item is None: 
                        break

                    pano = item['pano']
                    faces = item['faces']
                    temp_files = item['temp_files']
                    full_path = item['full_path']

                    logger.debug(f"Look Around: Converting panorama {pano.id} to equirectangular")

                    try:
                        try:
                            equirect_img = to_equirectangular_memory_efficient(faces)
                        except Exception as _:
                            equirect_img = lookaround.to_equirectangular(faces, pano.camera_metadata)

                        equirect_img.save(full_path, quality=95)
                        
                        try:
                            equirect_img.close()
                        except:
                            pass
                        
                        try:
                            temp_mm_path = getattr(equirect_img, '_temp_mm_path', None)
                            if temp_mm_path and os.path.exists(temp_mm_path):
                                os.remove(temp_mm_path)
                        except Exception:
                            pass
                        
                        logger.info(f"Look Around: Successfully converted panorama {pano.id} to equirectangular")

                        for face in faces:
                            try:
                                face.close()
                            except:
                                pass

                        for temp_file in temp_files:
                            try:
                                if temp_file and os.path.exists(temp_file):
                                    os.remove(temp_file)
                                    logger.debug(f"Look Around: Removed temp file {temp_file}")
                            except Exception as e:
                                logger.warning(f"Look Around: Could not remove temp file {temp_file}: {e}")

                        del equirect_img, faces, temp_files
                        
                        with progress_lock:
                            total_panos += 1
                            processed_panos += 1
                        
                        gc.collect()

                    except Exception as e:
                        with progress_lock:
                            equirect_failures += 1
                            processed_panos += 1
                        
                        logger.warning(f"Look Around: Equirectangular conversion failed for {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}: {str(e)}")
                        logger.info(f"Look Around: Saving individual faces for panorama {pano.id}")
                        
                        try:
                            error_log_path = os.path.join(output_dir, "lookaround_errors.log")
                            with open(error_log_path, "a", encoding="utf-8") as f:
                                f.write(f"CONVERSION_ERROR|{pano.id}|{pano.lat:.6f}|{pano.lon:.6f}|{str(e)}\n")
                        except Exception as log_err:
                            logger.warning(f"Look Around: Could not write to error log: {log_err}")

                        for face_idx, face in enumerate(faces):
                            face_path = os.path.join(output_dir, f"{pano.id}_face{face_idx}.jpg")
                            face.save(face_path, quality=95)
                            logger.debug(f"Look Around: Saved face {face_idx} for panorama {pano.id}")
                            try:
                                face.close()
                            except:
                                pass

                        for temp_file in temp_files:
                            try:
                                if temp_file and os.path.exists(temp_file):
                                    os.remove(temp_file)
                                    logger.debug(f"Look Around: Removed temp HEIC file {temp_file}")
                            except Exception as e:
                                logger.warning(f"Look Around: Could not remove temp HEIC file {temp_file}: {e}")

                        del faces, temp_files
                        
                        with progress_lock:
                            total_panos += 1
                        gc.collect()

                    download_queue.task_done()

                except Empty:
                    continue

        finally:
            processing_done.set()
            logger.debug("Look Around: Processing worker finished")

    def progress_updater():
        """Periodically updates the progress bar."""
        while not processing_done.is_set():
            with progress_lock:
                current_processed = processed_panos
            
            progress.update(
                task_id,
                completed=current_processed,
                description="[magenta]Look Around"
            )
            
            threading.Event().wait(0.5)  

    download_thread = threading.Thread(target=download_worker, name="LookAround-Downloader")
    processing_thread = threading.Thread(target=processing_worker, name="LookAround-Processor")
    progress_thread = threading.Thread(target=progress_updater, name="LookAround-Progress")

    download_thread.start()
    processing_thread.start()
    progress_thread.start()

    download_thread.join()
    processing_thread.join()
    processing_done.set()  
    progress_thread.join()
    result_text = f"[bold green]✓ Look Around ({total_panos} panoramas"
    if errors > 0:
        result_text += f", {errors} errors"
    if equirect_failures > 0:
        result_text += f", {equirect_failures} conversion errors"
    result_text += ")"
    progress.update(task_id, completed=total_expected_panos, description=result_text)


def to_equirectangular_memory_efficient(faces, output_width=None, strip_height=256):
    """Converts 6 cube faces to an equirectangular image in a memory-efficient way (without PyTorch).

    Assumptions:
    - faces: List of 6 PIL Images of the same size in the order:
      [right(+X), left(-X), top(+Y), bottom(-Y), front(+Z), back(-Z)]
      This order is typical for Apple Look Around. If it differs,
      the faces will be automatically permuted as soon as a wrong order
      is detected, but without guarantee. Adjust here if necessary.
    - output_width: Width of the equirectangular image. Default = 4 * face_width
    - strip_height: Number of rows per processing slice (RAM control)

    Returns: PIL Image (RGB)
    """
    if len(faces) != 6:
        raise ValueError("Exactly 6 faces are required for a cube map")

    face_images = [f.convert('RGB') for f in faces]

    face_w, face_h = face_images[0].size
    for f in face_images:
        if f.size != (face_w, face_h):
            raise ValueError("All faces must have the same size")

    if output_width is None:
        output_width = 4 * face_w
    output_height = 2 * face_h

    IDX_RIGHT, IDX_LEFT, IDX_TOP, IDX_BOTTOM, IDX_FRONT, IDX_BACK = range(6)
    face_np = [np.asarray(img) for img in face_images]

    temp_dir = None
    try:
        temp_dir = os.path.dirname(getattr(face_images[0], 'filename', '') or '.')
    except Exception:
        temp_dir = '.'

    temp_mm_path = os.path.join(temp_dir or '.', f"_tmp_equirect_{os.getpid()}_{random.randint(0, 1_000_000)}.mm")
    out = np.memmap(temp_mm_path, dtype=np.uint8, mode='w+', shape=(output_height, output_width, 3))

    xs = np.linspace(-np.pi, np.pi, output_width, endpoint=False)

    for y_start in range(0, output_height, strip_height):
        y_end = min(y_start + strip_height, output_height)
        h = y_end - y_start

        ys = np.linspace(np.pi / 2, -np.pi / 2, output_height, endpoint=False)[y_start:y_end]

        lon = np.broadcast_to(xs[np.newaxis, :], (h, output_width)).astype(np.float32, copy=False)  
        lat = np.broadcast_to(ys[:, np.newaxis], (h, output_width)).astype(np.float32, copy=False)  
        cos_lat = np.cos(lat, dtype=np.float32)
        x_dir = (cos_lat * np.cos(lon)).astype(np.float32, copy=False)
        y_dir = np.sin(lat, dtype=np.float32)
        z_dir = (cos_lat * np.sin(lon)).astype(np.float32, copy=False)

        ax = np.abs(x_dir, dtype=np.float32)
        ay = np.abs(y_dir, dtype=np.float32)
        az = np.abs(z_dir, dtype=np.float32)

        face_idx = np.argmax(np.stack([ax, ax, ay, ay, az, az], axis=0), axis=0)

        mask = face_idx == IDX_RIGHT
        denom = ax
        denom[denom == 0] = 1e-8
        u = np.where(mask, -z_dir / denom, 0)  
        v = np.where(mask, -y_dir / denom, 0)

        mask = face_idx == IDX_LEFT
        denom = ax
        denom[denom == 0] = 1e-8
        u = np.where(mask, z_dir / denom, u)
        v = np.where(mask, -y_dir / denom, v)

        mask = face_idx == IDX_TOP
        denom = ay
        denom[denom == 0] = 1e-8
        u = np.where(mask, x_dir / denom, u)
        v = np.where(mask, z_dir / denom, v)

        mask = face_idx == IDX_BOTTOM
        denom = ay
        denom[denom == 0] = 1e-8
        u = np.where(mask, x_dir / denom, u)
        v = np.where(mask, -z_dir / denom, v)

        mask = face_idx == IDX_FRONT
        denom = az
        denom[denom == 0] = 1e-8
        u = np.where(mask, x_dir / denom, u)
        v = np.where(mask, -y_dir / denom, v)

        mask = face_idx == IDX_BACK
        denom = az
        denom[denom == 0] = 1e-8
        u = np.where(mask, -x_dir / denom, u)
        v = np.where(mask, -y_dir / denom, v)

        u_px = ((u + 1) * 0.5 * (face_w - 1)).astype(np.int32)
        v_px = ((v + 1) * 0.5 * (face_h - 1)).astype(np.int32)

        strip = np.zeros((h, output_width, 3), dtype=np.uint8)

        sel = face_idx == IDX_RIGHT
        if np.any(sel):
            strip[sel] = face_np[IDX_RIGHT][v_px[sel], u_px[sel]]
        sel = face_idx == IDX_LEFT
        if np.any(sel):
            strip[sel] = face_np[IDX_LEFT][v_px[sel], u_px[sel]]

        sel = face_idx == IDX_TOP
        if np.any(sel):
            strip[sel] = face_np[IDX_TOP][v_px[sel], u_px[sel]]

        sel = face_idx == IDX_BOTTOM
        if np.any(sel):
            strip[sel] = face_np[IDX_BOTTOM][v_px[sel], u_px[sel]]

        sel = face_idx == IDX_FRONT
        if np.any(sel):
            strip[sel] = face_np[IDX_FRONT][v_px[sel], u_px[sel]]

        sel = face_idx == IDX_BACK
        if np.any(sel):
            strip[sel] = face_np[IDX_BACK][v_px[sel], u_px[sel]]

        out[y_start:y_end] = strip

        del lon, lat, cos_lat, x_dir, y_dir, z_dir, ax, ay, az, face_idx, u, v, u_px, v_px, strip

    out.flush()
    del out
    out_np = np.memmap(temp_mm_path, dtype=np.uint8, mode='r', shape=(output_height, output_width, 3))
    
    img_data = np.array(out_np, copy=True)
    del out_np
    
    img = Image.fromarray(img_data, mode='RGB')
    del img_data
    
    img._temp_mm_path = temp_mm_path
    return img


def get_coordinates():
    """Interactive coordinate input."""
    print("\nCoordinate Input")
    print("Enter at least 2 coordinate pairs")
    print("\nSupported formats:")
    print("  • Decimal:  52.520008,13.404954")
    print("  • DMS:      52°30'57.6\"N 13°20'50.7\"E")
    print("\nTip: Press Enter without input for random coordinates in Berlinᴰᴱ\n")
    
    coords = []
    first_empty = False
    
    while True:
        prompt_text = f"Coordinate {len(coords) + 1}"
        if len(coords) == 0:
            prompt_text += " (or Enter for random)"
        
        coord_input = input(prompt_text + ": ").strip()
        
        if not coord_input:
            if len(coords) == 0:
                coords = generate_random_berlin_coords()
                print("Generated random coordinates:")
                print(f"   Point 1: {coords[0][0]:.6f}, {coords[0][1]:.6f}")
                print(f"   Point 2: {coords[1][0]:.6f}, {coords[1][1]:.6f}")
                break
            elif len(coords) >= 2:
                break
            else:
                print("At least 2 coordinates are required!")
                continue
        
        try:
            lat, lon = parse_coordinate(coord_input)
            coords.append((lat, lon))
            print(f"Added: {lat:.6f}, {lon:.6f}")
        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Please use one of the supported formats!")
    
    return coords


def get_filename_template():
    """Interactive input for the filename template."""
    print("\nFilename Template")
    print("Available variables:")
    print("  {id}           - Panorama ID")
    print("  {lat}          - Latitude")
    print("  {lon}          - Longitude")
    print("  {date}         - Capture date (Look Around: YYYYMMDD; Street View: YYYYMM)")
    # print("  {heading}      - Heading (North = 0°, East = 90°, etc.)")
    # print("  {pitch}        - Pitch (radians)")
    # print("  {roll}         - Roll (radians)")
    # print("  {elevation}    - Elevation above sea level (meters)")
    # print("  {country_code} - Country code (e.g., DE, US)")
    # print("  {source}       - Source (Google, Apple, etc.)")
    # print("  {build_id}     - Build ID (Look Around only)")
    # print("  {coverage_type}- Coverage type (Look Around only)")
    # print("  {has_blurs}    - Has blurs (Look Around only)")
    
    example_template = "{lat}_{lon}_{date}.jpg"
    print(f"\nExample template: {example_template}")
    
    template = input(f"\nFilename template (default: {example_template}): ").strip()
    if not template:
        template = example_template
    
    # Validation
    if not template.endswith(('.jpg', '.jpeg', '.png')):
        template += '.jpg'
    return template


def adapt_filename_template(template, platform):
    """
    Adapts filename template for specific platform by removing unavailable variables.

    Args:
        template (str): Original filename template with placeholders like {variable}
        platform (str): Either 'streetview' or 'lookaround'

    Returns:
        str: Adapted template with only available variables for the platform
    """
    import re

    if platform == 'streetview':
        template = template.replace('{build_id}', '')
        template = template.replace('{coverage_type}', '')
        template = template.replace('{has_blurs}', '')

        template = re.sub(r'[_\-\s]+', '_', template)
        template = template.strip('_- ')
        if template and not template.endswith('.jpg'):
            template = template.rstrip('_-.')
            if not template.endswith('.jpg'):
                template += '.jpg'

    return template


def main():
    """Main function - Interactive CLI."""
    global logger

    console = Console()

    ascii_art = r"""
.d88b.  w                     w      888b.                                           
YPwww. w8ww 8d8b .d88b .d88b w8ww    8  .8 .d88 8d8b. .d8b. 8d8b .d88 8d8b.d8b. .d88 
    d8  8   8P   8.dP' 8.dP'  8      8wwP' 8  8 8P Y8 8' .8 8P   8  8 8P Y8P Y8 8  8 
`Y88P'  Y8P 8    `Y88P `Y88P  Y8P    8     `Y88 8   8 `Y8P' 8    `Y88 8   8   8 `Y88 
                                                                                     
    """
    
    console.print(ascii_art, style="bold cyan")
    console.print("Street View & Look Around", style="bold cyan", justify="center")
    console.print("=" * 100, style="cyan")

    coords = get_coordinates()

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    tiles = get_tiles_in_bbox(min(lats), min(lons), max(lats), max(lons))

    console.print(f"\nInput Summary:", style="bold")
    console.print(f"Coordinates: {len(coords)} points")
    console.print(f"Bounding Box: {min(lats):.6f},{min(lons):.6f} to {max(lats):.6f},{max(lons):.6f}")
    console.print(f"Tiles (Zoom 17): {len(tiles)}")
    if len(tiles) > 100:
        console.print(f"[yellow] Warning: {len(tiles)} tiles is a lot - the download may take a long time![/yellow]")

    console.print("\nDownload Mode", style="bold")
    download_streetview = input("Download Street View? (y/n, default: y): ").strip().lower()
    download_streetview = download_streetview in ('y', 'yes', '')

    download_lookaround = input("Download Look Around? (y/n, default: y): ").strip().lower()
    download_lookaround = download_lookaround in ('y', 'yes', '')

    if not download_streetview and not download_lookaround:
        console.print("No download option selected. The program will now exit.")
        return

    filename_template = get_filename_template()

    lookaround_zoom = 2
    if download_lookaround:
        console.print("\nLook Around Quality", style="bold green")
        console.print("Zoom Level (0-7, lower = better quality but slower):", style="white")
        console.print("  • 0: Highest quality, only recommended with 64GB RAM", style="bright_red")
        console.print("  • 1: Very high quality", style="red")
        console.print("  • 2-3: High quality", style="yellow")
        console.print("  • 4-5: Medium", style="green")
        console.print("  • 6-7: Low (fast)", style="blue")

        zoom_input = input("Zoom Level (default: 2): ").strip()
        if zoom_input:
            try:
                lookaround_zoom = int(zoom_input)
                if lookaround_zoom < 0 or lookaround_zoom > 7:
                    console.print("Invalid value, using default (2 = High quality)")
                    lookaround_zoom = 2
            except ValueError:
                console.print("Invalid value, using default (2 = High quality)")
                lookaround_zoom = 0

    # Prepare session
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = timestamp
    os.makedirs(base_dir, exist_ok=True)

    # Initialize logging
    logger = setup_logging(base_dir)
    logger.info("Panorama Downloader session started")
    logger.info(f"Session directory: {base_dir}")
    logger.info(f"Coordinates: {coords}")
    logger.info(f"Download Street View: {download_streetview}, Look Around: {download_lookaround}")
    logger.info(f"Look Around zoom level: {lookaround_zoom}")

    streetview_dir = os.path.join(base_dir, "Panoramas_StreetView")
    lookaround_dir = os.path.join(base_dir, "Panoramas_LookAround")

    if download_streetview:
        os.makedirs(streetview_dir, exist_ok=True)
    if download_lookaround:
        os.makedirs(lookaround_dir, exist_ok=True)


    console.print(f"\nDownload startet...", style="bold green")
    console.print(f"Output: {base_dir}/")

    if download_lookaround:
        try:
            import torch
        except ImportError:
            console.print("PyTorch not installed - 6 Faces for ech panorama will be saved individually", style="yellow")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        threads = []

        if download_streetview:
            task_id = progress.add_task("[cyan]Street View", total=len(tiles))
            thread = threading.Thread(
                target=streetview_task,
                args=(progress, task_id, tiles, streetview_dir, filename_template)
            )
            threads.append(thread)

        if download_lookaround:
            task_id = progress.add_task(f"[magenta]Look Around", total=len(tiles))
            thread = threading.Thread(
                target=lookaround_task,
                args=(progress, task_id, tiles, lookaround_dir, filename_template, lookaround_zoom)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    console.print(f"\nDownload finished!", style="bold green")
    console.print(f"Results saved in: {base_dir}/")
    logger.info("Panorama Downloader session completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted.")
        sys.exit(0)
