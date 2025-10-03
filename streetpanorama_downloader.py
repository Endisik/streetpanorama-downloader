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
from datetime import datetime
from pathlib import Path
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

# HEIC Support aktivieren
register_heif_opener()

# Logging konfigurieren
def setup_logging(session_dir):
    """Logging für die aktuelle Session konfigurieren."""
    log_file = os.path.join(session_dir, "panorama_downloader.log")

    # Logger konfigurieren
    logger = logging.getLogger('panorama_downloader')
    logger.setLevel(logging.DEBUG)

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Console Handler (nur Warnings und Errors)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Globaler Logger
logger = None


def latlon_to_tile(lat, lon, zoom=17):
    """Konvertiert Lat/Lon zu Tile-Koordinaten (Slippy Map XYZ Format)."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def get_tiles_in_bbox(lat1, lon1, lat2, lon2, zoom=17):
    """Berechnet alle Tiles innerhalb eines Bounding Box."""
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
    Konvertiert Grad/Minuten/Sekunden Format zu Dezimal.
    Beispiel: 52°30'57.6"N 13°20'50.7"E -> (52.516, 13.347416...)
    """
    # Pattern für DMS: 52°30'57.6"N 13°20'50.7"E
    pattern = r'(\d+)°(\d+)\'([\d.]+)"([NSWE])\s+(\d+)°(\d+)\'([\d.]+)"([NSWE])'
    match = re.match(pattern, dms_str.strip())
    
    if not match:
        return None
    
    lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = match.groups()
    
    # Breitengrad berechnen
    lat = float(lat_deg) + float(lat_min) / 60 + float(lat_sec) / 3600
    if lat_dir == 'S':
        lat = -lat
    
    # Längengrad berechnen
    lon = float(lon_deg) + float(lon_min) / 60 + float(lon_sec) / 3600
    if lon_dir == 'W':
        lon = -lon
    
    return lat, lon


def parse_coordinate(coord_str):
    """
    Parst eine Koordinate in verschiedenen Formaten:
    - Dezimal: 52.520008,13.404954
    - DMS: 52°30'57.6"N 13°20'50.7"E
    """
    coord_str = coord_str.strip()
    
    # Versuche DMS-Format
    if '°' in coord_str:
        result = parse_dms_to_decimal(coord_str)
        if result:
            return result
        raise ValueError("Ungültiges DMS-Format")
    
    # Versuche Dezimal-Format
    try:
        parts = coord_str.split(',')
        if len(parts) == 2:
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            return lat, lon
    except:
        pass
    
    raise ValueError("Ungültiges Koordinatenformat")


def generate_random_berlin_coords():
    """Generiert zufällige Koordinaten in Berlin."""
    lat1, lon1 = 52.5127, 13.3422
    lat2, lon2 = 52.5160, 13.3656
    
    # Erstes Koordinatenpaar: Zufällig im Bereich
    coord1_lat = random.uniform(lat1, lat2)
    coord1_lon = random.uniform(lon1, lon2)
    
    # Zweites Koordinatenpaar: +0.001, +0.005
    coord2_lat = coord1_lat + 0.001
    coord2_lon = coord1_lon + 0.005
    
    return [(coord1_lat, coord1_lon), (coord2_lat, coord2_lon)]


def streetview_task(progress, task_id, tiles, output_dir, filename_template):
    """Worker-Funktion zum Download von Street View-Panoramen."""
    global logger

    total_tiles = len(tiles)
    completed = 0
    total_panos = 0
    downloaded_ids = set()
    errors = 0
    empty_tiles = 0

    progress.update(task_id, total=total_tiles, description="[cyan]Street View [0/0 Panos, 0/0 Tiles]")

    for tile_idx, (tile_x, tile_y) in enumerate(tiles):
        logger.debug(f"Street View: Processing tile {tile_idx+1}/{total_tiles} ({tile_x}, {tile_y})")

        try:
            panos = streetview.get_coverage_tile(tile_x, tile_y)

            if not panos:
                empty_tiles += 1
                logger.debug(f"Street View: No panoramas in tile {tile_x}, {tile_y}")
            else:
                logger.debug(f"Street View: Found {len(panos)} panoramas in tile {tile_x}, {tile_y}")

            for pano in panos:
                if pano.id in downloaded_ids:
                    logger.debug(f"Street View: Skipping already downloaded panorama {pano.id}")
                    continue

                downloaded_ids.add(pano.id)
                logger.info(f"Street View: Downloading panorama {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}")

                try:
                    full_pano = streetview.find_panorama_by_id(pano.id)

                    if full_pano is None:
                        errors += 1
                        logger.error(f"Street View: Could not find full panorama data for {pano.id}")
                        continue

                    # Dateiname aus Template generieren
                    # CaptureDate zu String konvertieren
                    if full_pano.date:
                        try:
                            # Versuche datetime-Objekt
                            date_str = full_pano.date.strftime("%Y%m%d")
                        except AttributeError:
                            # CaptureDate-Objekt: Konvertiere zu String
                            date_str = str(full_pano.date).replace("-", "").split()[0]
                    else:
                        date_str = "unknown"

                    filename = filename_template.format(
                        id=full_pano.id,
                        lat=full_pano.lat,
                        lon=full_pano.lon,
                        date=date_str
                    )
                    full_path = os.path.join(output_dir, filename)

                    logger.debug(f"Street View: Saving panorama {pano.id} to {filename}")
                    streetview.download_panorama(full_pano, full_path)
                    total_panos += 1
                    logger.info(f"Street View: Successfully downloaded panorama {pano.id}")

                except Exception as e:
                    errors += 1
                    logger.error(f"Street View: Error downloading panorama {pano.id}: {str(e)}")
                    # Debug: Fehler loggen
                    try:
                        with open(os.path.join(output_dir, "streetview_errors.log"), "a", encoding="utf-8") as f:
                            f.write(f"Fehler bei Pano {pano.id}: {str(e)}\n")
                    except:
                        pass

            completed += 1
            progress.update(
                task_id,
                advance=1,
                description=f"[cyan]Street View [{total_panos} Panos, {completed}/{total_tiles} Tiles]"
            )

        except Exception as e:
            completed += 1
            errors += 1
            logger.error(f"Street View: Error processing tile {tile_x}, {tile_y}: {str(e)}")
            progress.update(task_id, advance=1)

    result_text = f"[bold green]✓ Street View ({total_panos} Panoramen"
    if errors > 0:
        result_text += f", {errors} Fehler"
    result_text += ")"
    progress.update(task_id, description=result_text)


def lookaround_task(progress, task_id, tiles, output_dir, filename_template, zoom=0):
    """Worker-Funktion zum Download von Look Around-Panoramen."""
    global logger

    total_tiles = len(tiles)
    completed = 0
    total_panos = 0
    downloaded_ids = set()
    errors = 0
    equirect_failures = 0

    progress.update(task_id, total=total_tiles, description=f"[magenta]Look Around [0/0 Panos, 0/0 Tiles, Zoom {zoom}]")

    auth = lookaround.Authenticator()

    for tile_idx, (tile_x, tile_y) in enumerate(tiles):
        logger.debug(f"Look Around: Processing tile {tile_idx+1}/{total_tiles} ({tile_x}, {tile_y})")

        try:
            tile = lookaround.get_coverage_tile(tile_x, tile_y)

            if not tile.panos:
                logger.debug(f"Look Around: No panoramas in tile {tile_x}, {tile_y}")
            else:
                logger.debug(f"Look Around: Found {len(tile.panos)} panoramas in tile {tile_x}, {tile_y}")

            for pano in tile.panos:
                if pano.id in downloaded_ids:
                    logger.debug(f"Look Around: Skipping already downloaded panorama {pano.id}")
                    continue

                downloaded_ids.add(pano.id)
                logger.info(f"Look Around: Processing panorama {pano.id} at {pano.lat:.6f}, {pano.lon:.6f}")

                try:
                    # Dateiname aus Template generieren
                    # CaptureDate zu String konvertieren
                    if pano.date:
                        try:
                            # Versuche datetime-Objekt
                            date_str = pano.date.strftime("%Y%m%d")
                        except AttributeError:
                            # CaptureDate-Objekt: Konvertiere zu String
                            date_str = str(pano.date).replace("-", "").split()[0]
                    else:
                        date_str = "unknown"

                    filename = filename_template.format(
                        id=pano.id,
                        lat=pano.lat,
                        lon=pano.lon,
                        date=date_str
                    )
                    full_path = os.path.join(output_dir, filename)

                    logger.debug(f"Look Around: Downloading 6 faces for panorama {pano.id}")

                    # Alle 6 Faces herunterladen
                    faces = []
                    temp_files = []

                    for face_idx in range(6):
                        temp_path = os.path.join(output_dir, f"temp_{pano.id}_face{face_idx}.heic")
                        temp_files.append(temp_path)
                        logger.debug(f"Look Around: Downloading face {face_idx} for panorama {pano.id}")
                        lookaround.download_panorama_face(pano, temp_path, face_idx, zoom, auth)
                        face_img = Image.open(temp_path)
                        faces.append(face_img)

                    logger.info(f"Look Around: Downloaded all 6 faces for panorama {pano.id}")

                    # Zu Equirectangular umwandeln
                    try:
                        logger.debug(f"Look Around: Converting panorama {pano.id} to equirectangular")
                        equirect = lookaround.to_equirectangular(faces, pano.camera_metadata)
                        equirect.save(full_path, quality=95)
                        logger.info(f"Look Around: Successfully converted panorama {pano.id} to equirectangular")

                        # Nur bei Erfolg: Temporäre Dateien löschen
                        for temp_file in temp_files:
                            try:
                                os.remove(temp_file)
                                logger.debug(f"Look Around: Removed temp file {temp_file}")
                            except Exception as e:
                                logger.warning(f"Look Around: Could not remove temp file {temp_file}: {e}")

                        for face in faces:
                            face.close()

                        total_panos += 1

                    except Exception as e:
                        # Fallback: Speichere alle 6 Faces einzeln
                        equirect_failures += 1
                        logger.warning(f"Look Around: Equirectangular conversion failed for {pano.id}: {str(e)}")
                        logger.info(f"Look Around: Saving individual faces for panorama {pano.id}")

                        for face_idx, face in enumerate(faces):
                            face_path = os.path.join(output_dir, f"{pano.id}_face{face_idx}.jpg")
                            face.save(face_path, quality=95)
                            logger.debug(f"Look Around: Saved face {face_idx} for panorama {pano.id}")
                            face.close()

                        # Temporäre HEIC-Dateien löschen (bereits als JPG gespeichert)
                        for temp_file in temp_files:
                            try:
                                os.remove(temp_file)
                                logger.debug(f"Look Around: Removed temp HEIC file {temp_file}")
                            except Exception as e:
                                logger.warning(f"Look Around: Could not remove temp HEIC file {temp_file}: {e}")

                        total_panos += 1

                except Exception as e:
                    errors += 1
                    logger.error(f"Look Around: Error processing panorama {pano.id}: {str(e)}")

            completed += 1
            progress.update(
                task_id,
                advance=1,
                description=f"[magenta]Look Around [{total_panos} Panos, {completed}/{total_tiles} Tiles, Zoom {zoom}]"
            )

        except Exception as e:
            completed += 1
            errors += 1
            logger.error(f"Look Around: Error processing tile {tile_x}, {tile_y}: {str(e)}")
            progress.update(task_id, advance=1)

    result_text = f"[bold green]✓ Look Around ({total_panos} Panoramen"
    if errors > 0:
        result_text += f", {errors} Fehler"
    if equirect_failures > 0:
        result_text += f", {equirect_failures} Konvertierungsfehler"
    result_text += ")"
    progress.update(task_id, description=result_text)


def get_coordinates():
    """Interaktive Eingabe von Koordinaten."""
    print("\nKoordinaten-Eingabe")
    print("Geben Sie mindestens 2 Koordinatenpaare ein")
    print("\nUnterstützte Formate:")
    print("  • Dezimal:  52.520008,13.404954")
    print("  • DMS:      52°30'57.6\"N 13°20'50.7\"E")
    print("\nTipp: Enter ohne Eingabe = Zufällige Koordinaten in Berlin\n")
    
    coords = []
    first_empty = False
    
    while True:
        prompt_text = f"Koordinate {len(coords) + 1}"
        if len(coords) == 0:
            prompt_text += " (oder Enter für Zufall)"
        
        coord_input = input(prompt_text + ": ").strip()
        
        # Leere Eingabe
        if not coord_input:
            if len(coords) == 0:
                # Erste Eingabe leer = Zufallskoordinaten
                coords = generate_random_berlin_coords()
                print("Zufällige Koordinaten generiert:")
                print(f"   Punkt 1: {coords[0][0]:.6f}, {coords[0][1]:.6f}")
                print(f"   Punkt 2: {coords[1][0]:.6f}, {coords[1][1]:.6f}")
                break
            elif len(coords) >= 2:
                break
            else:
                print("Mindestens 2 Koordinaten erforderlich!")
                continue
        
        # Koordinate parsen
        try:
            lat, lon = parse_coordinate(coord_input)
            coords.append((lat, lon))
            print(f"Hinzugefügt: {lat:.6f}, {lon:.6f}")
        except ValueError as e:
            print(f"Fehler: {str(e)}")
            print("Verwenden Sie eines der unterstützten Formate!")
    
    return coords


def get_filename_template():
    """Interaktive Eingabe des Dateinamen-Templates."""
    print("\nDateiname-Template")
    print("Verfügbare Variablen:")
    print("  {id}   - Panorama ID")
    print("  {lat}  - Breitengrad")
    print("  {lon}  - Längengrad")
    print("  {date} - Aufnahmedatum (YYYYMMDD)")
    
    # Beispiel mit Berlin-Koordinate
    example_template = "{id}_{lat}_{lon}.jpg"
    print(f"\nBeispiel-Template: {example_template}")
    print(f"Beispiel-Ausgabe:  ABC123_52.520008_13.404954.jpg")
    
    template = input(f"\nDateiname-Template (default: {example_template}): ").strip()
    if not template:
        template = example_template
    
    # Validierung
    if not template.endswith(('.jpg', '.jpeg', '.png')):
        template += '.jpg'
        print(f"Hinweis: .jpg wurde hinzugefügt → {template}")
    
    return template


def main():
    """Hauptfunktion - Interaktives CLI."""
    global logger

    # Console für Rich-Ausgaben
    console = Console()

    # Header
    console.print("Panorama Downloader", style="bold cyan")
    console.print("Street View & Look Around")
    console.print("=" * 40)

    # 1. Koordinaten eingeben
    coords = get_coordinates()

    # Tiles berechnen (Bounding Box aus allen Koordinaten)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    tiles = get_tiles_in_bbox(min(lats), min(lons), max(lats), max(lons))

    # Zusammenfassung anzeigen
    console.print(f"\nEingabe-Zusammenfassung:", style="bold")
    console.print(f"Koordinaten: {len(coords)} Punkte")
    console.print(f"Bounding Box: {min(lats):.6f},{min(lons):.6f} bis {max(lats):.6f},{max(lons):.6f}")
    console.print(f"Tiles (Zoom 17): {len(tiles)}")

    # 2. Download-Modus wählen
    console.print("\nDownload-Modus", style="bold")
    download_streetview = input("Street View herunterladen? (y/n, default: y): ").strip().lower()
    download_streetview = download_streetview in ('y', 'yes', '')

    download_lookaround = input("Look Around herunterladen? (y/n, default: y): ").strip().lower()
    download_lookaround = download_lookaround in ('y', 'yes', '')

    if not download_streetview and not download_lookaround:
        console.print("Keine Download-Option gewählt. Programm wird beendet.")
        return

    # 3. Dateiname-Template
    filename_template = get_filename_template()

    # 4. Look Around Qualität (nur wenn Look Around aktiviert)
    lookaround_zoom = 0  # Default: Höchste Qualität
    if download_lookaround:
        console.print("\nLook Around Qualität", style="bold")
        console.print("Zoom Level (0-7, niedriger = bessere Qualität aber langsamer):")
        console.print("  • 0: Höchste Qualität")
        console.print("  • 1-3: Hoch")
        console.print("  • 4-5: Mittel")
        console.print("  • 6-7: Niedrig (schnell)")

        zoom_input = input("Zoom Level (default: 0): ").strip()
        if zoom_input:
            try:
                lookaround_zoom = int(zoom_input)
                if lookaround_zoom < 0 or lookaround_zoom > 7:
                    console.print("Ungültiger Wert, verwende Standard (0 = Höchste Qualität)")
                    lookaround_zoom = 0
            except ValueError:
                console.print("Ungültiger Wert, verwende Standard (0 = Höchste Qualität)")
                lookaround_zoom = 0

    # 5. Session vorbereiten
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = timestamp
    os.makedirs(base_dir, exist_ok=True)

    # Logging initialisieren
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

    # Start-Info
    console.print(f"\nDownload startet...", style="bold green")
    console.print(f"Session: {timestamp}")
    console.print(f"Output: {base_dir}/")
    console.print(f"Log file: {base_dir}/panorama_downloader.log")

    # Warnung für Look Around Equirectangular-Konvertierung
    if download_lookaround:
        console.print("\nHinweis: Look Around Bilder werden als 6 einzelneFaces heruntergeladen.", style="yellow")
        console.print("Die automatische Zusammenführung zu 360°-Panoramen erfordert PyTorch.")
        console.print("Ohne PyTorch werden dieFaces einzeln als JPG gespeichert.")

        try:
            import torch
            console.print("✓ PyTorch ist installiert - Equirectangular-Konvertierung verfügbar", style="green")
        except ImportError:
            console.print("⚠ PyTorch ist nicht installiert -Faces werden einzeln gespeichert", style="yellow")
            console.print("  Installieren Sie PyTorch für vollständige 360°-Panoramen:")
            console.print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    # 6. Download mit Progress-Bar
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

    # Abschluss
    console.print(f"\nDownload abgeschlossen!", style="bold green")
    console.print(f"Ergebnisse gespeichert in: {base_dir}/")
    console.print(f"Detailliertes Log: {base_dir}/panorama_downloader.log")
    logger.info("Panorama Downloader session completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload abgebrochen.")
        sys.exit(0)
