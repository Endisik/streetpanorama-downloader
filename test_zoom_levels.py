"""
Test Script für Look Around Zoom Levels
Testet verschiedene Zoom-Level und zeigt die Ergebnisse
"""

import os
import sys
from streetlevel import lookaround
from PIL import Image
from pillow_heif import register_heif_opener

# HEIC Support aktivieren
register_heif_opener()

def test_zoom_levels():
    """Testet verschiedene Zoom-Level für Look Around."""
    # Test-Koordinaten (Berlin)
    test_lat, test_lon = 52.520008, 13.404954

    print("🔍 Teste Look Around Zoom-Level...")
    print(f"📍 Test-Koordinaten: {test_lat}, {test_lon}")

    # Zoom-Level testen
    zoom_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for zoom in zoom_levels:
        print(f"\n🧪 Teste Zoom Level {zoom}...")

        try:
            # Coverage Tile abrufen
            tile = lookaround.get_coverage_tile_by_latlon(test_lat, test_lon)

            if not tile.panos:
                print(f"   ❌ Keine Panoramen bei Zoom {zoom}")
                continue

            pano = tile.panos[0]
            print(f"   📷 Gefunden: {pano.id}")

            # Authenticator erstellen
            auth = lookaround.Authenticator()

            # Test: Eine Face herunterladen
            temp_path = f"test_zoom_{zoom}_face0.heic"

            try:
                lookaround.download_panorama_face(pano, temp_path, 0, zoom, auth)

                # Bild öffnen und Größe prüfen
                img = Image.open(temp_path)
                width, height = img.size
                print(f"   ✅ Erfolgreich: {width}x{height} Pixel")

                # Aufräumen
                os.remove(temp_path)

            except Exception as e:
                print(f"   ❌ Fehler beim Download: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            print(f"   ❌ Fehler bei Zoom {zoom}: {str(e)}")

if __name__ == "__main__":
    test_zoom_levels()
