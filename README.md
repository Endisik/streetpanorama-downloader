# Panorama Downloader

An interactive downloader for 360° panoramas from street-level imagery. Download high-resolution 360° images based on coordinates.

## ✨ Features

- 🌍 **Dual-Source Support**: Google Street View & Apple Look Around
- 📍 **Flexible Input**: 
  - Decimal coordinates (52.520652, 13.409052)
  - DMS format (52°31'14.4"N 13°24'32.6"E)
  - At least 2 coordinate pairs are required in decimal or DMS format. More than 2 coordinate pairs will be interpreted as a polygon.
- 🎨 **Customizable Exportnames**: Merge Variables for custom exportnames ({id}, {lat}, {lon}, {date}) More Coming soon ({heading}, {pitch}, {roll}, {elevation}, {country_code}, {source})

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/panorama-downloader.git
cd panorama-downloader
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install streetlevel --no-deps
```

> **Note:** The first command may fail with "pyfrpc" - this is expected and can be ignored. Therefore, the second command completes the installation.

## 📖 Usage

### 1. Start the program

```bash
python streetpanorama.py
```

### 2. Workflow

1. **Enter coordinates**
   - At least 2 coordinate-pairs in 2 possible formats:
     - Decimal: `52.520008,13.404954`
     - DMS: `52°30'57.6"N 13°20'50.7"E`
   - Leave empty to get a random coordinate-pair within a small rectangle in Berlin ᴰᴱ
  
> [!NOTE]
> The Coordinate-Area may download more panoramas than you expect. Panoramas will download in chunks as [Map-Tiles of Zoom-Level 17](https://labs.mapbox.com/what-the-tile/). <br>
  > 🟢 Look Around imagery<br>
  > 🔵 Street View imagery<br>
  > 🔴 Input Coordinates

> <img width="834" height="803" alt="far" src="https://github.com/user-attachments/assets/1621208f-dd92-4919-b743-26ce81cc1645" />
> <img width="834" height="803" alt="near" src="https://github.com/user-attachments/assets/bed136ba-755a-4669-87bc-e53fc96875a4" />

2. **Choose imagery sources**
   - Download Street View? (Yes/No)
   - Download Look Around? (Yes/No)
3. **Set filename template**
   - Default: `{lat}_{lon}_{date}.jpg`
   - Variables: `{id}`, `{lat}`, `{lon}`, `{date}`

4. **Choose Look Around quality**
   - Zoom Level 0-7 (lower = better resolution, slower Equirectangular convertion)
   - Default: 1 (high quality)
   - Zoom examples (highly zoomed in) (example picture on different resolutions. [Free to Use](https://polyhaven.com/a/modern_evening_street).)
     - Zoom Level 0 (16384 x 8192; ≈21.000 KB)
     <img width="430" height="325" alt="0" src="https://github.com/user-attachments/assets/d978f1fe-9a5b-42e5-86f0-6a144fba038b" />

     - Zoom Level 1 (12016 x 6008; ≈12.000 KB)
     <img width="430" height="325" alt="1" src="https://github.com/user-attachments/assets/34e0bb67-d7c1-4c2b-bcf0-c7724d834dc5" />

     - Zoom Level 2 (6416 x 3208; ≈4.100 KB)
     <img width="430" height="325" alt="2" src="https://github.com/user-attachments/assets/4f8fa61f-4bc6-47fb-9e98-77b1982721e6" />

     - Zoom Level 3 (4560 x 2280; ≈2.100 KB)
     <img width="430" height="325" alt="3" src="https://github.com/user-attachments/assets/d68ae478-941a-4ebb-a600-d64b12e30a4d" />

     - Zoom Level 4 (3216 x 1608; ≈1.000 KB)
     <img width="430" height="325" alt="4" src="https://github.com/user-attachments/assets/910b17b6-cab4-48ed-85fe-6cddb5b3bfd1" />

     - Zoom Level 5 (2288 x 1144; ≈550 KB)
     <img width="430" height="325" alt="5" src="https://github.com/user-attachments/assets/5a7c7f70-e5f2-4e7b-bac6-98e3f8cdcb69" />

     - Zoom Level 6 (1632 x 816; ≈290 KB)
     <img width="430" height="325" alt="6" src="https://github.com/user-attachments/assets/9f8adffa-f0e7-4d2a-98ac-1ba7958ce842" />

     - Zoom Level 7 (1168 x 584; ≈160 KB)
     <img width="430" height="325" alt="7" src="https://github.com/user-attachments/assets/1e32ddca-fe10-4015-8f80-25a2c77f9cf6" />

5. **Download starts**
   - Live progressbar

### Output

All downloads are saved in a timestamped folder:
```
YYYYMMDDHHMMSS/
├── Panoramas_StreetView/
│   ├── {template}.jpg
│   └── ...
├── Panoramas_LookAround/
│   ├── {template}.jpg
│   └── ...
└── ...
```

## 🛠️ Technical Details

### Image Formats

- **Street View**: JPG
- **Look Around**: 6 HEIC cube faces → JPG (automatic conversion)

## 🐛 Troubleshooting

### No panoramas found

**Possible causes:**
- Coordinates are outside Street View/Look Around coverage
- Bounding box too small (increase the area)
- Network issues

**Tip:** Check coverage on ᴳᵒᵒᵍˡᵉ ᴹᵃᵖˢ [Street View](https://sv-map.netlify.app/) and/or ᴬᵖᵖˡᵉ ᴹᵃᵖˢ [Look Around](https://lookmap.eu.pythonanywhere.com/).

## 🤝 Contributing

Contributions are welcome! Please open an [issue](https://github.com/Endisik/streetpanorama-downloader/issues) or [pull request](https://github.com/Endisik/streetpanorama-downloader/pulls).

## ⚠️ Disclaimer

- Respect the terms of service of imagery providers
- Use the software responsibly
- Large downloads may consume time and bandwidth
- Some regions have no imagery coverage

## 📧 Support

For questions or issues, please open an [Issue](https://github.com/Endisik/streetpanorama-downloader/issues).

---
