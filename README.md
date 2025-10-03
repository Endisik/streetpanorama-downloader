# Panorama Downloader

An interactive downloader for 360° panoramas from street-level imagery. Download high-resolution 360° images based on coordinates.

## ✨ Features

- 🌍 **Dual-Source Support**: Google Street View & Apple Look Around
- 📍 **Flexible Input**: 
  - Decimal coordinates (52.520652, 13.409052)
  - DMS format (52°31'14.4"N 13°24'32.6"E)
  - At least 2 coordinate pairs are required in decimal or DMS format. More than 2 coordinate pairs will be interpreted as a polygon.
- 🎨 **Customizable Exportnames**: Merge Variables for custom exportnames ({id}, {lat}, {lon}, {date})

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
python panorama_downloader.py
```

### 2. Workflow

1. **Enter coordinates**
   - At least 2 coordinate-pairs in 2 possible formats:
     - Decimal: `52.520008,13.404954`
     - DMS: `52°30'57.6"N 13°20'50.7"E`
   - Leave empty to get a random coordinate-pair within a small rectangle in Berlin ᴰᴱ

2. **Choose imagery sources**
   - Download Street View? (Yes/No)
   - Download Look Around? (Yes/No)
3. **Set filename template**
   - Default: `{id}_{lat}_{lon}.jpg`
   - Variables: `{id}`, `{lat}`, `{lon}`, `{date}`

4. **Choose Look Around quality**
   - Zoom Level 0-7 (lower = better quality but slower)
   - Default: 0 (highest quality)
   - Higher zoom levels result in lower resolution

**Download starts**
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

- **Street View**: JPG (direct, highest quality)
- **Look Around**: HEIC → JPG (automatic conversion)
- **Resolution**: Configurable Zoom Level 0-7 (default: 0 = highest quality)

### Equirectangular Projection

Look Around panoramas are assembled from 6 cube faces into an equirectangular 360° image.

## 🐛 Troubleshooting

### No panoramas found

**Possible causes:**
- Coordinates are outside Street View/Look Around coverage
- Bounding box too small (increase the area)
- Network issues

**Tip:** Check coverage on ᴳᵒᵒᵍˡᵉ ᴹᵃᵖˢ [Street View](https://sv-map.netlify.app/) or ᴬᵖᵖˡᵉ ᴹᵃᵖˢ [Look Around](https://lookmap.eu.pythonanywhere.com/).

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request.

## ⚠️ Disclaimer

- Respect the terms of service of imagery providers
- Use the software responsibly
- Large downloads may consume time and bandwidth
- Some regions have no imagery coverage

## 📧 Support

For questions or issues, please open an Issue.

---
