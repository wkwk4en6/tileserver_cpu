# Tile Server & Map Viewer

[Japanese](README_ja.md)

A lightweight tile server and map viewer designed to generate and serve map tiles from PMTiles data for display in web browsers via Leaflet.

---

## Overview

- **Tile Serving**: Loads PMTiles data, renders vector tiles, and serves them to clients.
- **Caching Mechanism**: High-speed tile delivery powered by SQLite and file-based caching.
- **Frontend**: Interactive map viewer built with Leaflet.
- **Grid & Debug Display**: Built-in support for displaying zoom levels ($z$) and tile coordinates ($x, y$) for debugging.

---

## Requirements

- Python 3.12+
- (Refer to `requirements.txt` for third-party dependencies)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/wkwk4en6/tileserver_cpu.git
cd tileserver_cpu
```

### 2. Create Virtual Environment & Install Dependencies
This tool is intended for forensic investigation use cases. We recommend using Miniconda or Anaconda for superior environment isolation, reproducibility, and binary dependency management.

Using Conda (Recommended):
```Bash
# Create virtual environment
conda create -n tile-server python=3.12 -y
conda activate tile-server

# Install dependencies
pip install -r requirements.txt
```
Using pyenv / venv:
```Bash
python -m venv .venv
source .venv/bin/activate  # On Windows: `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Directory Setup & PMTiles Data Placement
PMTiles data is required to run this server. Create a world-pmtiles/ directory and place your data files inside it by following the steps below.

#### Create the world-pmtiles Directory
```Bash
mkdir world-pmtiles
```

#### Prepare Low-Zoom / Overview Data
Use the `pmtiles` CLI tool to extract low-zoom data (zoom levels 0–7) from Protomaps official remote builds. The output file size is approximately 200 MB.

On Windows, download the pre-compiled binary from [protomaps/go-pmtiles](https://github.com/protomaps/go-pmtiles/releases) releases.
```Bash
pmtiles extract https://build.protomaps.com/2026xxxx.pmtiles world-pmtiles/planet_x0-z7.pmtiles --maxzoom=7
```
_Note: Replace 2026xxxx in the URL with the date of the latest build as needed._

### Download Detailed Map Data (BBBike)
- Visit BBBike extracts: https://data.bbbike.org/osm/region/

- Select your target area, choose PM Vector tiles Shortbread as the format, and download the dataset and unzip.

- Place the unzipped .pmtiles file into the world-pmtiles/ directory.

## Usage
Start the tile server:
```Bash
python tileserver_cpu.py
```
### Once running, open your browser and navigate to:`http://localhost:8990`


## Directory Structure
```Plaintext
.
├── assets/                # Frontend assets (Leaflet, CSS, Fonts, etc.)
│   ├── leaflet.js
│   ├── leaflet.css
│   └── fonts/NotoSansCJK-Regular/NotoSansCJK-Regular.ttc             
├── world-pmtiles/         # Directory for storing PMTiles data
│   ├── planet_x0-z7.pmtiles
│   └── *.pmtiles
├── cache_tiles/           # Generated tile cache (Auto-created on server start)
├── tile_cache.db          # SQLite cache database (Auto-created on server start)
├── tileserver_cpu.py      # Main tile server script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Licenses & Acknowledgments
License details for this project and its integrated third-party assets:

- Leaflet: BSD 2-Clause License
- Noto Sans CJK: SIL Open Font License 1.1
- Map Data: © OpenStreetMap contributors
