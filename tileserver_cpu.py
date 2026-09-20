import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
import gzip
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import List, Optional, Tuple

from fastapi import FastAPI, Query, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader
import skia
import uvicorn

BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "tile_cache.db"
TILES_DIR = BASE_DIR / "cache_tiles"


# --- PMTiles のメタデータとメモリキャッシング構造 ---
class LoadedPMTiles:
    def __init__(self, name: str, path: Path, file_obj, reader: Reader, header: dict):
        self.name = name
        self.path = path
        self.file_obj = file_obj
        self.reader = reader
        self.min_zoom = header.get("min_zoom", 0)
        self.max_zoom = header.get("max_zoom", 30)

        # e7 形式で保存されている緯度経度を通常の Float 度数に変換
        self.min_lon = header.get("min_lon_e7", -1800000000) / 1e7
        self.min_lat = header.get("min_lat_e7", -90000000) / 1e7
        self.max_lon = header.get("max_lon_e7", 1800000000) / 1e7
        self.max_lat = header.get("max_lat_e7", 90000000) / 1e7

    def intersects_tile(self, z: int, x: int, y: int) -> bool:
        """指定された z/x/y タイルがこの pmtiles のズーム範囲および地理範囲と交差するか判定"""
        if not (self.min_zoom <= z <= self.max_zoom):
            return False

        n = 1 << z
        tile_min_lon = x / n * 360.0 - 180.0
        tile_max_lon = (x + 1) / n * 360.0 - 180.0

        def tile2lat(y_val: int, z_val: int) -> float:
            n_val = math.pi - (2.0 * math.pi * y_val) / (1 << z_val)
            return math.degrees(math.atan(math.sinh(n_val)))

        tile_max_lat = tile2lat(y, z)
        tile_min_lat = tile2lat(y + 1, z)

        # 境界での僅かな誤差を許容するためマージンを持たせる
        margin = 1e-5
        if (tile_max_lon + margin) < self.min_lon or (tile_min_lon - margin) > self.max_lon:
            return False
        if (tile_max_lat + margin) < self.min_lat or (tile_min_lat - margin) > self.max_lat:
            return False

        return True


priority_pmtiles: List[LoadedPMTiles] = []
normal_pmtiles: List[LoadedPMTiles] = []

executor = ProcessPoolExecutor()

# このヘッダーは、`/tile/{z}/{x}/{y}.png` エンドポイントから PNG タイル画像を返却する際に HTTP レスポンスヘッダーとして付与されます。
# これにより、一度読み込んだタイル画像は 24 時間ブラウザ側にキャッシュされ、無駄な再リクエストを防ぐ仕組みになっています。
CACHE_HEADERS = {"Cache-Control": "public, max-age=86400"}

# 色の定数定義
LAND_COLOR = skia.Color(245, 243, 240, 255)
WATER_COLOR = skia.Color(170, 211, 223, 255)
GREEN_COLOR = skia.Color(200, 228, 195, 255)
BUILDING_COLOR = skia.Color(220, 215, 208, 255)
BUILDING_STROKE = skia.Color(190, 185, 178, 255)
ROAD_COLOR = skia.Color(255, 255, 255, 255)
LINE_COLOR = skia.Color(200, 200, 200, 255)

GREEN_KEYWORDS = {
    "park", "forest", "wood", "grass", "green", "meadow", "garden",
    "golf_course", "recreation_ground", "cemetery", "nature_reserve",
    "pitch", "playground", "leisure", "national_park", "farmland"
}


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tile_metadata (
                z INTEGER,
                x INTEGER,
                y INTEGER,
                file_path TEXT,
                created_at REAL,
                last_accessed REAL,
                PRIMARY KEY (z, x, y)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def get_tile_file_path(z: int, x: int, y: int) -> Optional[str]:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_path FROM tile_metadata WHERE z=? AND x=? AND y=?",
            (z, x, y)
        )
        row = cursor.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            return row[0]
        return None
    except Exception as e:
        print(f"[DB Read Error] {e}")
        return None
    finally:
        conn.close()


def register_tile_to_db(z: int, x: int, y: int, file_path: str):
    try:
        now = time.time()
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO tile_metadata (z, x, y, file_path, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (z, x, y, file_path, now, now)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB Write Error] {e}")


def save_png_file(z: int, x: int, y: int, data: bytes) -> str:
    tile_dir = TILES_DIR / str(z) / str(x)
    tile_dir.mkdir(parents=True, exist_ok=True)
    file_path = tile_dir / f"{y}.png"
    with open(file_path, "wb") as f:
        f.write(data)
    return str(file_path)


def create_empty_tile_png() -> bytes:
    surface = skia.Surface(256, 256)
    surface.getCanvas().clear(LAND_COLOR)
    image = surface.makeImageSnapshot()
    return image.encodeToData().bytes()


EMPTY_TILE_BYTES = create_empty_tile_png()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    for folder in ["world-pmtiles"]:
        dir_path = BASE_DIR / folder
        if dir_path.exists():
            for p in dir_path.rglob("*.pmtiles"):
                if any(part.startswith(".") for part in p.parts):
                    continue
                try:
                    f = open(p, "rb")
                    source = MmapSource(f)
                    reader = Reader(source)
                    header = reader.header()
                    item = LoadedPMTiles(p.stem, p, f, reader, header)
                    if p.name == "planet_z0-z7.pmtiles":
                        priority_pmtiles.append(item)
                    else:
                        normal_pmtiles.append(item)
                except Exception as e:
                    print(f"[Error] Failed to load {p.name}: {e}")

    yield

    for item in priority_pmtiles + normal_pmtiles:
        try:
            item.file_obj.close()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


def fetch_pbf_from_filepath(
    priority_list: List[LoadedPMTiles],
    normal_list: List[LoadedPMTiles],
    z: int, x: int, y: int
) -> Tuple[List[bytes], int, int, int, List[str]]:
    """
    該当するすべての PMTiles から PBF データを取得してリストで返す
    """
    max_tile = 1 << z
    if x < 0 or x >= max_tile or y < 0 or y >= max_tile:
        return [], z, x, y, []

    pbf_list = []
    used_filenames = []

    for dz in range(0, z + 1):
        curr_z = z - dz
        curr_x = x >> dz
        curr_y_xyz = y >> dz
        curr_y_tms = (1 << curr_z) - 1 - curr_y_xyz

        candidate_priority = [p for p in priority_list if p.intersects_tile(curr_z, curr_x, curr_y_xyz)]
        candidate_normal = [p for p in normal_list if p.intersects_tile(curr_z, curr_x, curr_y_xyz)]

        for pmtile in candidate_priority + candidate_normal:
            for check_y in [curr_y_xyz, curr_y_tms]:
                try:
                    tile_data = pmtile.reader.get(curr_z, curr_x, check_y)
                    if tile_data and len(tile_data) > 0:
                        if tile_data[:2] == b"\x1f\x8b":
                            tile_data = gzip.decompress(tile_data)
                        pbf_list.append(tile_data)
                        used_filenames.append(pmtile.path.name)
                        break # この PMTiles から取得できたら次の PMTiles の判定へ
                except Exception:
                    continue

        # 現在のズームレベルで 1 つ以上の PMTiles からデータが得られたらそのズームで確定
        if pbf_list:
            return pbf_list, curr_z, curr_x, curr_y_xyz, used_filenames

    return [], z, x, y, []


def render_3x3_tile_skia(
    target_z: int,
    target_x: int,
    target_y: int,
    pbf_tiles_data: List[Tuple[int, int, Optional[bytes], int, int, int]]
) -> bytes:
    # z >= 18 のときは 3x3 (768x768)、z < 18 のときは 1x1 (256x256)
    is_multi_tile = target_z >= 18
    canvas_size = 768 if is_multi_tile else 256
    tile_size = 256.0

    base_surface = skia.Surface(canvas_size, canvas_size)
    base_canvas = base_surface.getCanvas()
    base_canvas.clear(LAND_COLOR)

    render_text = target_z >= 18
    if render_text:
        label_surface = skia.Surface(canvas_size, canvas_size)
        label_canvas = label_surface.getCanvas()
        label_canvas.clear(skia.ColorTRANSPARENT)

        font_path = BASE_DIR / "assets" / "fonts" / "NotoSansCJK-Regular" / "NotoSansCJK-Regular.ttc"
        if font_path.exists():
            typeface = skia.Typeface.MakeFromFile(str(font_path))
        else:
            typeface = skia.Typeface.MakeFromName("sans-serif", skia.FontStyle.Normal())

        font = skia.Font(typeface, 11)
        paint_text = skia.Paint(Color=skia.Color(50, 50, 50, 255), AntiAlias=True)
        paint_text_halo = skia.Paint(Color=skia.Color(255, 255, 255, 230), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0)
        placed_boxes: List[skia.Rect] = []

        def is_colliding(new_rect: skia.Rect) -> bool:
            padded_rect = skia.Rect.MakeXYWH(
                new_rect.left() - 4, new_rect.top() - 4,
                new_rect.width() + 8, new_rect.height() + 8
            )
            return any(skia.Rect.Intersects(padded_rect, box) for box in placed_boxes)

    paint_water = skia.Paint(Color=WATER_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)
    paint_green = skia.Paint(Color=GREEN_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)
    paint_land = skia.Paint(Color=LAND_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)

    paint_building_fill = skia.Paint(Color=BUILDING_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)
    paint_building_stroke = skia.Paint(Color=BUILDING_STROKE, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.5)

    paint_expressway = skia.Paint(Color=skia.Color(255, 140, 0, 255), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0)
    paint_primary = skia.Paint(Color=skia.Color(255, 215, 0, 255), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.2)
    paint_secondary = skia.Paint(Color=skia.Color(250, 235, 150, 255), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.8)
    paint_road = skia.Paint(Color=ROAD_COLOR, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.9)
    paint_line = skia.Paint(Color=LINE_COLOR, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.8)

    # 描画優先順位の定義（背景陸地 -> 水域 -> グリーン・施設 -> 建物 -> 道路 -> ラベル）
    def get_layer_priority(name: str) -> int:
        n = name.lower()
        if "earth" in n or "land" in n or "boundary" in n: return 5
        if any(k in n for k in ["water", "ocean", "river", "lake"]): return 10
        if any(k in n for k in ["landcover", "landuse", "park", "green", "leisure"]): return 20
        if "building" in n: return 40
        if any(k in n for k in ["road", "street", "transportation", "highway"]): return 50
        if any(k in n for k in ["place", "poi", "address", "label", "location"]): return 60
        return 25

    tiles_to_process = []
    for dx, dy, pbf_bytes_list, actual_z, actual_x, actual_y in pbf_tiles_data:
        for pbf_data in pbf_bytes_list:
            if pbf_data:
                try:
                    decoded = mapbox_vector_tile.decode(pbf_data, y_coord_down=True)
                    tiles_to_process.append((dx, dy, decoded, actual_z, actual_x, actual_y))
                except Exception as e:
                    print(f"[Decode Error] {e}")

    if not tiles_to_process:
        return EMPTY_TILE_BYTES

    all_layer_names = set()
    for _, _, tile_dict, _, _, _ in tiles_to_process:
        all_layer_names.update(tile_dict.keys())

    sorted_layer_names = sorted(all_layer_names, key=get_layer_priority)

    for layer_name in sorted_layer_names:
        layer_lower = layer_name.lower()

        is_label_layer = any(k in layer_lower for k in ["place", "poi", "address", "label", "location"])
        if not render_text and is_label_layer:
            continue

        is_earth_layer = "earth" in layer_lower or "land" in layer_lower
        is_water_layer = any(k in layer_lower for k in ["water", "ocean", "river", "lake"])
        is_green_layer = any(k in layer_lower for k in ["park", "forest", "green", "leisure"])
        is_building_layer = "building" in layer_lower
        is_road_layer = any(k in layer_lower for k in ["road", "street", "transportation", "highway"])

        if target_z < 11 and is_road_layer:
            continue

        for dx, dy, tile_dict, actual_z, actual_x, actual_y in tiles_to_process:
            if layer_name not in tile_dict:
                continue

            layer = tile_dict[layer_name]
            extent = float(layer.get("extent", 4096))

            dz = target_z - actual_z
            scale_factor = 1 << dz
            sub_x = (target_x + dx) - (actual_x << dz)
            sub_y = (target_y + dy) - (actual_y << dz)

            unit = extent / scale_factor
            min_x = sub_x * unit
            min_y = sub_y * unit
            scale = tile_size / unit

            # z < 18 (is_multi_tile=False) の場合はオフセット 0.0
            offset_x = (dx + 1) * tile_size if is_multi_tile else 0.0
            offset_y = (dy + 1) * tile_size if is_multi_tile else 0.0

            for feature in layer.get("features", []):
                geom_type = feature.get("geometry", {}).get("type")
                coords = feature.get("geometry", {}).get("coordinates", [])
                properties = feature.get("properties", {})

                if geom_type in ["Polygon", "MultiPolygon"]:
                    polygons = coords if geom_type == "MultiPolygon" else [coords]
                    prop_values = {str(v).lower() for v in properties.values()}
                    is_green = is_green_layer or bool(prop_values & GREEN_KEYWORDS)
                    is_water = is_water_layer or "water" in prop_values

                    fill_paint = paint_land
                    if is_building_layer: fill_paint = paint_building_fill
                    elif is_water: fill_paint = paint_water
                    elif is_green: fill_paint = paint_green
                    elif is_earth_layer: fill_paint = paint_land

                    path = skia.Path()
                    path.setFillType(skia.PathFillType.kEvenOdd)

                    for poly in polygons:
                        for ring in poly:
                            if len(ring) < 3: continue
                            px = offset_x + (ring[0][0] - min_x) * scale
                            py = offset_y + (ring[0][1] - min_y) * scale
                            path.moveTo(px, py)
                            for pt in ring[1:]:
                                px = offset_x + (pt[0] - min_x) * scale
                                py = offset_y + (pt[1] - min_y) * scale
                                path.lineTo(px, py)
                            path.close()

                    base_canvas.drawPath(path, fill_paint)
                    if is_building_layer:
                        base_canvas.drawPath(path, paint_building_stroke)

                elif geom_type in ["LineString", "MultiLineString"]:
                    rings = [coords] if geom_type == "LineString" else coords
                    line_paint = paint_line
                    if is_road_layer:
                        road_class = str(properties.get("class", properties.get("kind", properties.get("highway", "")))).lower()
                        if any(k in road_class for k in ["motorway", "expressway", "trunk"]): line_paint = paint_expressway
                        elif any(k in road_class for k in ["primary", "national"]): line_paint = paint_primary
                        elif any(k in road_class for k in ["secondary", "tertiary"]): line_paint = paint_secondary
                        else: line_paint = paint_road

                    path = skia.Path()
                    for line in rings:
                        if len(line) < 2: continue
                        px = offset_x + (line[0][0] - min_x) * scale
                        py = offset_y + (line[0][1] - min_y) * scale
                        path.moveTo(px, py)
                        for pt in line[1:]:
                            px = offset_x + (pt[0] - min_x) * scale
                            py = offset_y + (pt[1] - min_y) * scale
                            path.lineTo(px, py)
                    base_canvas.drawPath(path, line_paint)

                elif render_text and geom_type in ["Point", "MultiPoint"]:
                    def to_str(val):
                        if isinstance(val, bytes):
                            try: return val.decode("utf-8")
                            except UnicodeDecodeError: return ""
                        return str(val) if val is not None else ""

                    city_types = {
                        "country", "state", "region", "province",
                        "city", "municipality", "town", "village",
                        "district", "county"
                    }
                    detail_types = {
                        "suburb", "neighbourhood", "quarter", "block",
                        "street", "house", "building", "address", "poi"
                    }

                    place_type = to_str(
                        properties.get("place") or properties.get("class") or 
                        properties.get("subclass") or properties.get("type")
                    ).lower()

                    local_name = properties.get("name") or properties.get("name:ja") or properties.get("name_ja")
                    en_name = properties.get("name:en") or properties.get("name_en")
                    local_str = to_str(local_name)
                    en_str = to_str(en_name)

                    housenumber = to_str(properties.get("addr:housenumber") or properties.get("housenumber"))
                    street = to_str(properties.get("addr:street") or properties.get("street") or properties.get("block_number"))

                    if local_str and en_str and local_str.lower() != en_str.lower():
                        name_label = f"{local_str} ({en_str})"
                    else:
                        name_label = local_str or en_str

                    address_parts = [p for p in [street, housenumber] if p]
                    address_label = " ".join(address_parts)

                    label_str = ""
                    if place_type in detail_types or not place_type:
                        label_str = f"{name_label} ({address_label})" if name_label and address_label else (name_label or address_label)
                    elif place_type in city_types:
                        label_str = name_label
                    else:
                        label_str = name_label or address_label

                    if not label_str:
                        continue

                    points = [coords] if geom_type == "Point" else coords
                    for pt in points:
                        if len(pt) < 2: continue
                        px = offset_x + (pt[0] - min_x) * scale
                        py = offset_y + (pt[1] - min_y) * scale

                        if 0 <= px <= canvas_size and 0 <= py <= canvas_size:
                            text_width = font.measureText(label_str)
                            text_bounds = skia.Rect.MakeXYWH(px, py - 11, text_width, 11)

                            if not is_colliding(text_bounds):
                                label_canvas.drawString(label_str, px, py, font, paint_text_halo)
                                label_canvas.drawString(label_str, px, py, font, paint_text)
                                placed_boxes.append(text_bounds)

    if render_text:
        label_image = label_surface.makeImageSnapshot()
        base_canvas.drawImage(label_image, 0, 0)

    full_image = base_surface.makeImageSnapshot()

    # 3x3 描画時は中央の 256x256 をクロップ、1x1 描画時はそのまま出力
    if is_multi_tile:
        final_surface = skia.Surface(256, 256)
        final_canvas = final_surface.getCanvas()
        crop_src = skia.Rect.MakeXYWH(256, 256, 256, 256)
        crop_dst = skia.Rect.MakeWH(256, 256)
        final_canvas.drawImageRect(full_image, crop_src, crop_dst)
        image = final_surface.makeImageSnapshot()
    else:
        image = full_image

    return image.encodeToData().bytes()


ASSETS_DIR = BASE_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tile Server Map Viewer</title>
        <link rel="stylesheet" href="./assets/leaflet/leaflet.css" />
        <script src="./assets/leaflet/leaflet.js"></script>
        <style>
            html, body, #map { width: 100%; height: 100%; margin: 0; padding: 0; }
            .debug-info-control {
                background: rgba(255, 255, 255, 0.9);
                padding: 6px 10px;
                font-family: monospace;
                font-size: 12px;
                border-radius: 4px;
                box-shadow: 0 0 5px rgba(0,0,0,0.3);
                line-height: 1.5;
            }
            .debug-info-control a { color: #0066cc; text-decoration: underline; font-weight: bold; }
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            const map = L.map('map').setView([34.6873, 135.5262], 4);

            const baseTileLayer = L.tileLayer('/tile/{z}/{x}/{y}.png', {
                minZoom: 0,
                maxZoom: 18,
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);

            const tileGridLayer = L.gridLayer({ minZoom: 0, maxZoom: 18 });
            tileGridLayer.createTile = function (coords) {
                const tile = document.createElement('div');
                tile.style.boxSizing = 'border-box';
                tile.style.border = '1px solid rgba(255, 0, 0, 0.5)';
                tile.style.fontFamily = 'monospace';
                tile.style.fontSize = '11px';
                tile.style.color = 'red';
                tile.style.padding = '4px';
                tile.style.pointerEvents = 'none';
                tile.innerHTML = `z:${coords.z}<br>x:${coords.x}<br>y:${coords.y}`;
                return tile;
            };

            const overlayMaps = { "タイルグリッド (マス目)": tileGridLayer };
            L.control.layers(null, overlayMaps, { position: 'topright' }).addTo(map);

            const DebugControl = L.Control.extend({
                options: { position: 'bottomleft' },
                onAdd: function (map) {
                    const container = L.DomUtil.create('div', 'debug-info-control');
                    this._container = container;
                    this.update();
                    return container;
                },
                update: function () {
                    const center = map.getCenter();
                    const zoom = map.getZoom();

                    const latRad = center.lat * Math.PI / 180;
                    const n = Math.pow(2, zoom);
                    const tileX = Math.floor((center.lng + 180) / 360 * n);
                    const tileY = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);

                    const tileUrl = `/tile/${zoom}/${tileX}/${tileY}.png`;

                    this._container.innerHTML = `
                        <b>Center Tile Debug Info</b><br>
                        Zoom: ${zoom} | X: ${tileX} | Y: ${tileY}<br>
                        <a href="${tileUrl}" target="_blank" rel="noopener">Open Current Center Tile PNG ↗</a>
                    `;
                }
            });

            const debugControl = new DebugControl();
            map.addControl(debugControl);

            map.on('moveend', function () {
                debugControl.update();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/tile/{z}/{x}/{y}.png")
async def get_png_tile(z: int, x: int, y: int):
    loop = asyncio.get_running_loop()

    cached_file_path = await loop.run_in_executor(None, get_tile_file_path, z, x, y)
    if cached_file_path:
        return FileResponse(path=cached_file_path, media_type="image/png", headers=CACHE_HEADERS)

    pbf_tiles_data = []
    used_pmtiles = set()

    # z >= 18 のときのみ 3x3 (周囲1マス) を取得。z < 18 は対象の 1 マスのみ
    offsets = [-1, 0, 1] if z >= 18 else [0]

    for dy in offsets:
        for dx in offsets:
            curr_x = x + dx
            curr_y = y + dy
            pbf_list, actual_z, actual_x, actual_y, filenames = fetch_pbf_from_filepath(
                priority_pmtiles, normal_pmtiles, z, curr_x, curr_y
            )
            pbf_tiles_data.append((dx, dy, pbf_list, actual_z, actual_x, actual_y))
            used_pmtiles.update(filenames)

    if not any(item[2] for item in pbf_tiles_data):
        return Response(content=EMPTY_TILE_BYTES, media_type="image/png", headers=CACHE_HEADERS)

    png_bytes = await loop.run_in_executor(
        executor, render_3x3_tile_skia, z, x, y, pbf_tiles_data
    )

    if png_bytes != EMPTY_TILE_BYTES:
        saved_path = await loop.run_in_executor(None, save_png_file, z, x, y, png_bytes)
        await loop.run_in_executor(None, register_tile_to_db, z, x, y, saved_path)
        return FileResponse(path=saved_path, media_type="image/png", headers=CACHE_HEADERS)

    return Response(content=png_bytes, media_type="image/png", headers=CACHE_HEADERS)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8990)