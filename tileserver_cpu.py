import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import gzip
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Query, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader
from pydantic import BaseModel
import skia
import uvicorn
import sys
import arabic_reshaper
from bidi.algorithm import get_display

# カレントディレクトリ（スクリプトのある場所）を検索パスに追加
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DB_PATH = BASE_DIR / "tile_cache.db"
TILES_DIR = BASE_DIR / "cache_tiles"

# --- 設定値 ---
MAX_WAIT_TIME = 1.5  # 秒（セマフォ待機がこれを超えた古いリクエストは破棄）
MAX_WORKERS = os.cpu_count() or 4  # スレッド数

# --- インメモリキャッシュ ---
memory_cache: Dict[str, bytes] = {}

# --- PMTiles のメタデータ構造 ---
class LoadedPMTiles:
    def __init__(self, name: str, path: Path, file_obj, reader: Reader, header: dict):
        self.name = name
        self.path = path
        self.file_obj = file_obj
        self.reader = reader
        self.min_zoom = header.get("min_zoom", 0)
        self.max_zoom = header.get("max_zoom", 30)

        self.min_lon = header.get("min_lon_e7", -1800000000) / 1e7
        self.min_lat = header.get("min_lat_e7", -90000000) / 1e7
        self.max_lon = header.get("max_lon_e7", 1800000000) / 1e7
        self.max_lat = header.get("max_lat_e7", 90000000) / 1e7

    def intersects_tile(self, z: int, x: int, y: int) -> bool:
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

        margin = 1e-3
        if (tile_max_lon + margin) < self.min_lon or (tile_min_lon - margin) > self.max_lon:
            return False
        if (tile_max_lat + margin) < self.min_lat or (tile_min_lat - margin) > self.max_lat:
            return False

        return True


priority_pmtiles: List[LoadedPMTiles] = []
normal_pmtiles: List[LoadedPMTiles] = []

executor: Optional[ThreadPoolExecutor] = None
render_semaphore: Optional[asyncio.Semaphore] = None

CACHE_HEADERS = {"Cache-Control": "public, max-age=86400"}

# スタイル・色定数
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
    data = image.encodeToData(skia.EncodedImageFormat.kPNG, 100)
    if data is None:
        data = image.encodeToData()
    return data.bytes() if data is not None else b""


EMPTY_TILE_BYTES = create_empty_tile_png()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor, render_semaphore
    init_db()
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    render_semaphore = asyncio.Semaphore(MAX_WORKERS)

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
                    print(f"[Loaded PMTiles] {p.name} (min_z: {item.min_zoom}, max_z: {item.max_zoom})")
                except Exception as e:
                    print(f"[Error] Failed to load {p.name}: {e}")

    yield

    if executor:
        executor.shutdown()

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
) -> Tuple[List[bytes], int, int, int, List[str], List[str], Dict[str, Dict[str, int]]]:
    max_tile = 1 << z
    if x < 0 or x >= max_tile or y < 0 or y >= max_tile:
        return [], z, x, y, [], [], {}

    pbf_list = []
    used_filenames = []
    intersected_filenames = []
    used_layer_details: Dict[str, Dict[str, int]] = {}

    for pmtile in priority_list + normal_list:
        if pmtile.intersects_tile(z, x, y):
            intersected_filenames.append(pmtile.path.name)

    for dz in range(0, z + 1):
        curr_z = z - dz
        curr_x = x >> dz
        curr_y_xyz = y >> dz

        candidate_priority = [p for p in priority_list if p.intersects_tile(curr_z, curr_x, curr_y_xyz)]
        candidate_normal = [p for p in normal_list if p.intersects_tile(curr_z, curr_x, curr_y_xyz)]

        all_candidates = candidate_priority + candidate_normal
        if not all_candidates:
            continue

        found_in_this_zoom = False
        zoom_layer_details: Dict[str, Dict[str, int]] = {}

        if curr_z <= 8:
            all_candidates = all_candidates[:1]

        for pmtile in all_candidates:
            check_targets = [curr_y_xyz]

            for check_y in check_targets:
                try:
                    tile_data = pmtile.reader.get(curr_z, curr_x, check_y)
                    if tile_data and len(tile_data) > 0:
                        if tile_data[:2] == b"\x1f\x8b":
                            tile_data = gzip.decompress(tile_data)
                        
                        try:
                            decoded = mapbox_vector_tile.decode(tile_data, default_options={"y_coord_down": True})
                            
                            layer_features_count = {}
                            valid_layers_count = 0

                            for lname, ldata in decoded.items():
                                f_list = ldata.get("features", [])
                                f_count = len(f_list)
                                if f_count > 0:
                                    layer_features_count[lname] = f_count
                                    valid_layers_count += 1

                            if valid_layers_count < 4:
                                continue

                        except Exception:
                            continue

                        pbf_list.append(tile_data)
                        if pmtile.path.name not in used_filenames:
                            used_filenames.append(pmtile.path.name)
                        
                        if pmtile.path.name not in zoom_layer_details:
                            zoom_layer_details[pmtile.path.name] = {}
                        for lname, count in layer_features_count.items():
                            zoom_layer_details[pmtile.path.name][lname] = zoom_layer_details[pmtile.path.name].get(lname, 0) + count

                        found_in_this_zoom = True
                        
                        if curr_z <= 8:
                            break
                except Exception:
                    continue
            
            if curr_z <= 8 and found_in_this_zoom:
                break

        if found_in_this_zoom:
            return pbf_list, curr_z, curr_x, curr_y_xyz, used_filenames, intersected_filenames, zoom_layer_details

    return [], z, x, y, [], [], {}


def render_3x3_tile_skia(
    target_z: int,
    target_x: int,
    target_y: int,
    pbf_tiles_data: List[Tuple[int, int, Optional[bytes], int, int, int]]
) -> bytes:
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

        # フォント読み込みと FontMgr / Typeface の設定
        font_mgr = skia.FontMgr.RefDefault()
        
        # --- 修正箇所 1: フォントを個別に読み込んで保持する ---
        cjk_font_path = BASE_DIR / "assets" / "fonts" / "NotoSansCJK-Regular" / "NotoSansCJK-Regular.ttc"
        arabic_font_path = BASE_DIR / "assets" / "fonts" / "Noto_Sans_Arabic" / "NotoSansArabic-Regular.ttf"

        # デフォルト（CJK/英語用）フォント
        if cjk_font_path.exists():
            tf_cjk = skia.Typeface.MakeFromFile(str(cjk_font_path))
        else:
            tf_cjk = skia.Typeface.MakeFromName("sans-serif", skia.FontStyle.Normal())
        font_cjk = skia.Font(tf_cjk, 11)

        # アラビア文字用フォント
        if arabic_font_path.exists():
            tf_arabic = skia.Typeface.MakeFromFile(str(arabic_font_path))
            font_arabic = skia.Font(tf_arabic, 11)
        else:
            font_arabic = font_cjk

        # アラビア文字が含まれているか判定するヘルパー関数
        def is_arabic_text(text: str) -> bool:
            return any('\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\u08A0' <= char <= '\u08FF' for char in text)

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
                    decoded = mapbox_vector_tile.decode(pbf_data, default_options={"y_coord_down": True})
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

                    # アラビア文字が含まれるか判定するヘルパー
                    def is_arabic_char(char: str) -> bool:
                        code = ord(char)
                        return (
                            0x0600 <= code <= 0x06FF or
                            0x0750 <= code <= 0x077F or
                            0x08A0 <= code <= 0x08FF or
                            0xFB50 <= code <= 0xFDFF or
                            0xFE70 <= code <= 0xFEFF
                        )

                    # テキスト整形・アラビア語変換用ヘルパー関数
                    def process_text_segment(text: str) -> str:
                        if not text:
                            return ""
                        if any(is_arabic_char(c) for c in text):
                            try:
                                reshaped = arabic_reshaper.reshape(text)
                                return get_display(reshaped)
                            except Exception:
                                pass
                        return text

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
                    local_raw = to_str(local_name)
                    en_str = to_str(en_name)

                    # 各テキスト要素ごとに個別にアラビア語判定と整形を行う
                    local_str = process_text_segment(local_raw)

                    housenumber = to_str(properties.get("addr:housenumber") or properties.get("housenumber"))
                    street = to_str(properties.get("addr:street") or properties.get("street") or properties.get("block_number"))

                    if local_str and en_str and local_raw.lower() != en_str.lower():
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

                    # 文字ごとに適切なフォント（Arabic or CJK/Default）を分割して描画するヘルパー
                    def draw_mixed_text(canvas, text, x, y, paint_halo, paint):
                        current_x = x
                        
                        # 衝突判定用に文字ごとの幅を計算して全体サイズを取得
                        total_width = 0.0
                        for char in text:
                            f = font_arabic if is_arabic_char(char) else font_cjk
                            total_width += f.measureText(char)

                        bounds = skia.Rect.MakeXYWH(x, y - 11, total_width, 11)
                        if is_colliding(bounds):
                            return False

                        # 文字単位で適合するフォントを判定して描画
                        for char in text:
                            # 記号や数字でアラビア語文字列に挟まれている場合などの判定補正
                            f = font_arabic if is_arabic_char(char) else font_cjk
                            
                            # 描画
                            canvas.drawString(char, current_x, y, f, paint_halo)
                            canvas.drawString(char, current_x, y, f, paint)
                            
                            # X座標を文字幅分進める
                            current_x += f.measureText(char)

                        placed_boxes.append(bounds)
                        return True

                    points = [coords] if geom_type == "Point" else coords
                    for pt in points:
                        if len(pt) < 2: continue
                        px = offset_x + (pt[0] - min_x) * scale
                        py = offset_y + (pt[1] - min_y) * scale

                        if 0 <= px <= canvas_size and 0 <= py <= canvas_size:
                            draw_mixed_text(label_canvas, label_str, px, py, paint_text_halo, paint_text)

    if render_text:
        label_image = label_surface.makeImageSnapshot()
        base_canvas.drawImage(label_image, 0, 0)

    full_image = base_surface.makeImageSnapshot()

    if is_multi_tile:
        final_surface = skia.Surface(256, 256)
        final_canvas = final_surface.getCanvas()
        crop_src = skia.Rect.MakeXYWH(256, 256, 256, 256)
        crop_dst = skia.Rect.MakeWH(256, 256)
        final_canvas.drawImageRect(full_image, crop_src, crop_dst)
        image = final_surface.makeImageSnapshot()
    else:
        image = full_image

    data = image.encodeToData(skia.EncodedImageFormat.kPNG, 100)
    if data is None:
        data = image.encodeToData()
        
    return data.bytes() if data is not None else EMPTY_TILE_BYTES


def generate_single_tile(z: int, x: int, y: int) -> bytes:
    """単一のタイル画像を直接生成して保存する内部ヘルパー関数"""
    cache_key = f"{z}/{x}/{y}"

    pbf_tiles_data = []
    used_pmtiles = set()
    offsets = [-1, 0, 1] if z >= 18 else [0]

    for dy in offsets:
        for dx in offsets:
            curr_x = x + dx
            curr_y = y + dy
            pbf_list, actual_z, actual_x, actual_y, filenames, _, _ = fetch_pbf_from_filepath(
                priority_pmtiles, normal_pmtiles, z, curr_x, curr_y
            )
            pbf_tiles_data.append((dx, dy, pbf_list, actual_z, actual_x, actual_y))
            used_pmtiles.update(filenames)

    if not any(item[2] for item in pbf_tiles_data):
        memory_cache[cache_key] = EMPTY_TILE_BYTES
        return EMPTY_TILE_BYTES

    png_bytes = render_3x3_tile_skia(z, x, y, pbf_tiles_data)

    if png_bytes != EMPTY_TILE_BYTES:
        memory_cache[cache_key] = png_bytes
        saved_path = save_png_file(z, x, y, png_bytes)
        register_tile_to_db(z, x, y, saved_path)

    return png_bytes


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
                background: rgba(255, 255, 255, 0.95);
                padding: 8px 12px;
                font-family: monospace;
                font-size: 11px;
                border-radius: 4px;
                box-shadow: 0 0 5px rgba(0,0,0,0.3);
                line-height: 1.4;
                max-width: 360px;
                max-height: 80vh;
                overflow-y: auto;
            }
            .debug-info-control a { color: #0066cc; text-decoration: underline; font-weight: bold; }
            .layer-details {
                margin-left: 10px;
                font-size: 10px;
                color: #444;
            }
            .preload-box {
                margin-top: 8px;
                padding-top: 6px;
                border-top: 1px dashed #ccc;
            }
            .preload-box select, .preload-box button {
                font-size: 11px;
                padding: 2px 4px;
            }
            .preload-status {
                margin-top: 4px;
                color: #d9534f;
                font-weight: bold;
            }
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
                tile.style.border = '1px solid rgba(255, 0, 0, 0.4)';
                tile.style.fontFamily = 'monospace';
                tile.style.fontSize = '9px';
                tile.style.color = 'red';
                tile.style.padding = '2px';
                tile.style.pointerEvents = 'none';
                tile.style.overflow = 'hidden';

                fetch(`/tile/debug/${coords.z}/${coords.x}/${coords.y}`)
                    .then(res => res.json())
                    .then(data => {
                        let detailsHtml = '';
                        if (data.layer_details) {
                            for (const [fname, layers] of Object.entries(data.layer_details)) {
                                detailsHtml += `&nbsp;&nbsp;<b>${fname}:</b><br>`;
                                for (const [lname, count] of Object.entries(layers)) {
                                    detailsHtml += `&nbsp;&nbsp;&nbsp;&nbsp;- ${lname}: ${count} feats<br>`;
                                }
                            }
                        }

                        tile.innerHTML = `<b>z:${coords.z} x:${coords.x} y:${coords.y}</b><br>` +
                                         `<span style="color:#008000">Intersected:</span> ${data.intersected.join(', ') || 'None'}<br>` +
                                         `<span style="color:#0000ff">Render Source:</span><br>${detailsHtml || 'None'}`;
                    }).catch(e => {});

                return tile;
            };

            const overlayMaps = { "Tile Grid & Debug Display": tileGridLayer };
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

                    fetch(`/tile/debug/${zoom}/${tileX}/${tileY}`)
                        .then(res => res.json())
                        .then(data => {
                            let detailsHtml = '';
                            if (data.layer_details && Object.keys(data.layer_details).length > 0) {
                                for (const [fname, layers] of Object.entries(data.layer_details)) {
                                    detailsHtml += `<div style="margin-top:2px;"><b>📁 ${fname}</b></div>`;
                                    for (const [lname, count] of Object.entries(layers)) {
                                        detailsHtml += `<div class="layer-details">▫️ ${lname}: <b>${count}</b> feats</div>`;
                                    }
                                }
                            } else {
                                detailsHtml = ' None';
                            }

                            let selectOptions = '';
                            for (let z = zoom; z <= 18; z++) {
                                selectOptions += `<option value="${z}" ${z === Math.min(zoom + 2, 18) ? 'selected' : ''}>Zoom ${z}</option>`;
                            }

                            this._container.innerHTML = `
                                <b>Center Tile Info</b><br>
                                Zoom: ${zoom} | X: ${tileX} | Y: ${tileY}<br>
                                Bounds: [${data.lat_min?.toFixed(3)}, ${data.lon_min?.toFixed(3)}] ~ [${data.lat_max?.toFixed(3)}, ${data.lon_max?.toFixed(3)}]<br>
                                <span style="color:#008000"><b>Intersected:</b></span> ${data.intersected.join(', ') || 'None'}<br>
                                <span style="color:#0000ff"><b>Render Source & Vector Content:</b></span><br>${detailsHtml}<br>
                                <div style="margin-top:4px;"><a href="${tileUrl}" target="_blank" rel="noopener">Open Current Center Tile PNG ↗</a></div>
                                
                                <div class="preload-box">
                                    <b>⚡Pre-generate tile images</b><br>
                                    Target: <select id="target-zoom-select">${selectOptions}</select>
                                    <button onclick="startTilePreload(${zoom})">Start Pre-generation</button>
                                    <div id="preload-status" class="preload-status"></div>
                                </div>`;
                        });
                }
            });

            const debugControl = new DebugControl();

            tileGridLayer.on('add', function () {
                map.addControl(debugControl);
            });

            tileGridLayer.on('remove', function () {
                map.removeControl(debugControl);
            });

            map.on('moveend', function () {
                if (map.hasLayer(debugControl)) {
                    debugControl.update();
                }
            });

            // 緯度経度からタイル座標(X, Y)を取得するヘルパー関数
            function latLonToTile(lat, lon, zoom) {
                const n = Math.pow(2, zoom);
                const latRad = lat * Math.PI / 180;
                const xtile = Math.floor((lon + 180) / 360 * n);
                const ytile = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
                return {
                    x: Math.max(0, Math.min(xtile, n - 1)),
                    y: Math.max(0, Math.min(ytile, n - 1))
                };
            }

            async function startTilePreload(currentZoom) {
                const targetZoom = parseInt(document.getElementById('target-zoom-select').value, 10);
                const statusDiv = document.getElementById('preload-status');
                const bounds = map.getBounds();

                // 1. 総枚数を計算
                let totalTiles = 0;
                for (let z = currentZoom; z <= targetZoom; z++) {
                    const sw = latLonToTile(bounds.getSouth(), bounds.getWest(), z);
                    const ne = latLonToTile(bounds.getNorth(), bounds.getEast(), z);
                    const xCount = Math.abs(sw.x - ne.x) + 1;
                    const yCount = Math.abs(sw.y - ne.y) + 1;
                    totalTiles += xCount * yCount;
                }

                // 2. 残り予想時間（秒）を計算
                const estimatedMsPerTile = 30; 
                let remainingSec = Math.ceil((totalTiles * estimatedMsPerTile) / 1000);

                const requestData = {
                    min_lat: bounds.getSouth(),
                    max_lat: bounds.getNorth(),
                    min_lon: bounds.getWest(),
                    max_lon: bounds.getEast(),
                    start_zoom: currentZoom,
                    target_zoom: targetZoom
                };

                let timerId = null;

                // 時間表示の更新処理（0秒になっても「処理中...」のまま待機する）
                const updateStatusText = (sec) => {
                    if (sec > 0) {
                        const m = Math.floor(sec / 60);
                        const s = sec % 60;
                        const timeStr = m > 0 ? `${m}m${s}s` : `${s}s`;
                        statusDiv.innerText = `processing... (total ${totalTiles} sheets / Est. left: ${timeStr})`;
                    } else {
                        statusDiv.innerText = `processing... (total ${totalTiles} sheets / Finalizing...)`;
                    }
                };

                // 初期表示
                updateStatusText(remainingSec);
                
                // 1秒ごとにカウントダウン（0秒以下になってもタイマーは停止し、Finalizing表示にする）
                timerId = setInterval(() => {
                    remainingSec--;
                    updateStatusText(remainingSec);
                    if (remainingSec <= 0 && timerId) {
                        clearInterval(timerId);
                    }
                }, 1000);

                const startTime = Date.now();

                try {
                    // 事前作成リクエストを発行（バックエンド側の処理完了を待つ）
                    const res = await fetch('/tile/preload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestData)
                    });
                    const data = await res.json();
                    
                    // サーバーからレスポンスが返ってきて初めて完了表示にする
                    if (timerId) clearInterval(timerId);
                    const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);
                    statusDiv.innerText = `Complete: ${data.generated_count} sheets created (skipped ${data.skipped_count} sheets / ${elapsedSec}s)`;
                } catch (e) {
                    if (timerId) clearInterval(timerId);
                    statusDiv.innerText = "An error occurred";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/tile/debug/{z}/{x}/{y}")
async def get_tile_debug(z: int, x: int, y: int):
    _, _, _, _, used_filenames, intersected_filenames, layer_details = fetch_pbf_from_filepath(
        priority_pmtiles, normal_pmtiles, z, x, y
    )
    
    n = 1 << z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    def tile2lat(y_val, z_val):
        n_val = math.pi - (2.0 * math.pi * y_val) / (1 << z_val)
        return math.degrees(math.atan(math.sinh(n_val)))
    lat_max = tile2lat(y, z)
    lat_min = tile2lat(y + 1, z)

    return {
        "intersected": intersected_filenames,
        "used": used_filenames,
        "layer_details": layer_details,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max
    }


class PreloadRequest(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    start_zoom: int
    target_zoom: int

def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """緯度経度からタイル座標 (x, y) を計算するヘルパー関数"""
    n = 1 << zoom
    lat_rad = math.radians(lat)
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (max(0, min(xtile, n - 1)), max(0, min(ytile, n - 1)))

class CheckCachedRequest(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    start_zoom: int
    target_zoom: int

@app.post("/tile/preload")
async def preload_tiles(req: PreloadRequest):
    """表示圏内のタイル画像を事前に一括生成・保存するAPI"""
    loop = asyncio.get_running_loop()

    tiles_to_generate = []

    for z in range(req.start_zoom, req.target_zoom + 1):
        x_min, y_max = latlon_to_tile(req.min_lat, req.min_lon, z)
        x_max, y_min = latlon_to_tile(req.max_lat, req.max_lon, z)

        x_start, x_end = min(x_min, x_max), max(x_min, x_max)
        y_start, y_end = min(y_min, y_max), max(y_min, y_max)

        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                tiles_to_generate.append((z, x, y))

    generated_count = 0
    skipped_count = 0
    tasks = []

    for z, x, y in tiles_to_generate:
        cache_key = f"{z}/{x}/{y}"
        if cache_key in memory_cache:
            skipped_count += 1
            continue

        cached_path = await loop.run_in_executor(None, get_tile_file_path, z, x, y)
        if cached_path:
            skipped_count += 1
            continue

        # スレッドプールで実行するタスクを登録
        tasks.append(loop.run_in_executor(executor, generate_single_tile, z, x, y))

    # すべてのタイルの事前生成が完了するまで待機
    if tasks:
        await asyncio.gather(*tasks)
        generated_count = len(tasks)

    return {
        "status": "success",
        "total_requested": len(tiles_to_generate),
        "generated_count": generated_count,
        "skipped_count": skipped_count
    }

@app.get("/tile/{z}/{x}/{y}.png")
async def get_png_tile(z: int, x: int, y: int):
    cache_key = f"{z}/{x}/{y}"
    start_time = time.time()
    loop = asyncio.get_running_loop()

    # インメモリキャッシュ（1次キャッシュ）
    if cache_key in memory_cache:
        return Response(content=memory_cache[cache_key], media_type="image/png", headers=CACHE_HEADERS)

    # ディスク/DBキャッシュ（2次キャッシュ）
    cached_file_path = await loop.run_in_executor(None, get_tile_file_path, z, x, y)
    if cached_file_path:
        try:
            with open(cached_file_path, "rb") as f:
                data = f.read()
                memory_cache[cache_key] = data
                return Response(content=data, media_type="image/png", headers=CACHE_HEADERS)
        except Exception:
            pass

    # セマフォ制御 ＆ タイムアウト廃棄
    async with render_semaphore:
        if time.time() - start_time > MAX_WAIT_TIME:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        if cache_key in memory_cache:
            return Response(content=memory_cache[cache_key], media_type="image/png", headers=CACHE_HEADERS)

        # PBFデータ読み込み
        pbf_tiles_data = []
        used_pmtiles = set()
        offsets = [-1, 0, 1] if z >= 18 else [0]

        for dy in offsets:
            for dx in offsets:
                curr_x = x + dx
                curr_y = y + dy
                pbf_list, actual_z, actual_x, actual_y, filenames, _, _ = fetch_pbf_from_filepath(
                    priority_pmtiles, normal_pmtiles, z, curr_x, curr_y
                )
                pbf_tiles_data.append((dx, dy, pbf_list, actual_z, actual_x, actual_y))
                used_pmtiles.update(filenames)

        if not any(item[2] for item in pbf_tiles_data):
            memory_cache[cache_key] = EMPTY_TILE_BYTES
            return Response(content=EMPTY_TILE_BYTES, media_type="image/png", headers=CACHE_HEADERS)

        # スレッドプール上での動的描画実行
        png_bytes = await loop.run_in_executor(
            executor, render_3x3_tile_skia, z, x, y, pbf_tiles_data
        )

        # キャッシュ登録 & ファイル保存
        if png_bytes != EMPTY_TILE_BYTES:
            memory_cache[cache_key] = png_bytes
            saved_path = await loop.run_in_executor(None, save_png_file, z, x, y, png_bytes)
            await loop.run_in_executor(None, register_tile_to_db, z, x, y, saved_path)

        return Response(content=png_bytes, media_type="image/png", headers=CACHE_HEADERS)


if __name__ == "__main__":
    filename = Path(__file__).stem

    # Windows環境の場合、SelectorEventLoopを使用する
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(f"{filename}:app", host="127.0.0.1", port=8990, workers=8)