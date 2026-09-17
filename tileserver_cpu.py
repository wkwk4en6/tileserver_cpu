import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Query, Response
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import HTMLResponse, PlainTextResponse
import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader
import skia
import uvicorn

BASE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = BASE_DIR / "tile_cache"
CACHE_DIR.mkdir(exist_ok=True)

pmtiles_file_list: List[Tuple[str, Path]] = []
executor = ProcessPoolExecutor()

# キャッシュヘッダー設定（1日キャッシュ）
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


def create_empty_tile_png() -> bytes:
    surface = skia.Surface(256, 256)
    surface.getCanvas().clear(LAND_COLOR)
    image = surface.makeImageSnapshot()
    return image.encodeToData().bytes()


EMPTY_TILE_BYTES = create_empty_tile_png()

@asynccontextmanager
async def lifespan(app: FastAPI):
    for folder in ["world-pmtiles"]:
        dir_path = BASE_DIR / folder
        if dir_path.exists():
            for p in dir_path.rglob("*.pmtiles"):
                try:
                    with open(p, "rb") as f:
                        source = MmapSource(f)
                        reader = Reader(source)
                        header = reader.header()
                        pmtiles_file_list.append((p.stem, p))
                        print(
                            f"[Info] Loaded: {p.stem} (Zoom: {header['min_zoom']} - {header['max_zoom']})"
                        )
                except Exception as e:
                    print(f"[Error] Failed to load {p.name}: {e}")
    yield


app = FastAPI(lifespan=lifespan)


def fetch_pbf_from_filepath(
    file_list: List[Tuple[str, Path]], z: int, x: int, y: int
) -> Tuple[Optional[bytes], int, int, int]:
    max_tile = 1 << z
    if x < 0 or x >= max_tile or y < 0 or y >= max_tile:
        return None, z, x, y

    for dz in range(0, z + 1):
        curr_z = z - dz
        curr_x = x >> dz
        curr_y_xyz = y >> dz
        curr_y_tms = (1 << curr_z) - 1 - curr_y_xyz

        for name, file_path in file_list:
            try:
                with open(file_path, "rb") as f:
                    reader = Reader(MmapSource(f))
                    header = reader.header()
                    if not (header["min_zoom"] <= curr_z <= header["max_zoom"]):
                        continue

                    for check_y in [curr_y_xyz, curr_y_tms]:
                        try:
                            tile_data = reader.get(curr_z, curr_x, check_y)
                            if tile_data and len(tile_data) > 0:
                                if tile_data[:2] == b"\x1f\x8b":
                                    tile_data = gzip.decompress(tile_data)
                                return tile_data, curr_z, curr_x, check_y
                        except Exception:
                            continue
            except Exception:
                continue

    return None, z, x, y


def render_3x3_tile_skia(
    target_z: int, target_x: int, target_y: int, file_list: List[Tuple[str, Path]]
) -> bytes:
    canvas_size = 768
    tile_size = 256.0

    base_surface = skia.Surface(canvas_size, canvas_size)
    base_canvas = base_surface.getCanvas()
    base_canvas.clear(LAND_COLOR)

    # ズームレベル 18 の時のみテキスト用の描画環境とフォントを準備
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
                new_rect.left() - 4,
                new_rect.top() - 4,
                new_rect.width() + 8,
                new_rect.height() + 8
            )
            for box in placed_boxes:
                if skia.Rect.Intersects(padded_rect, box):
                    return True
            return False

    # ペイント設定
    paint_water = skia.Paint(Color=WATER_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)
    paint_green = skia.Paint(Color=GREEN_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)
    paint_land = skia.Paint(Color=LAND_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)

    paint_building_fill = skia.Paint(Color=BUILDING_COLOR, AntiAlias=True, Style=skia.Paint.kFill_Style)
    paint_building_stroke = skia.Paint(Color=BUILDING_STROKE, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.5)

    paint_expressway = skia.Paint(Color=skia.Color(255, 140, 0, 255), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0)
    paint_primary = skia.Paint(Color=skia.Color(255, 215, 0, 255), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.2)
    paint_secondary = skia.Paint(Color=skia.Color(250, 235, 150, 255), AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.8)
    paint_road = skia.Paint(Color=ROAD_COLOR, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.2)
    paint_line = skia.Paint(Color=LINE_COLOR, AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.8)

    def get_layer_priority(name: str) -> int:
        n = name.lower()
        if any(k in n for k in ["land", "landcover", "landuse", "park", "green", "leisure"]): return 10
        if any(k in n for k in ["water", "ocean", "river", "lake"]): return 20
        if "building" in n: return 40
        if any(k in n for k in ["road", "street", "transportation", "highway"]): return 50
        if any(k in n for k in ["place", "poi", "address", "label", "location"]): return 60
        return 25

    tiles_to_process = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            curr_x = target_x + dx
            curr_y = target_y + dy
            pbf_data, actual_z, actual_x, actual_y = fetch_pbf_from_filepath(file_list, target_z, curr_x, curr_y)
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

        # テキストを描画しないズームレベルでは、ラベル系レイヤーの処理をスキップして高速化
        is_label_layer = any(k in layer_lower for k in ["place", "poi", "address", "label", "location"])
        if not render_text and is_label_layer:
            continue

        is_water_layer = any(k in layer_lower for k in ["water", "ocean", "river", "lake"])
        is_green_layer = any(k in layer_lower for k in ["park", "forest", "green", "leisure"])
        is_building_layer = "building" in layer_lower
        is_road_layer = any(k in layer_lower for k in ["road", "street", "transportation", "highway"])

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

            offset_x = (dx + 1) * tile_size
            offset_y = (dy + 1) * tile_size

            for feature in layer.get("features", []):
                geom_type = feature.get("geometry", {}).get("type")
                coords = feature.get("geometry", {}).get("coordinates", [])
                properties = feature.get("properties", {})

                # ポリゴン描画
                if geom_type in ["Polygon", "MultiPolygon"]:
                    rings = coords if geom_type == "Polygon" else [ring for poly in coords for ring in poly]
                    prop_values = {str(v).lower() for v in properties.values()}
                    is_green = is_green_layer or bool(prop_values & GREEN_KEYWORDS)
                    is_water = is_water_layer or "water" in prop_values

                    fill_paint = paint_land
                    if is_building_layer: fill_paint = paint_building_fill
                    elif is_water: fill_paint = paint_water
                    elif is_green: fill_paint = paint_green

                    path = skia.Path()
                    for ring in rings:
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

                # ライン描画
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

                # テキスト（ポイント）描画（Zoom Level 18 のみ）
                elif render_text and geom_type in ["Point", "MultiPoint"]:
                    def to_str(val):
                        if isinstance(val, bytes):
                            try: return val.decode("utf-8")
                            except UnicodeDecodeError: return ""
                        return str(val) if val is not None else ""

                    # 判定用セットの定義
                    city_types = {
                        "country", "state", "region", "province",
                        "city", "municipality", "town", "village",
                        "district", "county"
                    }
                    detail_types = {
                        "suburb", "neighbourhood", "quarter", "block",
                        "street", "house", "building", "address", "poi"
                    }

                    # 種別の取得 (place, class, subclass, type などを確認)
                    place_type = to_str(
                        properties.get("place") or 
                        properties.get("class") or 
                        properties.get("subclass") or 
                        properties.get("type")
                    ).lower()

                    # 名称の取得
                    local_name = properties.get("name") or properties.get("name:ja") or properties.get("name_ja")
                    en_name = properties.get("name:en") or properties.get("name_en")
                    local_str = to_str(local_name)
                    en_str = to_str(en_name)

                    # 住所・街区情報の取得
                    housenumber = to_str(properties.get("addr:housenumber") or properties.get("housenumber"))
                    street = to_str(properties.get("addr:street") or properties.get("street") or properties.get("block_number"))

                    # 表示用の名称（建物名・POI名など）を整形
                    if local_str and en_str and local_str.lower() != en_str.lower():
                        name_label = f"{local_str} ({en_str})"
                    else:
                        name_label = local_str or en_str

                    # 住所表記を組み立て
                    address_parts = [p for p in [street, housenumber] if p]
                    address_label = " ".join(address_parts)

                    # 表示ラベルの確定ロジック
                    label_str = ""

                    # 1. detail_types（詳細な街区・住所・建物・POI）の場合
                    if place_type in detail_types or not place_type:
                        if name_label and address_label:
                            label_str = f"{name_label} ({address_label})"
                        elif name_label:
                            label_str = name_label
                        elif address_label:
                            label_str = address_label

                    # 2. city_types（市区町村・都道府県など広域）の場合
                    elif place_type in city_types:
                        # ズーム18以上では広域名称を表示したい場合のみセット（不要なら continue でスキップ可）
                        label_str = name_label

                    # 3. その他のカテゴリ
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

    # ズームレベル 18 の時のみテキストをベース背景に合成
    if render_text:
        label_image = label_surface.makeImageSnapshot()
        base_canvas.drawImage(label_image, 0, 0)

    full_image = base_surface.makeImageSnapshot()

    # 中央 (256, 256) から 256x256px 切り出し
    final_surface = skia.Surface(256, 256)
    final_canvas = final_surface.getCanvas()

    crop_src = skia.Rect.MakeXYWH(256, 256, 256, 256)
    crop_dst = skia.Rect.MakeWH(256, 256)
    final_canvas.drawImageRect(full_image, crop_src, crop_dst)

    image = final_surface.makeImageSnapshot()
    return image.encodeToData().bytes()


app = FastAPI(lifespan=lifespan)

# ローカルの assets ディレクトリを /assets エンドポイントとして配信
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
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            const map = L.map('map').setView([34.6873, 135.5262], 4);

            L.tileLayer('/tile/{z}/{x}/{y}.png', {
                minZoom: 0,
                maxZoom: 18,
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/tile/{z}/{x}/{y}.png")
async def get_png_tile(z: int, x: int, y: int):
    cache_path = CACHE_DIR / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Response(
            content=cache_path.read_bytes(),
            media_type="image/png",
            headers=CACHE_HEADERS,
        )

    loop = asyncio.get_running_loop()
    png_bytes = await loop.run_in_executor(
        executor,
        render_3x3_tile_skia,
        z,
        x,
        y,
        pmtiles_file_list,
    )

    try:
        cache_path.write_bytes(png_bytes)
    except Exception as e:
        print(f"[Cache Write Error] {e}")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers=CACHE_HEADERS,
    )

if __name__ == "__main__":
    try:
        uvicorn.run("tileserver_cpu:app", host="127.0.0.1", port=8990, reload=False)
    except KeyboardInterrupt:
        print("\n[Info] Server stopped by user.")
        sys.exit(0)