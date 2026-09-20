# Tile Server & Map Viewer

PMTiles データから地図タイルを生成・配信し、Web ブラウザ（Leaflet）上で表示するための軽量なタイルサーバおよびマップビューアです。

---

## 概要 (Overview)

- **タイル配信機能**: PMTiles データの読み込みおよびタイルのレンダリング・配信
- **キャッシュ機能**: タイルデータのSQLiteキャッシュ/ファイルキャッシュによる高速化
- **フロントエンド**: Leaflet を使用したインタラクティブなマップビューア
- **グリッド/デバッグ表示**: ズームレベル（z）やタイル座標（x, y）の表示・デバッグ対応

---

## 必要条件 (Requirements)

- Python 3.10+
- （その他依存ライブラリは `requirements.txt` を参照）

---

## セットアップ手順 (Setup)

### 1. リポジトリのクローン
```bash
git clone https://github.com/wkwk4en6/tileserver_cpu.git

cd tileserver_cpu
```

### 2. 仮想環境の作成とライブラリのインストール
本ツールはフォレンジック調査での利用を想定しているため、環境の独立性・再現性およびバイナリ依存関係の管理に優れた Miniconda / Anaconda の使用を推奨しています。

Conda を使用する場合（推奨）:
```bash
# 仮想環境の作成
conda create -n tile-server python=3.10 -y
conda activate tile-server

# 依存ライブラリのインストール
pip install -r requirements.txt
```
pyenv を使用する場合：
```Bash
python -m venv .venv
source .venv/bin/activate  # Windowsの場合は `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. ディレクトリの作成と PMTiles データの配置（事前準備）
本サーバの動作には PMTiles データが必要です。はじめに world-pmtiles/ フォルダを作成し、以下手順で取得したデータを配置してください。

#### world-pmtilesフォルダの作成

```Bash
mkdir world-pmtiles
```
#### 広域データの準備

pmtiles CLI ツールを使用し、Protomaps の公式リモートデータからズームレベル 0〜7 までの広域抽出を行います。   
Windws環境の場合は、コンパイル済みのバイナリを[protomaps/go-pmtiles](https://github.com/protomaps/go-pmtiles/releases)からDLしてください。

```Bash
pmtiles extract https://build.protomaps.com/2026xxxx.pmtiles world-pmtiles/planet_x0-z7.pmtiles --maxzoom=7
```
※ URL の日付部分(2026xxxx)は必要に応じて最新のビルドデータに変更してください。

#### 詳細マップデータのダウンロード (BBBike)

BBBike extracts([https://data.bbbike.org/osm/region/](https://data.bbbike.org/osm/region/)) にアクセスします。

対象エリアを選択し、Format（フォーマット）で PM Vector tiles Shortbread を選択してデータを抽出・ダウンロードします。

ダウンロードした .pmtiles ファイルを world-pmtiles/ フォルダ内に配置します。

## 実行方法 (Usage)
サーバを起動します。

```Bash
python tileserver_cpu.py
```
起動後、ブラウザで以下のURLにアクセスしてください：
http://localhost:8990

## リポジトリの構成 (Directory Structure)
```Plaintext
.
├── assets/                # フロントエンドアセット（Leaflet, CSS, Fonts など）
│   ├── leaflet.js
│   ├── leaflet.css
│   └── fonts/NotoSansCJK-Regular/NotoSansCJK-Regular.ttc             
├── world-pmtiles/         # PMTiles データ格納用（.gitignoreで除外）
│   ├── planet_x0-z7.pmtiles
│   └── *.pmtiles
├── cache_tiles/           # 生成されたタイルキャッシュ（.gitignoreで除外）
├── tileserver_cpu.py                # タイルサーバのメインプログラム
├── requirements.txt       # Python依存パッケージ一覧
└── README.md
```
## ライセンス (Licenses & Acknowledgments)
本プロジェクトおよび使用しているサードパーティ製アセットのライセンス情報は以下の通りです。

Leaflet: BSD 2-Clause License

Noto Sans CJK: SIL Open Font License 1.1

Map Data: © OpenStreetMap contributors

作者 / ライセンス (Author / License)
This project is licensed under the MIT License - see the LICENSE file for details.