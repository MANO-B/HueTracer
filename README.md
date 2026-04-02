# HueTracer <img src="src/HueTracer.jpg" width=50>
## Cell-cell interaction prediction model for Visium HD spatial transcriptome platform
HueTracer is software for analyzing cell-cell interactions at single-cell resolution using Visium HD. It identifies cell positions using the Bin2Cell algorithm. It performs cell typing using label transfer from cell annotations in single-cell analysis or by combining Visium HD tissue images with gene expression profiles. It performs microenvironment clustering based on the expression patterns of cells surrounding each cell. It identifies cell-cell interactions using the NicheNet database.  
  
HueTracer is a software program created as a follow-up to [DeepCOLOR](https://github.com/kojikoji/deepcolor) software, which has some [programming error](https://github.com/kojikoji/deepcolor/issues/3).  


## Instalation
You can install HueTracer using pip command from your shell.
```shell
pip install huetracer
```

## Docker
```shell
# run at directory with Dockerfile
# run only once
# for PC with NVIDIA GPU
docker build -f Dockerfile -t huetracer-env .
# for PC without NVIDIA GPU or Mac
# Mac user: Docker environment cannot use MPU (apple silicon GPU).
#           Please run in an native python environment.
# docker build -f Dockerfile.cpu -t huetracer-env .

# In proxy environment, check proxy.server:port
# example
# docker build \
#   -f Dockerfile  # or Dockerfile.cpu
#   --build-arg http_proxy=http://gw.xxx.jp:8080 \
#   --build-arg https_proxy=http://gw.xxx.jp:8080 \
#   -t huetracer-env .

# run at upstream directory of 10x data files
# /app is the working directory (turorial directory, etc.)
# /data is set to your data directory
# CUBLAS_WORKSPACE_CONFIG is preset in Dockerfile for GPU image.
docker run -it \
  --gpus all \
  -p 8152:8152 \
  -v ${pwd}:/app \
  -v YourDataDirectory:/app/data \
  --shm-size=64g \
  --name huetracer \
  huetracer-env

# In proxy environment, check proxy.server:port
docker run -it \
  --gpus all \
  -p 8152:8152 \
  -v ${pwd}:/app \
  -v YourDataDirectory:/app/data \
  -e http_proxy=http://gw.xxx.jp:8080 \
  -e https_proxy=http://gw.xxx.jp:8080 \
  -e no_proxy="localhost,127.0.0.1,0.0.0.0" \
  --shm-size=64g \
  --name huetracer \
  huetracer-env
  
# run without GPU (add -e http~proxy=... in proxy environment)
docker run -it \
  -p 8152:8152 \
  -v ${pwd}:/app \
  -v YourDataDirectory:/app/data \
  -e no_proxy="localhost,127.0.0.1,0.0.0.0" \
  --shm-size=64g \
  --name huetracer \
  huetracer-cpu
```

## Usage
You need to prepare Visium HD spatial transcriptome data generated with SpaceRanger program by 10X. You can see the usage as follows.

#### Visium HD analysis with or without Chromium data
#####Tutorial with Colon adenocarcinoma sample obtained from 10X website (with version 0.0.15)
Download Chromium/Visium HD files.  
[Chromium single cell transctiptome aggregated files](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc), Feature barcode matrix (filtered)  
[Visium HD spatial transcriptome P2 CRC files](https://www.10xgenomics.com/jp/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc), Binned outputs (all bin levels), Microscope image (BTF)  
  
```
Example of data directory
.
├── P1_CRC
│   ├── Visium_HD_Human_Colon_Cancer_P1_tissue_image.btf
│   ├── binned_outputs
│   │   ├── square_002um
│   │   │   ├── Visium_HD_tissue_image_full_res.tiff
│   │   │   ├── filtered_feature_bc_matrix
│   │   │   │   ├── barcodes.tsv.gz
│   │   │   │   ├── features.tsv.gz
│   │   │   │   └── matrix.mtx.gz
│   │   │   ├── filtered_feature_bc_matrix.h5
│   │   │   ├── raw_feature_bc_matrix
│   │   │   │   ├── barcodes.tsv.gz
│   │   │   │   ├── features.tsv.gz
│   │   │   │   └── matrix.mtx.gz
│   │   │   ├── raw_feature_bc_matrix.h5
│   │   │   ├── raw_probe_bc_matrix.h5
│   │   │   └── spatial
│   │   │       ├── aligned_fiducials.jpg
│   │   │       ├── aligned_tissue_image.jpg
│   │   │       ├── cytassist_image.tiff
│   │   │       ├── detected_tissue_image.jpg
│   │   │       ├── scalefactors_json.json
│   │   │       ├── tissue_hires_image.png
│   │   │       ├── tissue_lowres_image.png
│   │   │       └── tissue_positions.parquet
│   │   └── square_008um
│   │       ├── filtered_feature_bc_matrix
│   │       │   ├── barcodes.tsv.gz
│   │       │   ├── features.tsv.gz
│   │       │   └── matrix.mtx.gz
│   │       ├── filtered_feature_bc_matrix.h5
│   │       ├── raw_feature_bc_matrix
│   │       │   ├── barcodes.tsv.gz
│   │       │   ├── features.tsv.gz
│   │       │   └── matrix.mtx.gz
│   │       ├── raw_feature_bc_matrix.h5
│   │       └── spatial
│   │           ├── aligned_fiducials.jpg
│   │           ├── aligned_tissue_image.jpg
│   │           ├── cytassist_image.tiff
│   │           ├── detected_tissue_image.jpg
│   │           ├── scalefactors_json.json
│   │           ├── tissue_hires_image.png
│   │           ├── tissue_lowres_image.png
│   │           └── tissue_positions.parquet
│   └── results (not mandatory)
│       └── (directory for result data)
├── P2_CRC
│   └── same above
├── P3_NAT
│   └── same above
├── P5_CRC
│   └── same above
├── P5_NAT
│   └── same above
└── SC_CRC
    └── filtered_feature_bc_matrix
        ├── barcodes.tsv.gz
        ├── features.tsv.gz
        └── matrix.mtx.gz

```
## Tutorials

### 1️⃣ Nucleus Segmentation

- [Nucleus segmentation tutorial](tutorial/nucleus_segmentation_tutorial_10x.ipynb)  
  As the BTF file is too large to handle with Bin2Cell, a cropped image is used in this tutorial.  
  This preprocessing step is not always necessary for other samples.

---

### 2️⃣ Cell Type Annotation

Cell type annotation differs depending on whether Chromium single-cell data is available.

#### 🔹 With Chromium Data

1. [Cell type annotation for single cell transcriptome tutorial](tutorial/single_cell_annotation_tutorial_10x.ipynb)  
2. [Cell type label transfer tutorial](tutorial/label_transfer_tutorial_10x.ipynb)  

#### 🔹 Without Chromium Data

- [Cell type annotation tutorial (without single-cell data)](tutorial/label_transfer_tutorial_without_single_cell_data.ipynb)

---

### 3️⃣ Microenvironment Analysis

- [Microenvironment prediction tutorial](tutorial/microenvironment_tutorial_10x.ipynb)

---

### 4️⃣ Cell–Cell Interaction Analysis

- [Cell-cell interaction tutorial](tutorial/cell_cell_interaction_tutorial_10x.ipynb)
