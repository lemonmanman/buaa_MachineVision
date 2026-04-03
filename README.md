# Basic Operations of Machine Vision
This repository is used for the Machine Vision course of BUAA.

## Set up your environment
Make sure you already download the environment set-up tool:
```bash
# create your environment
conda create -n vision python=3.12
# activet your environment
conda activate vision
# download opencv
pip install opencv-python numpy

# check
conda env list
```

## Operations
### Color conversion
Run the script:
```bash
conda activate vision
python color_convert.py
```
### Geometric transformation
Run the script:
```bash
conda activate vision
python geo_transform.py
```

### Image intensification
Run the script:
```bash
conda activate vision
python img_intensificate.py
```

### Edge features extraction
Run the script:
```bash
conda activate vision
python edge_extract.py
```

### Morphological operation
Run the script:
```bash
conda activate vision
python morphological_operate.py
```