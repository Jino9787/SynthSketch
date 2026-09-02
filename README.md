## Installation

```powershell
python -m pip install -r requirements.txt
```
### `main.py`

Generates and displays one random geometry.

```powershell
python main.py
```
### `dataset.py`

Generates geometries and saves as  `.npy` file.

```powershell
python dataset.py --samples 1000 --seed 100 --workers 4
```
Options:

```text
--samples       Number of geometries (default: 1)
--seed          Seed of the first geometry (default: 0)
--workers       Maximum number of parallel processes (default: 8)
--output-dir    Output directory (default: npy_outputs)
--folder-size   Files per subfolder (default: 10000)
--overwrite     Replace existing files
```

### `preview.py`

Generates 50 geometries and displays them.

```powershell
python preview.py --workers 8
```

### `view_tokens.py`

Prints `.npy` geometry file.

```powershell
python view_tokens.py npy_outputs\000001_010000\100.npy
```

### `token2geometry.py`

Reconstructs and displays a geometry from a `.npy` file.

```powershell
python token2geometry.py npy_outputs\000001_010000\100.npy
```

### `metrics.py`

Analyzes `.npy` files

```powershell
python metrics.py --input-dir npy_outputs --output-dir complexity_outputs
```

## Token Format

```text
SOL                  Start of loop
L, x, y              Line endpoint
A, mid_x, mid_y, end_x, end_y      Arc midpoint and endpoint
S, x1, y1, ...       B-spline control points after the first point
C, center_x, center_y, radius  Circle
EOS                  End of loop
```
