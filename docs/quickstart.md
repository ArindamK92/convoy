# Quickstart

## Environment
```bash
source ~/myenv/bin/activate
cd ~/CONVOY
```

## Install Docs Tooling
```bash
pip install -r docs/requirements.txt
```

## Serve Docs Locally
```bash
mkdocs serve
```

Open: `http://127.0.0.1:8000`

## Build Static Docs
```bash
mkdocs build
```

Output: `site/`

## Project Run Quickstart
Use the combined runner:

```bash
python3 convoy_main.py \
  --combined-details-csv data/combined_data_jd200_1.csv \
  --combined-dist-matrix-csv data/distance_matrix_jd200_1.csv \
  --combined-time-matrix-csv data/time_matrix_jd200_1.csv \
  --customer-num 10 \
  --charging-stations-num 3 \
  --ev-num 2 \
  --opt-rl-extra "--print-solution --save-model"
```
