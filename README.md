# SegImBud

SegImBud is a web app for satellite image segmentation and change analysis.

Users can upload one image for segmentation, compare two scenes for change detection, fetch recent imagery from coordinates, process multiple images in batch, and review or correct masks before exporting reports.

## Run The App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
python run.py
```

3. Open:

- `http://127.0.0.1:8000/scene`
- `http://127.0.0.1:8000/coordinate`
- `http://127.0.0.1:8000/batch`
- `http://127.0.0.1:8000/review`
- `http://127.0.0.1:8000/change`
- `http://127.0.0.1:8000/docs`

For development with reload:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## What It Can Do

- Segment a single satellite image
- Fetch recent daily satellite imagery from coordinates
- Show original, mask, overlay, confidence, and uncertainty views
- Run tiled inference for large scenes
- Process multiple scenes in one batch
- Review and manually correct predicted masks
- Calculate per-class pixel counts and percentages
- Estimate area when resolution is provided
- Compare two scenes and detect land-cover changes
- Export PNG, CSV, and PDF outputs

## Main Routes

- `/scene` for single-image analysis
- `/coordinate` for coordinate-based imagery fetch and segmentation
- `/batch` for multi-image processing
- `/review` for QA and manual mask correction
- `/change` for change detection
- `/docs` for API docs
- `/` redirects to `/scene`

## Project Structure

```text
app/
  main.py
  model_arch.py
  processor.py
  reporting.py
  schemas.py
  static/
  templates/
docs/
  frontend-architecture.md
run.py
weights/
```

## Model Weights

Default checkpoint path:

- `weights/best_model_loveda_EffKANSeg_v9_1024.pth`

If the file is missing, the app can download it automatically from Hugging Face.

Useful environment variables:

- `SEGIMBUD_WEIGHTS_DIR`
- `SEGIMBUD_WEIGHTS_FILE`
- `SEGIMBUD_HF_REPO_ID`
- `SEGIMBUD_AUTO_DOWNLOAD_WEIGHTS`

Disable auto-download with:

```bash
SEGIMBUD_AUTO_DOWNLOAD_WEIGHTS=false
```

## API Endpoints

- `GET /api/health`
- `POST /api/predict`
- `POST /api/coordinate-predict`
- `POST /api/batch-predict`
- `POST /api/change-detection`
- `POST /api/reports/pdf`

## Notes

- The app runs on CUDA when available, otherwise CPU.
- The main app lives on port `8000`.
- Coordinate fetch uses NASA GIBS daily imagery and needs an active internet connection.
