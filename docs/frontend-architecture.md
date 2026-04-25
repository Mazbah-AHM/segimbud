# Frontend Architecture

SegImBud uses FastAPI templates with separate CSS and JavaScript files.

This keeps the UI easier to maintain and keeps Python focused on the backend logic.

## Pages

- `scene.html` for single-image analysis
- `coordinate.html` for coordinate-based imagery fetch
- `batch.html` for multi-scene processing
- `review.html` for QA and manual correction
- `change.html` for two-image comparison

## Frontend Files

- `app/static/css/base.css` for shared styles
- `app/static/css/pages/` for page layouts
- `app/static/js/core.js` for shared browser helpers
- `app/static/js/pages/` for page logic

## Routes

- `/` redirects to `/scene`
- `/scene` is the main analysis page
- `/coordinate` handles coordinate-driven imagery analysis
- `/batch` handles batch processing
- `/review` handles QA and correction work
- `/change` is the change-detection page
- `/api/*` contains backend endpoints

## Local Run

- `python run.py` for the normal local run
- `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` for development
