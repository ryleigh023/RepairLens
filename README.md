# RepairLens
By - Team Ome Eight Eight 

AI-powered Snapchat AR Lens for repair shops. Detects device damage, shows AR overlays, explains repairs, and estimates costs — in English, Hindi, and Tamil.

**Lens ID:** 919abe25-5bfe-4978-a611-161007944142
**Try it:** https://lens.snapchat.com/919abe25-5bfe-4978-a611-161007944142

## Features

- Detects cracked screens, battery swelling, charging port damage
- Pulsing AR highlight on damaged area
- 
- Floating repair explanation panel
- Cost breakdown (parts + labor)
- Multilingual support (EN / HI / TA)
- Repair animation walkthrough

## Tech Stack

**AR:** Lens Studio, SnapML · **CV:** YOLOv8, TensorFlow Lite, OpenCV · **ML:** LSTM, RNN, PyTorch · **Translation:** IndicTrans2, HuggingFace · **Backend:** Python, FastAPI · **Cloud:** AWS SageMaker, EC2, S3

## Run It

```bash
# Backend
cd backend
pip3 install -r requirements.txt
python3 -m uvicorn main:app --port 8000
# API docs → http://127.0.0.1:8000/docs

# ML Pipeline
cd ml-pipeline
pip3 install -r requirements.txt
python3 dataset_prep.py --synthetic
```

Lens Studio → Open project → Preview 

## Project Structure

```
lensss/
├── backend/        # FastAPI + AI models
├── ml-pipeline/    # YOLOv8 training + dataset prep
├── lens-studio/    # Scripts + SnapML config
└── README.md
```

## How It Works

Camera → YOLOv8 detection → Bounding box → AR highlight → Explanation panel → User taps (cost / language / animation)
