"""
MMRT Backend API
FastAPI server for sign language recognition with LLM translation.
Pipeline: Video → MMRT Model → Gloss Sequence → LLM Translation → Chinese Text
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
import json
import tempfile
import shutil
import time
import io
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mmrt_inference_model import MMRTInferenceModel
from modules.video_transforms import extract_video_frames
from modules.llm_postprocessor import GlossTranslator

app = FastAPI(
    title="MMRT Sign Language Recognition API",
    description="Multi-Modal Recognition Transformer for sign language recognition",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

# Use CPU for stability (MPS has max_pool3d issues)
device = torch.device('cpu')
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

mmrt_model = None
translator = None
gloss_dict = None
dataset_info = None


@app.on_event("startup")
async def startup_event():
    global mmrt_model, translator, gloss_dict, dataset_info

    print(f"Loading MMRT model on device: {device}")

    # Load MMRT model
    weights_path = BASE_DIR / 'checkpoints' / 'mmrt_best.pt'
    config_path = BASE_DIR / 'configs' / 'inference.yaml'
    dataset_path = BASE_DIR / 'configs' / 'dataset.yaml'

    if not weights_path.exists():
        print(f"WARNING: Model weights not found at {weights_path}")
        return

    try:
        mmrt_model, gloss_dict, dataset_info = MMRTInferenceModel.from_pretrained(
            str(weights_path), str(config_path), str(dataset_path)
        )
        mmrt_model = mmrt_model.to(device)
        mmrt_model.eval()
        print(f"MMRT model loaded successfully | Device: {device}")
    except Exception as e:
        print(f"Failed to load MMRT model: {e}")
        import traceback
        traceback.print_exc()

    # Load translator
    train_path = BASE_DIR / 'data' / 'csl_daily' / 'train_8_2.json'
    if train_path.exists():
        try:
            import os
            translator = GlossTranslator(
                str(train_path),
                api_key=os.environ.get('DEEPSEEK_API_KEY'),
            )
            print(f"Translator loaded ({len(translator.exact_map)} exact entries)")
        except Exception as e:
            print(f"Failed to load translator: {e}")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "MMRT Multi-Modal Recognition Transformer",
        "architecture": "ResNet18 + TemporalConv + BiLSTM + CTC + LLM",
        "model_loaded": mmrt_model is not None,
        "translator_loaded": translator is not None,
        "device": str(device),
        "gloss_vocab_size": len(gloss_dict) if gloss_dict else 0,
    }


@app.post("/api/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Upload sign language video → MMRT inference → LLM translation → Chinese text.

    Pipeline:
    1. Extract video frames (256x256 → 224x224 center crop)
    2. MMRT model inference (ResNet18 → TemporalConv → BiLSTM → CTC)
    3. Gloss sequence decoding
    4. LLM translation (3-tier: exact/fuzzy/LLM)
    """
    if mmrt_model is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")

    start = time.time()
    suffix = Path(file.filename).suffix or '.mp4'
    tmp = UPLOAD_DIR / f"upload_{int(time.time()*1000)}{suffix}"

    try:
        # Save uploaded video
        with open(tmp, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # Step 1: Extract video frames
        t1 = time.time()
        frames = extract_video_frames(tmp)
        frames_time = time.time() - t1

        # Step 2: MMRT inference
        t2 = time.time()
        video = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
        video_length = torch.LongTensor([frames.shape[0]]).to(device)

        with torch.no_grad():
            result = mmrt_model(video, video_length)

        # Get predictions
        recognized = result["recognized_sents"][0]  # BiLSTM path
        conv = result["conv_sents"][0]  # Conv path
        inference_time = time.time() - t2

        # Step 3: LLM translation
        t3 = time.time()
        if translator:
            translation = translator.translate_list(recognized)
            pred_text = translation['text']
            tier = translation['tier']
            confidence = translation['confidence']
        else:
            pred_text = ''.join(recognized)
            tier = 0
            confidence = 0.1
        translate_time = time.time() - t3

        elapsed = time.time() - start

        return {
            "success": True,
            "predicted_text": pred_text,
            "predicted_glosses": recognized,
            "conv_glosses": conv,
            "confidence": round(confidence, 4),
            "translation_tier": tier,
            "processing_time": round(elapsed, 3),
            "pipeline": {
                "frame_extraction": round(frames_time, 3),
                "mmrt_inference": round(inference_time, 3),
                "llm_translation": round(translate_time, 3),
            },
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    finally:
        if tmp.exists():
            tmp.unlink()


@app.get("/api/stats")
async def get_stats():
    train_count, test_count = 0, 0
    try:
        train_path = BASE_DIR / 'data' / 'csl_daily' / 'train_8_2.json'
        test_path = BASE_DIR / 'data' / 'csl_daily' / 'test_8_2.json'
        if train_path.exists():
            with open(train_path, encoding='utf-8') as f:
                train_count = len(json.load(f))
        if test_path.exists():
            with open(test_path, encoding='utf-8') as f:
                test_count = len(json.load(f))
    except Exception:
        pass

    return {
        "dataset": {
            "name": "CSL-Daily",
            "total_samples": train_count + test_count,
            "train_samples": train_count,
            "test_samples": test_count,
            "split_ratio": "8:2",
            "gloss_vocab_size": len(gloss_dict) if gloss_dict else 0,
        },
        "model": {
            "architecture": "MMRT Multi-Modal Recognition Transformer",
            "modules": [
                "Module 4: ResNet18 + GCN visual feature extraction",
                "Module 2: Multi-scale temporal convolution + BiLSTM",
                "Module 8: CTC beam search decoder",
                "LLM: DeepSeek 3-tier translation (exact/fuzzy/LLM)",
            ],
            "input": "RGB Video Frames (224x224)",
            "output": "Chinese Text",
            "loaded": mmrt_model is not None,
            "device": str(device),
        },
        "tech_highlights": [
            "ResNet18 + Spatial/Temporal Graph Convolution",
            "Multi-scale 1D temporal convolution",
            "Bidirectional LSTM sequence modeling",
            "CTC beam search decoding",
            "DeepSeek LLM 3-tier translation",
            "Multi-dialect TTS synthesis",
        ]
    }


@app.post("/api/tts")
async def text_to_speech(payload: dict):
    """
    Multi-dialect TTS using edge-tts.
    Supported dialects: mandarin, cantonese, sichuan, taiwan, northeast, shanghai
    """
    text = payload.get("text", "").strip()
    dialect = payload.get("dialect", "mandarin")
    if not text:
        raise HTTPException(400, "text is required")

    voice_map = {
        "mandarin":  "zh-CN-XiaoxiaoNeural",
        "taiwan":    "zh-TW-HsiaoChenNeural",
        "cantonese": "zh-HK-HiuMaanNeural",
        "sichuan":   "zh-CN-YunxiNeural",
        "northeast": "zh-CN-YunjianNeural",
        "shanghai":  "zh-CN-XiaoyiNeural",
    }
    voice = voice_map.get(dialect, "zh-CN-XiaoxiaoNeural")

    try:
        import edge_tts
        import asyncio

        communicate = edge_tts.Communicate(text, voice)
        audio_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buf.write(chunk["data"])
        audio_buf.seek(0)
        return StreamingResponse(audio_buf, media_type="audio/mpeg",
                                 headers={"Content-Disposition": "inline; filename=tts.mp3"})
    except ImportError:
        raise HTTPException(501, "edge-tts not installed. Run: pip install edge-tts")
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
