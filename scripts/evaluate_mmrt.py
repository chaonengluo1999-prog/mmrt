"""
MMRT Evaluation Script
Evaluates the MMRT model on CSL-Daily test set (8:2 split, 454 samples).
Computes Gloss WER, BLEU scores, and sentence-level accuracy.

Usage:
    cd /Users/wujian1/Desktop/SignGraph/mmrt
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluate_mmrt.py
"""

import sys
import os
from dotenv import load_dotenv

# 加载项目根目录下 .env 文件
load_dotenv()

# 读取API密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY 未配置，请检查 .env 文件")

# Enable MPS fallback for ops not yet implemented on Apple Silicon
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import json
import glob
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# mmrt root must be first so mmrt/modules/ takes precedence over SignGraph/modules/
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
SIGNGRAPH_ROOT = ROOT.parent
sys.path.insert(0, str(SIGNGRAPH_ROOT))
sys.path.insert(0, str(ROOT))

import torch
import cv2

from models.mmrt_inference_model import MMRTInferenceModel
from modules.llm_postprocessor import GlossTranslator
from modules.video_transforms import Compose, CenterCrop, Resize, ToTensor



def compute_wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)

    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = d[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref)][len(hyp)]


def compute_bleu(ref_text, hyp_text, n=4):
    """Compute character-level BLEU-n score."""
    ref = list(ref_text.replace(' ', ''))
    hyp = list(hyp_text.replace(' ', ''))
    if not hyp:
        return 0.0
    import math
    scores = []
    for k in range(1, n + 1):
        ref_ngrams = defaultdict(int)
        for i in range(len(ref) - k + 1):
            ref_ngrams[tuple(ref[i:i+k])] += 1
        matches = 0
        for i in range(len(hyp) - k + 1):
            ng = tuple(hyp[i:i+k])
            if ref_ngrams[ng] > 0:
                matches += 1
                ref_ngrams[ng] -= 1
        denom = max(len(hyp) - k + 1, 1)
        scores.append(matches / denom if denom > 0 else 0.0)
    if min(scores) == 0:
        return 0.0
    log_avg = sum(math.log(s) for s in scores) / n
    bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return bp * math.exp(log_avg)


def load_frames(folder, target=256, crop=224):
    paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, (target, target),
                                       interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
        frames.append(img)
    if not frames:
        return None
    transform = Compose([CenterCrop(crop), Resize(1.0), ToTensor()])
    video, _ = transform(frames, None)
    return video.float() / 127.5 - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=str(ROOT / 'checkpoints/mmrt_best.pt'))
    parser.add_argument('--config', default=str(ROOT / 'configs/inference.yaml'))
    parser.add_argument('--dataset-config', default=str(ROOT / 'configs/dataset.yaml'))
    parser.add_argument('--dataset-root', default=os.path.expanduser('~/Desktop/CSL-Daily'))
    parser.add_argument('--train-json', default=str(ROOT / 'data/csl_daily/train_8_2.json'))
    parser.add_argument('--test-json', default=str(ROOT / 'data/csl_daily/test_8_2.json'))
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output', default=str(ROOT / 'data/eval_results.json'))
    parser.add_argument('--no-llm', action='store_true')
    args = parser.parse_args()

    # Default to CPU: MPS lacks max_pool3d, CUDA preferred if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading MMRT model...")
    model, gloss_dict, dataset_info = MMRTInferenceModel.from_pretrained(
        args.weights, args.config, args.dataset_config)
    model = model.to(device).eval()
    print(f"Model loaded. Vocab size: {len(gloss_dict)}")

    translator = None
    if not args.no_llm and os.path.exists(args.train_json):
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        translator = GlossTranslator(args.train_json, api_key=api_key)
        print(f"Translator loaded. Exact map: {len(translator.exact_map)} entries")

    with open(args.test_json, encoding='utf-8') as f:
        samples = json.load(f)
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Test samples: {len(samples)}")

    wer_total = 0
    wer_count = 0
    bleu_total = 0.0
    tier_counts = defaultdict(int)
    results = []
    skipped = 0
    start_time = time.time()

    for i, item in enumerate(samples):
        folder = os.path.join(args.dataset_root, 'frames_512x512', item['name'])
        if not os.path.isdir(folder):
            skipped += 1
            continue

        try:
            video = load_frames(folder)
            if video is None:
                skipped += 1
                continue

            video_input = video.unsqueeze(0).to(device)
            video_length = torch.LongTensor([video.shape[0]]).to(device)

            with torch.no_grad():
                out = model(video_input, video_length)

            recognized = out['recognized_sents'][0]
            ref_glosses = item['label_gloss']
            ref_text = item.get('text', '')

            edit = compute_wer(ref_glosses, recognized)
            wer_total += edit
            wer_count += max(len(ref_glosses), 1)

            if translator:
                trans = translator.translate_list(recognized)
                pred_text = trans['text']
                tier = trans['tier']
            else:
                pred_text = ''.join(recognized)
                tier = 0

            tier_counts[tier] += 1
            bleu = compute_bleu(ref_text, pred_text) if ref_text else 0.0
            bleu_total += bleu

            results.append({
                'id': item['name'],
                'ref_glosses': ref_glosses,
                'pred_glosses': recognized,
                'ref_text': ref_text,
                'pred_text': pred_text,
                'tier': tier,
                'gloss_edit': int(edit),
                'bleu': round(bleu, 4),
            })

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                wer = wer_total / max(wer_count, 1) * 100
                avg_bleu = bleu_total / max(len(results), 1) * 100
                print(f"[{i+1}/{len(samples)}] WER={wer:.1f}% GlossAcc={100-wer:.1f}% BLEU={avg_bleu:.1f}% "
                      f"Skipped={skipped} Time={elapsed:.0f}s")

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Error on sample {i} ({item['name']}): {e}")

    if not results:
        print("No results collected.")
        return

    wer = wer_total / max(wer_count, 1) * 100
    gloss_accuracy = 100.0 - wer
    avg_bleu = bleu_total / len(results) * 100
    tier_dist = {k: v / len(results) * 100 for k, v in tier_counts.items()}

    # Sentence accuracy: BLEU >= 0.5 threshold
    bleu_pass = sum(1 for r in results if r['bleu'] >= 0.5)
    sentence_acc = bleu_pass / len(results) * 100

    print("\n" + "="*60)
    print("MMRT EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset:           CSL-Daily (8:2 split, {len(results)} samples)")
    print(f"Skipped:           {skipped}")
    print(f"Gloss WER:         {wer:.2f}%")
    print(f"Gloss Accuracy:    {gloss_accuracy:.2f}%  {'✓ PASS' if gloss_accuracy >= 90 else '✗'} (target ≥90%)")
    print(f"Avg BLEU-4:        {avg_bleu:.2f}%")
    if translator:
        print(f"Sentence Acc (BLEU≥0.5): {sentence_acc:.2f}%")
        print(f"Translation tiers:")
        for t, pct in sorted(tier_dist.items()):
            labels = {0: 'Fallback', 1: 'Exact', 2: 'Fuzzy+LLM', 3: 'LLM'}
            print(f"  Tier {t} ({labels.get(t,'?')}): {pct:.1f}%")
    print("="*60)

    report = {
        'wer': round(wer, 4),
        'gloss_accuracy': round(gloss_accuracy, 4),
        'bleu4': round(avg_bleu, 4),
        'sentence_accuracy': round(sentence_acc, 4),
        'total': len(results),
        'skipped': skipped,
        'tier_distribution': {str(k): round(v, 2) for k, v in tier_dist.items()},
        'samples': results[:20],
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
