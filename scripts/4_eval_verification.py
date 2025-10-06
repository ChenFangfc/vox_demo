import os, sys, numpy as np
import torch
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier
from utils_audio import load_wav, crop_or_pad_2s

DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

TEST_WAV_ROOT = "hf_vox/vox1/vox1_test_wav/wav"
VERI_FILE = "hf_vox/vox1/txt/veri_test.txt"
N_AUG = 5

def get_embedding(classifier, wav_path):
    wav, sr = load_wav(wav_path)
    embs = []
    rng = np.random.default_rng(1234)
    for _ in range(N_AUG):
        seg = crop_or_pad_2s(wav, rng=rng)
        t = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = classifier.encode_batch(t)
        embs.append(e.squeeze(0).squeeze(0).detach().cpu().numpy())
    return np.mean(np.stack(embs,0), axis=0)

def cosine(a,b):
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a,b) / (na*nb))

def compute_eer(labels, scores):
    # labels: 1=same spk, 0=different spk
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    # sweep thresholds over all unique scores
    idx = np.argsort(scores)
    thresholds = scores[idx]
    P = labels.sum()
    N = len(labels) - P
    best_diff = 1.0
    eer = None
    for th in thresholds:
        # decide same if score >= th
        yhat = (scores >= th).astype(np.int32)
        fp = np.sum((yhat==1) & (labels==0))
        fn = np.sum((yhat==0) & (labels==1))
        fpr = fp / (N+1e-12)
        fnr = fn / (P+1e-12)
        diff = abs(fpr - fnr)
        if diff < best_diff:
            best_diff = diff
            eer = 0.5*(fpr+fnr)
    return float(eer)

def main():
    assert os.path.isfile(VERI_FILE), f"Missing {VERI_FILE}"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="data/sb_ecapa_eval",
        run_opts={"device": DEVICE},
    )
    pairs = []
    with open(VERI_FILE) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split()
            if len(parts)!=3: 
                print("Skip line:", line); continue
            lab, p1, p2 = parts
            lab = int(lab)
            pairs.append((lab, os.path.join(TEST_WAV_ROOT, p1), os.path.join(TEST_WAV_ROOT, p2)))

    cache = {}
    labels, scores = [], []
    for lab, f1, f2 in tqdm(pairs, total=len(pairs)):
        if f1 not in cache: cache[f1] = get_embedding(classifier, f1)
        if f2 not in cache: cache[f2] = get_embedding(classifier, f2)
        s = cosine(cache[f1], cache[f2])
        labels.append(lab); scores.append(s)

    eer = compute_eer(labels, scores)
    print(f"EER: {eer*100:.2f}%   (lower is better)")

if __name__ == "__main__":
    main()
