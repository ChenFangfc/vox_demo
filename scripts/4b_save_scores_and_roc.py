import os, glob, json, numpy as np, torch
from tqdm import tqdm

# SpeechBrain import (backward compatible)
try:
    from speechbrain.pretrained import EncoderClassifier
except Exception:
    from speechbrain.inference import EncoderClassifier

from utils_audio import load_wav, crop_or_pad_2s

DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
TEST_WAV_ROOT = "hf_vox/vox1/vox1_test_wav/wav"

def find_veri_file():
    for pat in ["hf_vox/vox1/txt/veri_test.txt",
                "hf_vox/vox1/txt/*veri*test*.txt",
                "hf_vox/vox1/*veri*test*.txt"]:
        for p in glob.glob(pat):
            if os.path.isfile(p):
                return p
    return None

def parse_pairs(veri_file, root):
    pairs=[]
    with open(veri_file) as f:
        for line in f:
            parts=line.strip().split()
            if len(parts)!=3: continue
            lab, r1, r2 = int(parts[0]), parts[1], parts[2]
            pairs.append((lab, os.path.join(root,r1), os.path.join(root,r2)))
    return pairs

def get_embedding(enc, path, n_aug=5):
    wav, sr = load_wav(path)
    rng = np.random.default_rng(1234)
    embs=[]
    for _ in range(n_aug):
        seg = crop_or_pad_2s(wav, rng=rng)
        t = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = enc.encode_batch(t)
        embs.append(e.squeeze(0).squeeze(0).cpu().numpy())
    return np.mean(np.stack(embs,0),0)

def cosine(a,b):
    na=np.linalg.norm(a)+1e-9; nb=np.linalg.norm(b)+1e-9
    return float(np.dot(a,b)/(na*nb))

def eer_and_points(labels, scores):
    labels=np.asarray(labels).astype(np.int32)
    scores=np.asarray(scores).astype(np.float64)
    idx=np.argsort(scores)
    ths=scores[idx]
    P=labels.sum(); N=len(labels)-P
    best_diff=1.0; eer=None; eer_th=None
    roc=[]
    for th in ths:
        yhat=(scores>=th).astype(np.int32)
        fp=np.sum((yhat==1)&(labels==0))
        tp=np.sum((yhat==1)&(labels==1))
        fn=np.sum((yhat==0)&(labels==1))
        fpr=fp/(N+1e-12); tpr=tp/(P+1e-12); fnr=fn/(P+1e-12)
        roc.append((fpr,tpr,th))
        diff=abs(fpr-fnr)
        if diff<best_diff:
            best_diff=diff; eer=0.5*(fpr+fnr); eer_th=th
    return float(eer), float(eer_th), np.array(roc)

def threshold_at_target_fpr(roc, target_fpr=0.01):
    roc=roc[np.argsort(roc[:,0])]
    idx=np.searchsorted(roc[:,0], target_fpr, side="left")
    idx=min(idx, len(roc)-1)
    return float(roc[idx,2]), float(roc[idx,0]), float(roc[idx,1])

def main():
    os.makedirs("data/analysis", exist_ok=True)
    veri=find_veri_file()
    if not veri:
        raise SystemExit("No veri_test.txt found. Put it in hf_vox/vox1/txt and rerun.")
    print("Using:", veri)

    enc=EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="data/sb_ecapa_eval",
        run_opts={"device": DEVICE},
    )

    pairs=parse_pairs(veri, TEST_WAV_ROOT)
    cache={}; labels=[]; scores=[]
    for lab,f1,f2 in tqdm(pairs, total=len(pairs)):
        if f1 not in cache: cache[f1]=get_embedding(enc,f1)
        if f2 not in cache: cache[f2]=get_embedding(enc,f2)
        labels.append(lab); scores.append(cosine(cache[f1],cache[f2]))

    # save raw scores
    import csv
    out_csv="data/analysis/voxceleb1_scores.csv"
    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["label","score"])
        for l,s in zip(labels,scores): w.writerow([l,s])
    print("Wrote", out_csv)

    # compute thresholds
    eer, eer_th, roc = eer_and_points(labels, scores)
    th_1fpr, got_fpr, got_tpr = threshold_at_target_fpr(roc, target_fpr=0.01)
    with open("data/analysis/thresholds.json","w") as f:
        json.dump({"eer":eer, "eer_threshold":eer_th,
                   "thr_at_1pct_fpr": th_1fpr,
                   "achieved_fpr": got_fpr, "achieved_tpr": got_tpr}, f, indent=2)
    print(f"EER={eer*100:.2f}% at th={eer_th:.4f} | FPR~1% th={th_1fpr:.4f} (TPR={got_tpr*100:.1f}%)")

    # plot ROC
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(roc[:,0], roc[:,1], lw=2)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("VoxCeleb1 Test ROC (ECAPA)")
    plt.grid(True)
    plt.savefig("data/analysis/roc.png", dpi=150, bbox_inches="tight")
    print("Saved ROC to data/analysis/roc.png")

if __name__=="__main__":
    main()
