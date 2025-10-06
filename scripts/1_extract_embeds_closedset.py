import os, json, numpy as np, torch
from tqdm import tqdm
try:
    from speechbrain.pretrained import EncoderClassifier
except:
    from speechbrain.inference import EncoderClassifier
from utils_audio import load_wav, crop_or_pad_2s
DEVICE="mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"
enc=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                   savedir="data/sb_ecapa_closed", run_opts={"device":DEVICE})
def emb(path, n_aug=3):
    import numpy as np, torch
    wav, sr=load_wav(path); rng=np.random.default_rng(1234); xs=[]
    for _ in range(n_aug):
        seg=crop_or_pad_2s(wav, rng=rng)
        t=torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): e=enc.encode_batch(t)
        xs.append(e.squeeze(0).squeeze(0).cpu().numpy())
    return np.mean(np.stack(xs,0),0)
def make(split):
    import numpy as np
    pairs=json.load(open(f"data/embeds_closedset/{split}.json"))
    X=[]; y=[]
    for spk, wav in tqdm(pairs, desc=split):
        X.append(emb(wav)); y.append(spk)
    np.savez(f"data/embeds_closedset/{split}.npz", X=np.stack(X,0), y=np.array(y))
for s in ["train","val","test"]: make(s)
