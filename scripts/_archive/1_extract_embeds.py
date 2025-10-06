# import os, json
# import numpy as np
# from tqdm import tqdm
# from speechbrain.pretrained import EncoderClassifier
# from utils_audio import load_wav, crop_or_pad_2s
# import torch

# DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
# print("Using device:", DEVICE)

# SPLIT_JSON = "data/splits.json"
# OUT_DIR = "data/embeds"
# N_AUG = 2 

# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     with open(SPLIT_JSON) as f:
#         splits = json.load(f)

#     classifier = EncoderClassifier.from_hparams(
#         source="speechbrain/spkrec-ecapa-voxceleb",
#         savedir="data/sb_ecapa",
#         run_opts={"device": DEVICE},
#     )

#     rng = np.random.default_rng(1234)

#     for split in ["train", "val", "test"]:
#         X, y = [], []
#         spk_list = sorted(splits[split].keys())
#         spk_to_idx = {spk:i for i,spk in enumerate(spk_list)}
#         for spk in spk_list:
#             for path in tqdm(splits[split][spk], desc=f"{split}:{spk}"):
#                 wav, sr = load_wav(path)
#                 embs = []
#                 for _ in range(N_AUG):
#                     seg = crop_or_pad_2s(wav, rng=rng)     
#                     tensor = torch.from_numpy(seg).unsqueeze(0).to(DEVICE) 
#                     with torch.no_grad():
#                         emb = classifier.encode_batch(tensor)        
#                     emb = emb.squeeze(0).squeeze(0).detach().cpu().numpy()
#                     embs.append(emb)
#                 emb_mean = np.mean(np.stack(embs, axis=0), axis=0)
#                 X.append(emb_mean); y.append(spk_to_idx[spk])
#         X = np.stack(X, 0).astype("float32")
#         y = np.array(y, dtype="int64")
#         np.savez_compressed(os.path.join(OUT_DIR, f"{split}.npz"), X=X, y=y, spk_list=spk_list)
#         print(split, X.shape, y.shape)

# if __name__ == "__main__":
#     main()
