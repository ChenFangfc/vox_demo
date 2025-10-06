import os, glob, json, random
random.seed(0)
ROOT="hf_vox/vox1/vox1_train_wav/wav"
spks=sorted(glob.glob(os.path.join(ROOT,"id*")))
def pick(fs, k=20):
    random.shuffle(fs); return fs[:min(k,len(fs))]
items=[]
for spk_dir in spks:
    wavs=glob.glob(os.path.join(spk_dir,"*","*.wav"))
    if len(wavs)>=3:
        for w in pick(wavs, 20): items.append((os.path.basename(spk_dir), w))
random.shuffle(items)
n=len(items); tr=int(n*0.7); va=int(n*0.15)
splits={"train":items[:tr], "val":items[tr:tr+va], "test":items[tr+va:]}
os.makedirs("data/embeds_closedset", exist_ok=True)
for k,v in splits.items():
    with open(f"data/embeds_closedset/{k}.json","w") as f:
        json.dump(v,f)
print({k:len(v) for k,v in splits.items()})
