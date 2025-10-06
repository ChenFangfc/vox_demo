# import os, json, random, glob
# from collections import defaultdict

# random.seed(100)

# DEV_WAV_ROOT = "hf_vox/vox1/vox1_dev_wav/wav"
# TEST_WAV_ROOT = "hf_vox/vox1/vox1_test_wav/wav"
# OUT_SPLIT_JSON = "data/splits.json"

# def collect_spk_to_wavs(root):
#     spk_to_wavs = defaultdict(list)
#     for spk_dir in sorted(glob.glob(os.path.join(root, "id*"))):
#         spk = os.path.basename(spk_dir)
#         for wav in glob.glob(os.path.join(spk_dir, "*", "*.wav")):
#             spk_to_wavs[spk].append(wav)
#     return spk_to_wavs

# def main():
#     assert os.path.isdir(DEV_WAV_ROOT), f"Missing {DEV_WAV_ROOT}"
#     assert os.path.isdir(TEST_WAV_ROOT), f"Missing {TEST_WAV_ROOT}"

#     dev_spk2wavs = collect_spk_to_wavs(DEV_WAV_ROOT)
#     test_spk2wavs = collect_spk_to_wavs(TEST_WAV_ROOT)

#     dev_spks = list(dev_spk2wavs.keys())
#     random.shuffle(dev_spks)

#     n_dev = len(dev_spks)
#     n_val = max(1, int(0.1 * n_dev))
#     val_spks = set(dev_spks[:n_val])
#     train_spks = set(dev_spks[n_val:])

#     splits = {"train": {}, "val": {}, "test": {}}
#     for spk in train_spks:
#         splits["train"][spk] = dev_spk2wavs[spk]
#     for spk in val_spks:
#         splits["val"][spk] = dev_spk2wavs[spk]
#     for spk, wavs in test_spk2wavs.items():
#         splits["test"][spk] = wavs

#     os.makedirs("data", exist_ok=True)
#     with open(OUT_SPLIT_JSON, "w") as f:
#         json.dump(splits, f, indent=2, ensure_ascii=False)
#     print(f"Saved splits to {OUT_SPLIT_JSON}")
#     print(f"Speakers -> train:{len(train_spks)}  val:{len(val_spks)}  test:{len(test_spk2wavs)}")

# if __name__ == "__main__":
#     main()
