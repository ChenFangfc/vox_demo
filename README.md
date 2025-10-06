python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf_vox/vox1/{vox1_test_wav.zip, vox1_dev_wav.zip}
hf_vox/vox1/txt/veri_test.txt
mkdir -p hf_vox/vox1/vox1_test_wav hf_vox/vox1/vox1_train_wav
unzip -n hf_vox/vox1/vox1_test_wav.zip -d hf_vox/vox1/vox1_test_wav
unzip -n hf_vox/vox1/vox1_dev_wav.zip  -d hf_vox/vox1/vox1_train_wav
python scripts/4_eval_verification.py
python scripts/4b_save_scores_and_roc.py
python scripts/spk_cli.py A.wav B.wav
