import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_split(name):
    d = np.load(f"data/embeds/{name}.npz", allow_pickle=True)
    return d["X"], d["y"], d["spk_list"]

Xtr, ytr, _ = load_split("train")
Xva, yva, _ = load_split("val")
Xte, yte, _ = load_split("test")

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(Xtr, ytr)
print("LogReg val acc:", accuracy_score(yva, clf.predict(Xva)))
print("LogReg test acc:", accuracy_score(yte, clf.predict(Xte)))
