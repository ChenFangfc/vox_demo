# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# def load_split(name):
#     d = np.load(f"data/embeds/{name}.npz", allow_pickle=True)
#     return d["X"], d["y"], d["spk_list"]

# Xtr, ytr, _ = load_split("train")
# Xva, yva, _ = load_split("val")
# Xte, yte, _ = load_split("test")

# knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
# knn.fit(Xtr, ytr)
# print("KNN val acc:", accuracy_score(yva, knn.predict(Xva)))
# print("KNN test acc:", accuracy_score(yte, knn.predict(Xte)))
