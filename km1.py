import numpy as np

data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

k = 3

centroids = data[[data[d] for d in range(k)]]

print(centroids)

while True:
        cluster = [[] for _ in range(k)]        # [ [], [], [] ]

        for points in data:
                distance = [np.abs(points - c) for c in centroids]      # sab cendroid se distance nikala

                nearest_index = np.argmin(distance)     # sabse chota distance nikala, uska index niakala
                cluster[nearest_index].append(points)   # uss index ke list me append kiya 

        new_centroids = np.array([
                np.mean(c) if len(c) > 0 else centroids[i] for i, c in enumerate(cluster)
        ])

        if np.allclose(centroids, new_centroids):
                break

        centroids = new_centroids

for i, c in enumerate(cluster, 1):
        print("Cluster ", i, ": ")
        for point in c:
                print(point)

print("Final Centroids: ", centroids)



