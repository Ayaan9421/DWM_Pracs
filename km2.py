import numpy as np
import math 

data = np.array([
        [2,10],
        [2,5],
        [8,4],
        [5,8],
        [7,5],
        [6,4],
        [1,2],
        [4,9]
])

k = 3 
centroids = data[np.random.choice(len(data), k, replace= False)]
print(centroids)

while True:
        cluster = [[] for _ in range(k)]

        for point in data:
                distance = [ math.dist(point, c) for c in centroids]

                nearest_index = np.argmin(distance)
                cluster[nearest_index].append(point)

        new_centroid = np.array([
                np.mean(c, axis=0) if len(c) > 0 else centroids[i] for i,c in enumerate(cluster)
        ])

        if np.allclose(centroids, new_centroid):
                break

        centroids = new_centroid

for i,c in enumerate(cluster,1):
        print("Cluster: ",i)
        for point in c:
                print(point)

print("Final Centroids: \n", centroids)