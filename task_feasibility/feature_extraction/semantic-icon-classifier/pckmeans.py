import numpy as np
from collections import Counter
from tqdm import tqdm
import gc

np.random.seed(1000)


def _initialize_centroids(points, k):
    """returns k initial points as centroids"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def pckmeans(points, num_clusters, num_iter,
             constrained_pts_map, pos_violation_weight):
    centroids = _initialize_centroids(points, num_clusters)
    gc.collect()
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    cluster_assignments = np.argmin(distances, axis=0)
    # print   c.size
    # c = centroids

    num_iter_bar = tqdm(range(num_iter), desc="PCKmeans iter")
    for i in num_iter_bar:
        # print "kmeans iteration number: {}/{}".format(i, num_iter)
        # Step 1: assign points to clusters

        # Step 1.1: Find the distances of each pt from each cluster center
        distances = np.sqrt(
            ((points - centroids[:, np.newaxis])**2).sum(axis=2))

        # Step 1.2 Find the number of violations for different cluster assignments
        # This is going to be slow :-(
        num_violations = np.zeros((num_clusters, points.shape[0]))
        all_constrained_pts = constrained_pts_map.keys()
        for pt_idx in all_constrained_pts:
            # we skip the points that have no constraints
            constrained_pts = constrained_pts_map[pt_idx]
            total_constraints = len(constrained_pts)

            constrained_pts_clusters = [
                cluster_assignments[pt] for pt in constrained_pts]
            cluster_counts = Counter(constrained_pts_clusters)
            for cluster_idx in range(num_clusters):
                num_violations[cluster_idx][pt_idx] = total_constraints - \
                    cluster_counts.get(cluster_idx, 0)

        # 1.3 Combine distances and violations to form loss function
        losses = distances + pos_violation_weight * num_violations

        # 1.4 Pick clusters for each function so that loss function is
        # minimized
        cluster_assignments = np.argmin(losses, axis=0)

        # Step 1.2: Print out average distance of pt from cluster center to see
        # progress
        min_distances = np.amin(distances, axis=0)
        avg_distance_per_dim = min_distances.sum(
        ) / float(points.shape[0] * points.shape[1])
        num_iter_bar.set_postfix(dist=avg_distance_per_dim)
        num_iter_bar.write("     PCKmeans iter {} : {:,}".format(i, avg_distance_per_dim))
        # print "     PCKmeans iter {} : {:,}".format(i, avg_distance_per_dim)

        # Step 2: Update cluster centroids
        # c = move_centroids(points, closest_centroids, c)
        # print [len(points[cluster_assignments==k]) for k in
        # range(centroids.shape[0])]
        centroids = np.array([points[cluster_assignments == k].mean(axis=0) if len(points[cluster_assignments == k]) != 0 else
                              centroids[k] for k in range(centroids.shape[0])])  # if there are no points assigned to a cluster,
        # its center does not change

    return (cluster_assignments, centroids)
