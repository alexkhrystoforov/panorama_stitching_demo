import numpy as np
import statistics


def find_best_matches(matches):
    """
    Filter matches by distance
    Args:
         matches: list
    Returns:
        best_matches: list
    """
    best_matches = []
    for m in matches:
        if m.distance < 30:
            best_matches.append(m)

    return best_matches


def lowe_ratio(matches, ratio_thresh):
    """
    Filter matches using the Lowe's ratio test
    Args:
        matches: list
        ratio_thresh: float
    Returns:
        best_matches: list
     """

    best_matches = []

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                best_matches.append(m)

        except ValueError:
            pass

    return best_matches


def check_points(dst):
    ''' The idea is to calculate distance between object corners points and then find
        difference between the longest and the shortest length between points to avoid
        strange shapes. '''
    status = True
    all_dist = []

    if len(dst) == 4:
        for i in range(len(dst)):

            if i == 0:
                dist = np.linalg.norm(dst[[0]] - dst[[1]])
                all_dist.append(dist)
            else:
                dist = np.linalg.norm(dst[[i - 1]] - dst[[i]])
                all_dist.append(dist)

        all_dist.sort()

        # difference between the longest and the shortest length between points
        difference = all_dist[-1] - all_dist[0]

        if difference > all_dist[0] * 1.9 or min(all_dist) < 20:
            status = False

    return status

