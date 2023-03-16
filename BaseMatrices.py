import sys
from typing import Tuple, List

import numpy as np
from itertools import combinations
from scipy.special import comb
from PIL import Image

"""
Code for generating base matrices to use for combinatorial inkjet printing.
Please cite https://doi.org/10.1039/d1ta08841f if used.
"""


def combs(a: np.ndarray, r: int) -> np.ndarray:
    """
    Return successive r-length combinations of elements in the array a.
    Should produce the same output as array(list(combinations(a, r))), but
    faster.
    From:
    https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy

    :param a: Array with elements to make combinations from
    :type a: numpy.ndarray
    :param r: Number of elements to combine
    :type r: int
    :return: 2d array of all combinations
    :rtype: numpy.ndarray
    """
    a = np.asarray(a)
    dt = np.dtype([('', a.dtype)] * r)
    b = np.fromiter(combinations(a, r), dt)
    return b.view(a.dtype).reshape(-1, r)


def droplet_position_optimization(n: int, k: int) -> Tuple[List[int], List[int]]:
    """
    Function for generating a base matrix with side n and filling k.

    :param n: Side length of base matrix
    :type n: int
    :param k: Filling of the base matrix
    :type k: int
    :return: Tuple of two lists of the optimal droplet positions,
             one for the x position and one for the corresponding y positions
    :rtype: Tuple[List[int], List[int]]
    """
    x, y = np.indices((n, n))
    pos = combs(np.arange(n ** 2), k)
    x_base_pos = x.flatten()[pos]
    y_base_pos = y.flatten()[pos]
    points = comb(n ** 2, k, exact=True)
    x_pos = np.zeros((points, k * 9))
    y_pos = np.zeros((points, k * 9))
    count = 0
    for y_rep in [-1, 0, 1]:
        for x_rep in [-1, 0, 1]:
            roi = slice(k * count, k * (count + 1))
            x_pos[:, roi] = x_base_pos + (x_rep * n)
            y_pos[:, roi] = y_base_pos + (y_rep * n)
            count += 1
    x_pos = np.tile(x_pos, (k, 1, 1))
    y_pos = np.tile(y_pos, (k, 1, 1))
    distances = np.sqrt(
        np.power(x_pos - np.broadcast_to(x_base_pos, (k * 9, points, k)).T, 2)
        + np.power(y_pos - np.broadcast_to(y_base_pos, (k * 9, points, k)).T, 2)
    )
    distances[np.where(distances < 1)] = (2 * n) ** 2
    min_dist = np.min(distances, axis=2)
    worst_point = np.argmin(min_dist, axis=0)
    if np.max(min_dist[worst_point, np.arange(points)]) > 1:
        best_option_index = np.argmax(min_dist[worst_point, np.arange(points)])
    else:
        worst_min_dist = np.min(min_dist, axis=0)
        min_dist_copy = min_dist
        min_dist_copy[np.where(min_dist != worst_min_dist)] = 0
        best_option_index = np.argmin(np.sum(min_dist_copy, axis=0))
    return x_base_pos[best_option_index], y_base_pos[best_option_index]


def generate_bases(size: int) -> List[np.ndarray]:
    """
    Function for generating all base matrices for a certain size.

    :param size: Side length of the base matrix
    :type size: int
    :return: A list of the base matrices as size*size numpy arrays
    :rtype: List[numpy.ndarray]
    """
    if size > 5:
        print('For size above 4 the algorithm takes a long time.')
    high = int(np.ceil((size ** 2 + 1) / 2)) - 1
    bases = [np.zeros((size, size), dtype=bool)]
    for fill in range(high):
        best_x, best_y = droplet_position_optimization(size, fill + 1)
        base = np.zeros((size, size), dtype=bool)
        for idx in range(len(best_x)):
            base[best_y[idx], best_x[idx]] = True
        bases.append(base)
    remaining = size ** 2 + 1 - len(bases)
    for idx in range(remaining):
        bases.append(np.invert(bases[remaining - idx - 1]))
    return bases


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from matplotlib import patches

    base_size = 0
    if len(sys.argv) != 3:
        print("Please provide 2 arguments\n" +
              "    1st: Size of the base\n" +
              "    2nd: Location to save images")
    try:
        base_size = int(sys.argv[1])
    except ValueError:
        print("Unable to convert first argument to integer.")
        sys.exit(-1)

    save_dir = sys.argv[2]
    if not os.path.isdir(save_dir):
        print('"%s" is not a directory' % save_dir)
        sys.exit(-1)
    test_bases = generate_bases(base_size)

    fig, axs = plt.subplots(ncols=base_size, nrows=base_size, figsize=(6, 5))
    for base_idx, test_base in enumerate(test_bases):
        img = Image.fromarray(test_base.T)
        img.save(
            os.path.join(
                save_dir,
                'base_%d_fill_%d_of_%d.png' % (base_size, base_idx, len(test_bases))
            ),
            bits=1, optimize=True
        )
        if base_idx > 0:
            ax = axs[(base_idx - 1) // base_size, (base_idx - 1) % base_size]
            ax.imshow(np.tile(test_base, [3, 3]), cmap='binary', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(
                patches.Rectangle((base_size - 0.5, base_size - 0.5),
                                  base_size, base_size, edgecolor='C3',
                                  fill=False)
            )

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'base_%d.png') % base_size, dpi=300)
