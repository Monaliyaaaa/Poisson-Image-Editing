"""
Sylvia Liu
122090359
"""

import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix


# Define a function to determine whether the point is outside or not.
def is_outside(point, mask):
    # 0 means black, which means the point is outside.
    return mask[point] == 0


# Define a function to determine whether the point is inside or on the boundary.
def is_inside(point, mask):
    for pt in get_neighbors(point):
        # if one of those neighbors is outside, then this point is on the boundary.
        if is_outside(pt, mask):
            return False
    return True


# To get the neighborhood points.
def get_neighbors(point):
    i, j = point  # 'point' is a two-dimensional array.
    return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]


# Use the Laplacian operator.
def laplacian(source, point):
    i, j = point
    result = (4 * source[i, j]) - (1 * source[i + 1, j]) - (1 * source[i - 1, j]) - (1 * source[i, j + 1]) - (1 * source[i, j - 1])
    return result


# Get the points where the mask is white/1.
def get_points(mask):
    m = np.nonzero(mask)
    return list(zip(m[0], m[1]))


# Create a sparse matrix
def sparse(points):
    lens = len(points)
    A = lil_matrix((lens, lens))
    for i, pt in enumerate(points):
        A[i, i] = 4  # diagonal
        for a in get_neighbors(pt):
            if a not in points:
                continue  # This point is on the boundary.
            b = points.index(a)  # Store the index of variable a in variable b.
            A[i, b] = -1
    return A


# Main method : execute image editing for each channel.
def img_edit(source, target, mask):
    points = get_points(mask)
    A = sparse(points)
    b = np.zeros(len(points))  # Create a nx1 matrix, where n is ken(points).
    for i, pt in enumerate(points):
        b[i] = laplacian(source, pt)
        # If the point is on the boundary: add target constraints.
        if not is_inside(pt, mask):
            for a in get_neighbors(pt):
                if is_outside(a, mask):
                    b[i] += target[a]

    # Solve the matrix equation.
    x = linalg.cg(A, b)
    # Copy the target photo first, for the outside part.
    composite = np.copy(target).astype(int)
    # Then add new pixel value on the inside part.
    for i, pt in enumerate(points):
        composite[pt] = x[0][i].astype(int)
    return composite
