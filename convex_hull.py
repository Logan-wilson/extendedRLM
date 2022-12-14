from pybresenham import line as bline


def isContour(obj, x, y):
    """
    Returns whether a given pixel is a contour of an object using a 4-connected space.

    :param obj: object the pixel comes from.
    :param x: x coordinate of the pixel.
    :param y: y coordinate of the pixel.
    :return: True if pixel(x,y) is a contour of the object, False otherwise.
    """
    if x < 0 or y < 0 or x >= len(obj) or y >= len(obj[0]) or not (obj[x][y]):  # not part of the object -> not a contour.
        return False
    if x == 0 or y == 0 or x == len(obj) - 1 or y == len(obj) - 1:  # is along the border of image so it's an object's contour.
        return True
    # if a pixel in a 4-connected space is void, it means the original pixel is a contour of the object.
    return not (obj[x - 1][y]) or not (obj[x + 1][y]) or not (obj[x][y - 1]) or not (obj[x][y + 1])


def object_contour(obj):
    """
    Goes through all pixels of the image of the object and returns a list of all pixels considered as contour in a
    4-connected space.

    :param obj: object image to detect contours of.
    :return: list of tuples(x,y) with x and y = coordinates of each pixel.
    """
    cont = []
    for i in range(len(obj)):
        for j in range(len(obj)):
            if isContour(obj, i, j):
                cont.append((i, j))
    return cont


def get_corner_pixel(contour):
    """
    Find the pixel that is going to act as the pivot point of the convex hull computation.

    :param contour: list of points considered as contours of the image.
    :return: tuple(x,y), the point minimal x value or y value if multiple points have the same x min value.
    """
    min_pix = (9999, 9999)
    for pixel in contour:
        if pixel[0] < min_pix[0] or (pixel[0] == min_pix[0] and pixel[1] < min_pix[1]):
            min_pix = pixel
    return min_pix


def compute_slope(p1, p2):
    """
    Compute the slope of a vector defined by its 2 ends. If the x value of both points are equals, return 1 to avoid
    division by zero.

    :param p1: first end of the vector.
    :param p2: second end of the vector.
    :return: slope of the vector.
    """
    if p2[0] - p1[0] == 0:
        return 1
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def compute_power(line_p1, line_p2, point):
    """
    Compute the power of a vector to a point.
    :param line_p1: first endpoint of the vector.
    :param line_p2: second endpoint of the vector.
    :param point: point used to compare the vector to.
    :return: float value, either negative or positive (>0 -> left side of the vector, <0 -> right side of the vector).
    """
    dy = line_p2[1] - line_p1[1]
    dx = line_p2[0] - line_p1[0]
    return int(dy * (point[0] - line_p1[0]) - dx * (point[1] - line_p1[1]))


def sortby_slope(minpix, contour):
    """
    Create a copy and sort the list of contour points according to the slope of each point compared to the starting
    point of the convex hull computation.
    :param minpix: starting point of the convex hull computation (min(x), min(y)).
    :param contour: list of the points considered as a contour.
    :return:sorted list.
    """
    sorted_contour = sorted(contour, key=lambda s: compute_slope(minpix, s))
    return sorted_contour


def convex_hull(obj):
    """
    compute the convex hull of an object using Graham's algorithm.

    1) find the pivot point (x min, or y min).
    2) compute slopes of line of pivot point to all other points of the object.
    3) sort points by their slope (with the pivot point).
    4) remove incrementally concave angles using the power of a segment to a point.

    :param obj: object to compute the convex hull of.
    :return: point used as starting point, list of points from the convex hull.
    """
    contour = object_contour(obj)
    min_pix = get_corner_pixel(contour)
    sorted_contour = sortby_slope(min_pix, contour)
    ch = sorted_contour
    i = 1
    while i < len(ch) - 1:
        power = compute_power(ch[i - 1], ch[i + 1], ch[i])
        if power >= 0:
            i += 1
        else:
            ch.pop(i)
            i -= 1
    return min_pix, ch


def compute_polygon(min_pix, ch):
    """
    compute the polygon that acts as a convex hull of an object.
    :param min_pix: starting point of the convex hull algorithm.
    :param ch: list of points acting as the convex hull.
    :return: list of bresenham points to draw.
    """
    points = list(bline(min_pix[0], min_pix[1], ch[0][0], ch[0][1]))
    for i in range(len(ch) - 1):
        x1 = ch[i]
        x2 = ch[i + 1]
        points.extend(list(bline(x1[0], x1[1], x2[0], x2[1])))
    points.extend(list(bline(ch[-1][0], ch[-1][1], min_pix[0], min_pix[1])))
    return points


def get_closest_convex_hull_points(points1, points2):
    """
    Compute the shortest line from any two pixels of each convex hulls. Used for the computation of the middle point of the
    RLM.

    :param points1: list of points from the first convex hull.
    :param points2: list of points from the second convex hull.
    :return: middle point of the shortest line.
    """
    min1 = 0
    min2 = 0
    dist = 9999
    for p1 in points1:
        for p2 in points2:
            if manhattan(p1[0], p1[1], p2[0], p2[1]) < dist:
                dist = manhattan(p1[0], p1[1], p2[0], p2[1])
                min1 = p1
                min2 = p2
    return min1, min2


def connected8_check(x, y, obj_list):
    """
    Checks if, for a given x and y values, a point is in an 8-connected space.

    :param x: x value of the point.
    :param y: y value of the point.
    :param obj_list: list of points to check, is mainly used to compare intersection with a convex hull.
    :return: True if (x,y) are coordinates of, or a neighbor of a point in the list.
    """
    matrix = [(x + 1, y + 1), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x, y), (x, y - 1), (x - 1, y + 1), (x - 1, y), (x - 1, y - 1)]
    return any(pt in obj_list for pt in matrix)


def manhattan(x1, y1, x2, y2):
    """
    return the manhattan distance of two points. Since the distance is computed on discretized points (pixels), the
    distance is an integer.

    :param x1: x value of first point.
    :param y1: y value of first point.
    :param x2: x value of second point.
    :param y2: y value of second point.
    :return: manhattan distance.
    """
    return abs(x2 - x1) + abs(y2 - y1)