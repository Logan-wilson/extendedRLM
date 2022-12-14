import json
import math

import numpy as np
from PIL import Image
from convex_hull import *


def image_segmentation(imagename: str, backgroundcolor):
    """
    Segment images such as SimpleShapes images where the background color is removed, and masks of objects of the same
    colors are obtained.

    :param imagename: name of the image to mask.
    :param backgroundcolor: Color of the background to remove.
    :return: list of binary images (mask of object 1, mask of object 2).
    """
    filepath = imagename.split("/")
    filename = filepath[-1]
    path = '/'.join(filepath[:-1])
    im = Image.open(f"{path}/{filename}")
    im = im.convert('RGBA')
    data = np.array(im)
    colors = set(tuple(clr) for p in data for clr in p)
    colors.remove(backgroundcolor)
    objects = []
    for n, c in enumerate(colors):
        data_copy = data.copy()
        R, G, B, A = data.T
        non_colored_area = (R != c[0]) & (G != c[1]) & (B != c[2])
        data_copy[..., :-1][non_colored_area.T] = (0, 0, 0)
        colored_area = ~non_colored_area
        data_copy[..., :-1][colored_area.T] = (255, 255, 255)
        objects.append(colored_area)
    return objects


def center_point(objects):
    """
    Find the center points between both object using convex hulls. The point found correponds to the point from which
    half lines are drawn to compute the radial line model.

    :param objects: both objects found in the images.
    :return: x and y coordinates of the middle point.
    """
    hulls = []
    for obj in objects:
        min_pix, c_hull = convex_hull(obj)
        hull = []
        size_c_hull = len(c_hull)
        for p in range(size_c_hull):
            hull.extend(list(
                bline(c_hull[p][0], c_hull[p][1], c_hull[(p + 1) % size_c_hull][0], c_hull[(p + 1) % size_c_hull][1])))
        hulls.append(hull)
    pt1 = barycentre(objects[0])
    pt2 = barycentre(objects[1])
    lin = list(bline(pt1[0], pt1[1], pt2[0], pt2[1]))
    it1 = False
    it2 = False
    fp1 = None
    fp2 = None
    for pt in lin:  # detects when the line crosses the convex hull using an 8-connected space.
        if connected8_check(pt[0], pt[1], hulls[0]) and not it1:
            it1 = True
            fp1 = (pt[0], pt[1])
        if connected8_check(pt[0], pt[1], hulls[1]) and not it2:
            it2 = True
            fp2 = (pt[0], pt[1])
        if it1 and it2:
            break
    if fp1 is None:
        fp1 = pt1
    if fp2 is None:
        fp2 = pt2
    return int((fp1[0] + fp2[0]) / 2), int((fp1[1] + fp2[1]) / 2)


def lines_diameters(objects, x, y, step):
    """
    Returns a list of lines and diameters from which to compute forces and the radial line mode.

    :param objects: objects to compute data from.
    :param x: x coordinate of the middle points.
    :param y: y coordinate of the middle points.
    :param step: step of the angle.
    :return: (List of z half-lines, List of z/2 diameters)
    """
    radius = radial_line_model_radius(objects, x, y)
    lines = []
    diameters = []
    for i in range(int(math.pi * 2 / step)):
        point = round(x + radius * math.cos(step * -i)), round(y + radius * math.sin(step * -i))
        point_2 = round(x + radius * math.cos(step * -i + math.pi)), round(y + radius * math.sin(step * -i + math.pi))
        lines.append(list(bline(x, y, point[0], point[1])))
        diameters.append(list(bline(point[0], point[1], point_2[0], point_2[1])))
    return lines, diameters


def barycentre(obj):
    """
    Return the barycentre of the object.

    :param obj: object to compute the barycentre from : usually 2D binary list.
    :return:x and y coordinates.
    """
    x_pixels = [x for x in range(len(obj)) for y in range(len(obj[x])) if obj[x][y]]
    y_pixels = [y for x in range(len(obj)) for y in range(len(obj[x])) if obj[x][y]]
    return int(sum(x_pixels) / len(x_pixels)), int(sum(y_pixels) / len(y_pixels))


def radial_line_model_radius(objects, x, y):
    """
    Compute the minimum radius of the RLM needed to cover the two objects to compute Spatial Relations from.

    :param objects: list of two 2D binary list. (binary masks of both of the objects).
    :param x: x coordinates of the middle point.
    :param y: y coordinates of the middle point.
    :return: Integer: maximum length between the middle point and the farthest point of any object.
    """
    bounding_boxes = []
    for obj in objects:
        maxX = maxY = 0
        minX = minY = len(obj)
        for i, listJ in enumerate(obj):
            for j, value in enumerate(listJ):
                if value:
                    minX = i if i < minX else minX
                    minY = j if j < minY else minY
                    maxX = i if i > maxX else maxX
                    maxY = j if j > maxY else maxY
        bounding_boxes.append([minX, maxX, minY, maxY])
    x_max = 0
    y_max = 0
    o = 0
    for bb in bounding_boxes:
        for x1 in range(2):
            for y1 in range(2):
                if manhattan(x, y, bb[x1], bb[y1 + 2]) > o:
                    o = manhattan(x, y, bb[x1], bb[y1 + 2])
                    x_max = bb[x1]
                    y_max = bb[y1 + 2]
    return int(math.dist((x, y), (x_max, y_max)))


def forces(objects, diameters, force_type):
    """
    Compute forces between the objects.

    :param objects: list of two 2D binary list. (binary masks of both of the objects).
    :param diameters: list of diameters from the middle point to compute forces from.
    :param force_type: type of force, usually f=0 or f=2 is used.
    :return: force computed from each diameter in a list. Can be constructed as a histogram.
    """
    travels = []
    for line in diameters:
        line_travel = []
        for pt in line:
            if 0 <= pt[0] < len(objects[1]) and 0 <= pt[1] < len(objects[1][0]) and objects[1][pt[0]][pt[1]]:
                line_travel.append("A")
            elif 0 <= pt[0] < len(objects[0]) and 0 <= pt[1] < len(objects[0][0]) and objects[0][pt[0]][pt[1]]:
                line_travel.append("B")
            else:
                line_travel.append("_")
        travels.append(line_travel)
    forces = []
    for i, fline in enumerate(travels):
        force = 0
        if 'A' in fline and 'B' in fline:
            for p in range(len(fline)):
                if fline[p] == 'A':
                    for r in range(p, len(fline)):
                        if fline[r] == 'B':
                            if force_type == 0:
                                force += int(math.dist((diameters[i][p][0], diameters[i][p][1]), (diameters[i][r][0], diameters[i][r][1])))
                            else:
                                dist = int(math.dist((diameters[i][p][0], diameters[i][p][1]), (diameters[i][r][0], diameters[i][r][1])))
                                force += math.log(pow(dist + 1, 2) / (dist * (dist + 2)))
        forces.append(force)
    return forces


def radial_line_model(lines, objects):
    """
    Compute the Radial Line Model data  from the objects on the lines.

    :param lines: half-lines from the middle point.
    :param objects: list of two 2D binary list. (binary masks of both of the objects).
    :return: list of the size of the number of half-lines.
    """
    histObj1 = []
    histObj2 = []
    for line in lines:
        length = len(line)
        histObj1.append(len([p for p in line if point_overlap(p, objects[0])]) / length)
        histObj2.append(len([p for p in line if point_overlap(p, objects[1])]) / length)
    return histObj1, histObj2


def point_overlap(pt, obj):
    """
    Return a Boolean that determines whether a point is overlapping on an object.

    :param pt: pt -> (x,y)
    :param obj: 2D list (list of lists) of boolean values.
    :return: True if (x,y) of pt is True in the list of the obj.
    """
    if 0 <= pt[0] < len(obj) and 0 <= pt[1] < len(obj[0]):
        return obj[pt[0]][pt[1]]
    return False


def image_processing(imagename, background, step, force_type):
    """
    Compute from an image, the RLM of the first and the second object and forces histogram.
    :param imagename: name of the file of the image.
    :param background: background color of the image.
    :param step: step of the angle needed to compute half-lines and diameters for the RLM and F-histogram.
    :param force_type: type of force to use for the computation (usually 0 or 2).
    :return: 3 histograms of the same size: RLM1, RLM2, F-histogram.
    """
    objects = image_segmentation(imagename, background)
    x, y = center_point(objects)
    lines, diameters = lines_diameters(objects, x, y, step * math.pi / 180)
    rlm1, rlm2 = radial_line_model(lines, objects)
    force = forces(objects, diameters, force_type)
    return rlm1, rlm2, force


def SpatialSense_learning(folder, annots, step_deg, force_type):
    """
    Uses annotations from the SpatialSense dataset to segment objects to save histograms value in a json.

    :param folder: folder where SpatialSense images are saved.
    :param annots: path to the annotation file (.json)
    :param step_deg: step of the angle to use for computation of the histograms. In Degree.
    :param force_type: type of force to use.
    :return: list of all Histograms of step_deg*3 (RLM1, RLM2, F-Histogram) value, list of predicates of the images.
    """
    data = []
    number = 0
    forces_data = []
    relation = []
    step_rad = step_deg * math.pi / 180
    for ant in annots:
        if 'flickr' in ant["url"]:
            path = folder + "flickr/"
        else:
            path = folder + "nyu/"
        if '/' in path:
            path += ant["url"].split("/")[-1]
        size_x = ant["width"]
        size_y = ant["height"]
        for rel in ant["annotations"]:
            predicate = rel["predicate"]
            if rel["label"] and predicate in ["to the left of", "to the right of", "above", "under"]:
                # print(rel)
                sub = rel["subject"]
                obj = rel["object"]
                obj_img = [[bbox_value(obj, x, y) for y in range(size_x)] for x in range(size_y)]
                sub_img = [[bbox_value(sub, x, y) for y in range(size_x)] for x in range(size_y)]
                objs = [sub_img, obj_img]
                x, y = center_point(objs)
                lines, diameters = lines_diameters(objs, x, y, step_rad)
                rlm1, rlm2 = radial_line_model(lines, objs)
                force = forces(objs, diameters, force_type)
                data.append({
                    'filename':path,
                    "rel": predicate,
                    "sub": sub["name"],
                    "obj": obj["name"],
                    'forces': force + rlm1 + rlm2
                })
                number += 1
                forces_data.append(force + rlm1 + rlm2)
                relation.append(predicate)
    print(number)
    with open("output/SpatialSense_data.json", "w") as f:
        json.dump(data, f, indent=2)
    return forces, relation


def bbox_value(obj, x, y):
    bbox = obj["bbox"]
    x_min = bbox[0]
    x_max = bbox[1]
    y_min = bbox[2]
    y_max = bbox[3]
    return y_min <= y <= y_max and x_min <= x <= x_max