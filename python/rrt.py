import math
import random
import sys

import numpy as np
import numpy.linalg
from matplotlib import cm
from matplotlib import pyplot as ppl
from scipy.ndimage import imread


def rrt(img, start, goal, ax=None, step_size=20):
    MIN_NUM_VERT = 5  # Minimum number of vertex in the graph
    MAX_NUM_VERT = 20  # Maximum number of vertex in the graph

    if not img[start[1]][start[0]] or not img[goal[1]][goal[0]]:
        return []

    # hundreds = 100
    points = []
    graph = []
    points.append(start)
    graph.append((start, []))
    # print 'Generating and conecting random points'
    occupied = True
    phase_two = False
    # Phase two values (points 5 step distances around the goal point)
    min_x = goal[0] - 5 * step_size if goal[0] - 5 * step_size > 0 else 0
    max_x = goal[0] + 5 * step_size if goal[0] + 5 * step_size < len(img[0]) - 1 else len(img[0]) - 1
    min_y = goal[1] - 5 * step_size if goal[1] - 5 * step_size > 0 else 0
    max_y = goal[1] + 5 * step_size if goal[1] + 5 * step_size < len(img) - 1 else len(img) - 1

    i = 0
    while (goal not in points) and (len(points) < MAX_NUM_VERT):
        # if i % 100 == 0:
        #     print i, 'points randomly generated'
        # if len(points) % hundreds == 0:
        #     print len(points), 'vertex generated'
        #     hundreds += hundreds
        trial_count = 0
        while occupied:
            trial_count += 1
            if trial_count > 100:
                break
            if phase_two and random.random() > 0.8:
                point = [random.randint(min_x, max_x), random.randint(min_y, max_y)]
            else:
                point = [random.randint(0, len(img[0]) - 1), random.randint(0, len(img) - 1)]
            if img[point[1]][point[0]]:
                occupied = False

        occupied = True

        nearest = find_nearest_point(points, point)
        if not nearest:
            if i > MAX_NUM_VERT:
                return []
            else:
                continue
        point = np.array(point) - np.array(nearest)
        point = point * step_size / np.linalg.norm(point)
        point = (point + np.array(nearest)).tolist()

        new_points = connect_points(point, nearest, step_size, img)
        add_to_graph(ax, graph, new_points, point)
        new_points.pop(0)  # The first element is already in the points list
        points.extend(new_points)
        ppl.draw()
        i += 1

        dist = math.sqrt((goal[0] - point[0]) ** 2 + (goal[1] - point[1]) ** 2)
        if dist < step_size or len(points) >= MIN_NUM_VERT:
            nearest = find_nearest_point(points, goal)
            new_points = connect_points(goal, nearest, step_size, img)
            add_to_graph(ax, graph, new_points, goal)
            new_points.pop(0)
            points.extend(new_points)
            ppl.draw()

        # if len(points) >= MIN_NUM_VERT:
        #     phase_two = True
        # if phase_two:
        #     nearest = find_nearest_point(points, goal)
        #     new_points = connect_points(goal, nearest, step_size, img)
        #     add_to_graph(ax, graph, new_points, goal)
        #     new_points.pop(0)
        #     points.extend(new_points)
        #     ppl.draw()

    if goal in points:
        # print 'Goal found, total vertex in graph:', len(points), 'total random points generated:', i
        path = search_path(graph, start, [start])

        if ax:
            for i in range(len(path) - 1):
                ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='g', linestyle='-', linewidth=2)
                ppl.draw()
            # print 'Showing resulting map'
            # print 'Final path:', path
            # print 'The final path is made from:', len(path), 'connected points'
            ppl.show()
    else:
        path = []
        # print 'Reached maximum number of vertex and goal was not found'
        # print 'Total vertex in graph:', len(points), 'total random points generated:', i
        # print 'Showing resulting map'

    return path


def search_path(graph, point, path):
    for i in graph:
        if point == i[0]:
            p = i

    if p[0] == graph[-1][0]:
        return path

    for link in p[1]:
        path.append(link)
        final_path = search_path(graph, link, path)
        if final_path:
            return final_path
        else:
            path.pop()


def add_to_graph(ax, graph, new_points, point):
    if len(new_points) > 1:  # If there is anything to add to the graph
        for p in range(len(new_points) - 1):
            nearest = [nearest for nearest in graph if nearest[0] == [new_points[p][0], new_points[p][1]]]
            nearest[0][1].append(new_points[p + 1])
            graph.append((new_points[p + 1], []))

            if ax:
                if not p == 0:
                    ax.plot(new_points[p][0], new_points[p][1], '+k')  # First point is already painted
                ax.plot([new_points[p][0], new_points[p + 1][0]], [new_points[p][1], new_points[p + 1][1]], color='k',
                        linestyle='-', linewidth=1)
                if point in new_points:
                    ax.plot(point[0], point[1], '.g')  # Last point is green
                else:
                    ax.plot(new_points[p + 1][0], new_points[p + 1][1], '+k')  # Last point is not green


def connect_points(a, b, step_size, img):
    new_points = list()
    new_points.append([b[0], b[1]])
    step = [(a[0] - b[0]) / float(step_size), (a[1] - b[1]) / float(step_size)]
    # step[0] = step[0] if step[0] != 0 else 0.01
    # step[1] = step[1] if step[1] != 0 else 0.01

    # Set small steps to check for walls
    points_needed = int(math.floor(max(math.fabs(step[0]), math.fabs(step[1]))))
    if math.fabs(step[0]) > math.fabs(step[1]):
        if step[0] >= 0:
            step = [1, step[1] / math.fabs(step[0])]
        else:
            step = [-1, step[1] / math.fabs(step[0])]
    else:
        if step[1] >= 0:
            step = [step[0] / math.fabs(step[1]), 1]
        else:
            step = [step[0] / math.fabs(step[1]), -1]

    blocked = False
    for i in range(points_needed + 1):  # Creates points between graph and solitary point
        for j in range(step_size):  # Check if there are walls between points
            coord_x = int(new_points[i][0] + step[0] * j)
            coord_y = int(new_points[i][1] + step[1] * j)
            if coord_x == a[0] and coord_y == a[1]:
                break
            if coord_y >= len(img) or coord_x >= len(img[0]):
                break
            if not img[coord_y][coord_x]:
                blocked = True
            if blocked:
                break
        if blocked:
            break
        if not (coord_x == a[0] and coord_y == a[1]):
            new_points.append([new_points[i][0] + (step[0] * step_size), new_points[i][1] + (step[1] * step_size)])

    if not blocked:
        new_points.append([a[0], a[1]])
    return new_points


def find_nearest_point(points, point):
    best_found = False

    best = (sys.maxint, sys.maxint, sys.maxint)
    for p in points:
        if p == point:
            continue
        dist = math.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)
        if dist < best[2]:
            best = (p[0], p[1], dist)
            best_found = True

    if best_found:
        return best[0], best[1]
    else:
        return None


def select_start_goal_points(ax, img):
    print 'Select a starting point'
    ax.set_xlabel('Select a starting point')
    occupied = True
    while occupied:
        point = ppl.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2)
        start = [int(point[0][0]), int(point[0][1])]
        if img[start[1]][start[0]]:
            print 'Starting point:', start
            occupied = False
            ax.plot(start[0], start[1], '.r')
        else:
            print 'Cannot place a starting point there'
            ax.set_xlabel('Cannot place a starting point there, choose another point')

    print 'Select a goal point'
    ax.set_xlabel('Select a goal point')
    occupied = True
    while occupied:
        point = ppl.ginput(1, timeout=-1, show_clicks=False, mouse_pop=2)
        goal = [int(point[0][0]), int(point[0][1])]
        if img[goal[1]][goal[0]]:
            print 'Goal point:', goal
            occupied = False
            ax.plot(goal[0], goal[1], '.b')
        else:
            print 'Cannot place a goal point there'
            ax.set_xlabel('Cannot place a goal point there, choose another point')

    ppl.draw()
    return start, goal


def plan_trajectory(img, start, goal):
    path = rrt(img, start, goal)
    return path


def plan_trajectory_with_ui(img):
    fig = ppl.gcf()
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap=cm.Greys_r)
    ax.axis('image')
    ppl.draw()
    print 'Map is', len(img[0]), 'x', len(img)
    start, goal = select_start_goal_points(ax, img)
    path = rrt(img, start, goal, ax)
    return path


def main():
    pass


if __name__ == '__main__':
    main()
