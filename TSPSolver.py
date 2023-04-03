#!/usr/bin/python3
import time

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    pass
elif PYQT_VER == 'PYQT4':
    pass
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq


def reduce_matrix(matrix_to_reduce):
    # Subtract the smallest value in each row from all elements in that row
    matrix = np.copy(matrix_to_reduce)
    total_reduction = 0

    for row in matrix:
        min_value = min(row)
        if min_value != np.inf:
            total_reduction += min_value
        for i in range(len(row)):
            if row[i] != np.inf:
                row[i] -= min_value

    for i in range(matrix.shape[1]):
        col = matrix[:, i]
        min_value = min(col)
        if min_value != np.inf:
            total_reduction += min_value
        for j in range(len(col)):
            if col[j] != np.inf:
                col[j] -= min_value

    if total_reduction == np.inf:
        total_reduction = 0

    return matrix, total_reduction


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution,
        time spent to find solution, number of permutations tried during search, the
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results = {'cost': bssf.cost if foundTour else math.inf, 'time': end_time - start_time, 'count': count,
                   'soln': bssf, 'max': None, 'total': None, 'pruned': None}
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this
        algorithm</returns>
    '''

    def greedy(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        distances = cities_to_np_array(cities)
        n = distances.shape[0]

        best_cost = np.inf
        best_path = None

        start_time = time.time()
        for start_city in range(n):
            path = [start_city]
            total_cost = 0

            while len(path) < n:
                if time.time() - start_time > time_allowance:
                    break
                current_city = path[-1]
                next_city = None
                min_cost = np.inf

                for city in range(n):
                    if city not in path:
                        cost = distances[current_city, city]
                        if cost < min_cost:
                            min_cost = cost
                            next_city = city

                if next_city is None:
                    # if there is no path to any unvisited city from the current city,
                    # start again from a random unvisited city
                    unvisited_cities = set(range(n)) - set(path)
                    start_city = random.choice(list(unvisited_cities))
                    path.append(start_city)
                    continue

                path.append(next_city)
                total_cost += min_cost

            if len(path) == n:
                # a complete path was found
                total_cost += distances[path[-1], start_city]

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = path

        end_time = time.time()

        if best_path is not None:
            # convert path to city list
            city_path = [cities[i] for i in best_path]
            bssf = TSPSolution(city_path)
            results = {'cost': best_cost, 'time': end_time - start_time, 'count': 1,
                       'soln': bssf, 'max': None, 'total': None, 'pruned': None}
        else:
            results = {'cost': np.inf, 'time': end_time - start_time, 'count': 1,
                       'soln': None, 'max': None, 'total': None, 'pruned': None}

        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints:
        max queue size, total number of states created, and number of pruned states.</returns>
    '''

    def branchAndBound(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)

        # our greedy algorithm's solution cost. This isn't 100% the ideal solution, but it gives us a valid solution and thus a decent upper bound
        greedy_results = self.greedy()
        bssf = greedy_results['cost']

        print(f"total_cost greedy: {bssf}")

        # convert our list of cities to a cost matrix
        matrix = cities_to_np_array(cities)

        # reduce the cost matrix
        matrix, reduction_value = reduce_matrix(matrix)

        # create our heapq
        queue = []

        # our starting node is our starting city, or the first city on the list
        start_node = Node(matrix, reduction_value, 0)

        # keep track of the path that we're taking
        start_node.add_to_path(0)

        # push the starting node onto the heapq
        heapq.heappush(queue, start_node)

        # init some variables for later
        total_pruned = 0
        best_node_so_far = None
        num_solutions = 0
        nodes_created = 0
        max_stored = 0

        start_time = time.time()
        while queue:
            # get the current best node
            parent_node = heapq.heappop(queue)

            # if this node's cost is > bssf, discard it
            if parent_node.cost > bssf:
                total_pruned += 1
                continue

            parent_matrix = parent_node.matrix

            # our row is our current city, and the values in that row are the costs associated with traveling to each other city
            row = parent_matrix[parent_node.parent_index]

            # for each element in the row create a node and set its values
            for row_index, distance_to_city in enumerate(row):
                # if there is no path from the current node to the next node then skip the node
                if distance_to_city == np.inf:
                    continue

                if len(queue) > max_stored:
                    max_stored = len(queue)

                # we need to create a matrix for the child node
                # make a copy of the parent's matrix
                child_matrix = np.copy(parent_node.matrix)

                # set the row and col to infinity, the row is the value of the parent's node_value, and the col is the value of the node
                child_matrix[parent_node.parent_index, :] = np.inf
                child_matrix[:, row_index] = np.inf

                # set the path just traveled to infinity
                child_matrix[row_index, parent_node.parent_index] = np.inf

                # reduce the matrix if possible
                reduced_matrix, reduction_value = reduce_matrix(child_matrix)
                # create the child node
                nodes_created += 1

                child_node = Node(reduced_matrix, parent_node.cost + distance_to_city + reduction_value,
                                  row_index)  # set node_value to row_index
                # add the path to this node
                child_node.add_path_to_path(parent_node.path)
                child_node.add_to_path(row_index)

                # check if we're at the end of a branch/possible solution
                if len(child_node.path) == ncities:  # found valid solution
                    # if the leaf node's total cost is <= to our current best solution then remember it
                    if child_node.cost <= bssf:
                        bssf = child_node.cost
                        best_node_so_far = child_node
                        num_solutions += 1
                        print(f"updating bssf to: {bssf}")
                # if we're not a leaf node then add ourselves to the q, but only if our current cost is better than bssf
                elif child_node.cost <= bssf:
                    heapq.heappush(queue, child_node)

            # # exit our while loop if we're out of time
            if time.time() - start_time > time_allowance:
                queue = [] # exit our loop
                if best_node_so_far is None:
                    total_pruned += sum(1 for x in queue if x.cost > bssf)
                    return {'cost': greedy_results['cost'], 'time': time_allowance, 'count': 0,
                               'soln': greedy_results['soln'], 'max': max_stored, 'total': nodes_created, 'pruned': total_pruned}

        end_time = time.time()
        # print(f"bssf: {bssf}")
        print(f"solution path: {best_node_so_far.path}")

        solution_path = []
        for i in best_node_so_far.path:
            solution_path.append(cities[i])

        bs = TSPSolution(solution_path)
        return {'cost': bssf, 'time': end_time - start_time, 'count': num_solutions,
                   'soln': bs, 'max': max_stored, 'total': nodes_created, 'pruned': total_pruned}

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
    '''

    def fancy(self, time_allowance=60.0):
        pass


def cities_to_np_array(list_of_cities):
    num_cities = len(list_of_cities)

    matrix = np.zeros((num_cities, num_cities))
    for row_index in range(num_cities):
        for col_index in range(num_cities):
            matrix[row_index][col_index] = list_of_cities[row_index].costTo(list_of_cities[col_index])

    return matrix


class Node:
    def __init__(self, matrix=None, cost=0, parent_index=0):
        self.cost = cost
        self.matrix = matrix
        self.parent_index = parent_index
        self.path = []

    def add_to_cost(self, val):
        self.cost += val

    def add_to_path(self, i):
        self.path.append(i)

    def add_path_to_path(self, path):
        for item in path:
            self.path.append(item)

    def __lt__(self, other):
        # the below line is roughly 2x slower even though it encourages pursuing deeper nodes
        # return self.cost / len(self.path) < other.cost
        return self.cost < other.cost
