import heapq as hp

class Node:
    def __init__(self, current_state, current_level, cost, states):
        self.current_state = current_state
        self.current_level = current_level
        self.cost = cost
        self.states = states

    def __lt__(self, other):
        return self.cost < other.cost


def find_blank(state):
    """Find blank position in 3x3 puzzle."""
    for i in range(3):
        for j in range(3):
            if state[i][j] == ' ':
                return i, j


def children(state):
    """Generate possible next states from current state."""
    x, y = find_blank(state)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    new_states = []

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]  # deep copy
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            new_states.append(new_state)

    return new_states


def manhattan_distance(state, goal):
    """Heuristic: Manhattan distance between current and goal states."""
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != ' ':
                for x in range(3):
                    for y in range(3):
                        if goal[x][y] == val:
                            dist += abs(i - x) + abs(j - y)
    return dist


def puzzle_solve():
    initial_state = [
        [2, 8, 3],
        [1, 6, 4],
        [7, ' ', 5]
    ]

    goal_state = [
        [1, 2, 3],
        [8, ' ', 4],
        [7, 6, 5]
    ]

    pq = []
    start_cost = manhattan_distance(initial_state, goal_state)
    start_node = Node(initial_state, 0, start_cost, [initial_state])
    hp.heappush(pq, (start_node.cost, start_node))

    visited_state = set()

    while pq:
        _, node = hp.heappop(pq)

        if node.current_state == goal_state:
            print("Solved in", node.current_level, "steps!")
            for step in node.states:
                for row in step:
                    print(row)
                print()
            return

        state_key = tuple(tuple(row) for row in node.current_state)
        if state_key in visited_state:
            continue

        visited_state.add(state_key)

        for n_state in children(node.current_state):
            g = node.current_level + 1
            h = manhattan_distance(n_state, goal_state)
            new_node = Node(n_state, g, g + h, node.states + [n_state])
            hp.heappush(pq, (new_node.cost, new_node))


puzzle_solve()
