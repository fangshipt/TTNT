import heapq
from collections import deque
import random
import math
import itertools
import sys
sys.setrecursionlimit(10000)

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None
MOVES = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

def generate_states(state):
    blank_x, blank_y = find_blank(state)
    possible_moves = []
    for move, (dx, dy) in MOVES.items():
        new_x, new_y = blank_x + dx, blank_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = [row[:] for row in state]
            new_state[blank_x][blank_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[blank_x][blank_y]
            possible_moves.append((new_state, move))
    return possible_moves

def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                for gi in range(3):
                    for gj in range(3):
                        if goal_state[gi][gj] == val:
                            distance += abs(i - gi) + abs(j - gj)
                            break
    return distance

def count_inversions(state):
    flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    return inversions

def is_solvable(start_state, goal_state):
    return (count_inversions(start_state) % 2) == (count_inversions(goal_state) % 2)

def apply_moves(state, moves_seq):
    current_state = [row[:] for row in state]
    for move in moves_seq:
        blank_x, blank_y = find_blank(current_state)
        dx, dy = MOVES[move]
        new_x, new_y = blank_x + dx, blank_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            current_state[blank_x][blank_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[blank_x][blank_y]
        else:
            return None
    return current_state

# Các thuật toán tìm kiếm (giữ nguyên, chỉ liệt kê tên)
def bfs_solve(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set([tuple(map(tuple, start_state))])
    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state:
            return path
        for new_state, move in generate_states(current_state):
            state_tuple = tuple(map(tuple, new_state))
            if state_tuple not in visited:
                visited.add(state_tuple)
                queue.append((new_state, path + [move]))
    return None

def astar_solve(start_state, goal_state):
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    h = manhattan_distance(start_state, goal_state)
    g = 0
    f = g + h
    heapq.heappush(pq, (f, g, start_state, []))
    visited = {initial_tuple: f}
    while pq:
        f, g, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for new_state, move in generate_states(state):
            new_g = g + 1
            new_h = manhattan_distance(new_state, goal_state)
            new_f = new_g + new_h
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited or new_f < visited[new_tuple]:
                visited[new_tuple] = new_f
                heapq.heappush(pq, (new_f, new_g, new_state, path + [move]))
    return None

def dfs_solve(start_state, goal_state, max_depth=50):
    solution = None
    visited = set()

    def state_key(state):
        return tuple(map(tuple, state))

    def dfs(state, path, depth):
        nonlocal solution
        if depth < 0:
            return False
        key = state_key(state)
        if key in visited:
            return False
        visited.add(key)
        if state == goal_state:
            solution = path
            return True
        for new_state, move in generate_states(state):
            if dfs(new_state, path + [move], depth - 1):
                return True
        visited.remove(key)  
        return False

    dfs(start_state, [], max_depth)
    return solution


def ids_solve(start_state, goal_state, max_depth=50):
    solution = None
    def dls(state, path, depth, visited):
        nonlocal solution
        if state == goal_state:
            solution = path
            return True
        if depth == 0:
            return False
        visited.add(tuple(map(tuple, state)))
        for new_state, move in generate_states(state):
            if tuple(map(tuple, new_state)) not in visited:
                if dls(new_state, path + [move], depth - 1, visited):
                    return True
        visited.remove(tuple(map(tuple, state)))
        return False
    for depth in range(max_depth + 1):
        visited = set()
        if dls(start_state, [], depth, visited):
            return solution
    return None

def ida_star_solve(start_state, goal_state):
    solution = None
    bound = manhattan_distance(start_state, goal_state)
    def search(g, bound, path, moves):
        nonlocal solution
        state = path[-1]
        f = g + manhattan_distance(state, goal_state)
        if f > bound:
            return f
        if state == goal_state:
            solution = moves
            return -1
        min_threshold = float('inf')
        for new_state, move in generate_states(state):
            if any(tuple(map(tuple, s)) == tuple(map(tuple, new_state)) for s in path):
                continue
            path.append(new_state)
            t = search(g + 1, bound, path, moves + [move])
            if t == -1:
                return -1
            if t < min_threshold:
                min_threshold = t
            path.pop()
        return min_threshold

    path = [start_state]
    while True:
        t = search(0, bound, path, [])
        if t == -1:
            return solution
        if t == float('inf'):
            return None
        bound = t

def simple_hill_climbing_solve(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    current_h = manhattan_distance(current_state, goal_state)
    path = []
    if current_state == goal_state:
        return path
    while True:
        found_better = False
        for new_state, move in generate_states(current_state):
            new_h = manhattan_distance(new_state, goal_state)
            if new_h < current_h:
                current_state = new_state
                current_h = new_h
                path.append(move)
                found_better = True
                break
        if not found_better:
            return path if current_state == goal_state else None

def steepest_hill_climbing_solve(start_state, goal_state):
    current_state = [row[:] for row in start_state]
    current_h = manhattan_distance(current_state, goal_state)
    path = []
    if current_state == goal_state:
        return path
    while True:
        best_neighbor = None
        best_move = None
        best_h = current_h
        for new_state, move in generate_states(current_state):
            new_h = manhattan_distance(new_state, goal_state)
            if new_h < best_h:
                best_h = new_h
                best_neighbor = new_state
                best_move = move
        if best_neighbor is None:
            return path if current_state == goal_state else None
        current_state = best_neighbor
        current_h = best_h
        path.append(best_move)

def stochastic_hill_climbing_solve(start_state, goal_state, max_restarts=50, max_steps=1000):
    for r in range(max_restarts):
        current_state = [row[:] for row in start_state]
        current_h = manhattan_distance(current_state, goal_state)
        path = []
        steps = 0
        while steps < max_steps:
            if current_state == goal_state:
                return path
            better_neighbors = []
            for new_state, move in generate_states(current_state):
                new_h = manhattan_distance(new_state, goal_state)
                if new_h < current_h:
                    better_neighbors.append((new_state, move, new_h))
            if better_neighbors:
                new_state, move, new_h = random.choice(better_neighbors)
                current_state = new_state
                current_h = new_h
                path.append(move)
                steps += 1
            else:
                break
    return None

def simulated_annealing_solve(start_state, goal_state, T_init=1e5, cooling_rate=0.95, max_steps=10000):
    current_state = [row[:] for row in start_state]
    current_cost = manhattan_distance(current_state, goal_state)
    path = []
    T = T_init
    steps = 0
    while T > 1 and steps < max_steps:
        if current_state == goal_state:
            return path
        neighbors = generate_states(current_state)
        new_state, move = random.choice(neighbors)
        new_cost = manhattan_distance(new_state, goal_state)
        delta = new_cost - current_cost
        if delta < 0:
            current_state = new_state
            current_cost = new_cost
            path.append(move)
        else:
            probability = math.exp(-delta / T)
            if random.random() < probability:
                current_state = new_state
                current_cost = new_cost
                path.append(move)
        T *= cooling_rate
        steps += 1
    return path if current_state == goal_state else None

def greedy_best_first_search(start_state, goal_state):
    pq = []
    h = manhattan_distance(start_state, goal_state)
    heapq.heappush(pq, (h, start_state, []))
    visited = set()
    visited.add(tuple(map(tuple, start_state)))

    while pq:
        h, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for new_state, move in generate_states(state):
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_h = manhattan_distance(new_state, goal_state)
                heapq.heappush(pq, (new_h, new_state, path + [move]))
    return None


def beam_search_solve(start_state, goal_state, beam_width=3):
    beam = [(start_state, [])]
    visited = set()
    visited.add(tuple(map(tuple, start_state)))
    while beam:
        new_beam = []
        for state, path in beam:
            if state == goal_state:
                return path
            for new_state, move in generate_states(state):
                state_tuple = tuple(map(tuple, new_state))
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    new_beam.append((new_state, path + [move]))
        if not new_beam:
            return None
        new_beam.sort(key=lambda x: manhattan_distance(x[0], goal_state))
        beam = new_beam[:beam_width]
    return None

def genetic_algorithm_solve(start_state, goal_state, population_size=200, max_generations=1000, mutation_rate=0.05):
    moves_list = list(MOVES.keys())

    def fitness(moves_seq):
        new_state = apply_moves(start_state, moves_seq)
        if new_state is None:
            return float('inf')  
        manhattan = manhattan_distance(new_state, goal_state)
        length_penalty = len(moves_seq) * 0.1  
        return manhattan + length_penalty

    def smart_initialize():
        current_state = [row[:] for row in start_state]
        chromosome = []
        for _ in range(random.randint(5, 40)):  # Độ dài ngẫu nhiên từ 5 đến 40
            valid_moves = [move for _, move in generate_states(current_state)]
            if not valid_moves:
                break
            best_move = min(valid_moves, key=lambda m: manhattan_distance(
                apply_moves(current_state, [m]), goal_state) if apply_moves(current_state, [m]) else float('inf'))
            chromosome.append(best_move)
            current_state = apply_moves(current_state, [best_move])
            if current_state is None:
                break
        return chromosome

    population = [smart_initialize() for _ in range(population_size)]

    for generation in range(max_generations):
        fitness_values = [fitness(chromosome) for chromosome in population]
        min_fitness = min(fitness_values)
        if min_fitness < 0.1:
            best_idx = fitness_values.index(min_fitness)
            best_chromosome = population[best_idx]
            final_state = apply_moves(start_state, best_chromosome)
            if final_state and manhattan_distance(final_state, goal_state) == 0:
                return best_chromosome

        new_population = []
        for _ in range(population_size):
            tournament = random.sample(population, max(3, population_size // 10))
            best = min(tournament, key=fitness)
            new_population.append(best.copy())

        for i in range(0, population_size, 2):
            if i + 1 < population_size and random.random() < 0.8:
                parent1, parent2 = new_population[i], new_population[i + 1]
                if len(parent1) > 2 and len(parent2) > 2:
                    point1 = random.randint(1, min(len(parent1), len(parent2)) - 1)
                    point2 = random.randint(point1, min(len(parent1), len(parent2)))
                    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
                    new_population[i], new_population[i + 1] = child1, child2

        for chromosome in new_population:
            if random.random() < mutation_rate:
                if random.random() < 0.5 and len(chromosome) > 1:
                    idx = random.randint(0, len(chromosome) - 1)
                    chromosome[idx] = random.choice(moves_list)
                else:
                    if random.random() < 0.5 and len(chromosome) < 40:
                        idx = random.randint(0, len(chromosome))
                        chromosome.insert(idx, random.choice(moves_list))
                    elif len(chromosome) > 5:
                        idx = random.randint(0, len(chromosome) - 1)
                        del chromosome[idx]

        population = new_population

    fitness_values = [fitness(chromosome) for chromosome in population]
    best_idx = fitness_values.index(min(fitness_values))
    return population[best_idx]




def ucs_solve(start_state, goal_state):
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    heapq.heappush(pq, (0, start_state, []))
    visited = {initial_tuple: 0}
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for new_state, move in generate_states(state):
            new_cost = cost + 1
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited or new_cost < visited[new_tuple]:
                visited[new_tuple] = new_cost
                heapq.heappush(pq, (new_cost, new_state, path + [move]))
    return None

def greedy_solve(start_state, goal_state):
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    h = manhattan_distance(start_state, goal_state)
    heapq.heappush(pq, (h, start_state, []))
    visited = {initial_tuple}
    while pq:
        h, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for new_state, move in generate_states(state):
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_h = manhattan_distance(new_state, goal_state)
                heapq.heappush(pq, (new_h, new_state, path + [move]))
    return None



def generate_all_possible_states():
    all_states = []
    for perm in itertools.permutations(range(9)):
        mat = [list(perm[0:3]), list(perm[3:6]), list(perm[6:9])]
        all_states.append(mat)
    return all_states

def backtracking_solve(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None 

    visited = set()
    stack = [(start_state, [], 0)]  
    max_depth = 50  

    while stack:
        state, path, depth = stack.pop()
        state_tuple = tuple(map(tuple, state))
        goal_tuple = tuple(map(tuple, goal_state))

        if state_tuple == goal_tuple:
            return path

        if depth > max_depth:
            continue

        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        next_states = generate_states(state)
        next_states.sort(key=lambda x: manhattan_distance(x[0], goal_state))

        for new_state, move in reversed(next_states):
            new_state_tuple = tuple(map(tuple, new_state))
            if new_state_tuple not in visited:
                stack.append((new_state, path + [move], depth + 1))

    return None

import random

def q_learning_solve(start_state, goal_state,
                     episodes=5000,
                     max_steps_per_episode=200,
                     alpha=0.1,      
                     gamma=0.9,     
                     epsilon=0.1): 
    def state_key(state):
        return tuple(tuple(row) for row in state)

    actions = list(MOVES.keys())
    Q = {}
    def get_q(s_key, a):
        return Q.setdefault(s_key, {act: 0.0 for act in actions})[a]

    def choose_action(state):
        s_key = state_key(state)
        if random.random() < epsilon:
            return random.choice(actions)
        q_vals = Q.setdefault(s_key, {act: 0.0 for act in actions})
        max_q = max(q_vals.values())
        best = [a for a,v in q_vals.items() if v == max_q]
        return random.choice(best)
    def reward(new_state):
        return 100.0 if new_state == goal_state else -1.0

    for ep in range(episodes):
        state = [row[:] for row in start_state]
        for step in range(max_steps_per_episode):
            s_key = state_key(state)
            action = choose_action(state)
            blank_x, blank_y = find_blank(state)
            dx, dy = MOVES[action]
            nx, ny = blank_x + dx, blank_y + dy
            if not (0 <= nx < 3 and 0 <= ny < 3):
                r = -5.0
                next_state = state
            else:
                next_state = [row[:] for row in state]
                next_state[blank_x][blank_y], next_state[nx][ny] = next_state[nx][ny], next_state[blank_x][blank_y]
                r = reward(next_state)

            next_key = state_key(next_state)
            max_q_next = max(Q.setdefault(next_key, {act:0.0 for act in actions}).values())
            old_q = get_q(s_key, action)
            new_q = old_q + alpha * (r + gamma * max_q_next - old_q)
            Q[s_key][action] = new_q

            state = next_state

            if state == goal_state:
                break
    path = []
    state = [row[:] for row in start_state]
    visited = set([state_key(state)])
    for _ in range(1000):
        if state == goal_state:
            return path
        s_key = state_key(state)
        q_vals = Q.get(s_key, None)
        if not q_vals:
            break
        best_act = max(q_vals, key=q_vals.get)
        blank_x, blank_y = find_blank(state)
        dx, dy = MOVES[best_act]
        nx, ny = blank_x + dx, blank_y + dy
        if not (0 <= nx < 3 and 0 <= ny < 3):
            break
        state = [row[:] for row in state]
        state[blank_x][blank_y], state[nx][ny] = state[nx][ny], state[blank_x][blank_y]
        if state_key(state) in visited:
            break
        visited.add(state_key(state))
        path.append(best_act)
    return None



