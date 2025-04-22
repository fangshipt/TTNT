import tkinter as tk
from tkinter import messagebox
import heapq
import threading
from collections import deque
import random
import math
import itertools

# -------------------------------------
# HÀM HỖ TRỢ CHO 8-PUZZLE
# -------------------------------------
def find_blank(state):
    """Tìm vị trí ô trống (số 0) trong bàn cờ."""
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

def apply_moves(state, moves_seq):
    current_state = [row[:] for row in state]
    for move in moves_seq:
        blank_x, blank_y = find_blank(current_state)
        dx, dy = MOVES[move]
        new_x, new_y = blank_x + dx, blank_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            current_state[blank_x][blank_y], current_state[new_x][new_y] = current_state[new_x][new_y], current_state[blank_x][blank_y]
        else:
            return None  # Invalid move
    return current_state

# -------------------------------------
# CÁC THUẬT TOÁN TÌM KIẾM
# -------------------------------------
def bfs_solve(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    visited.add(tuple(map(tuple, start_state)))
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

def dfs_solve(start_state, goal_state):
    solution = None
    visited = set()
    def dfs(state, path):
        nonlocal solution
        if state == goal_state:
            solution = path
            return True
        visited.add(tuple(map(tuple, state)))
        for new_state, move in generate_states(state):
            if tuple(map(tuple, new_state)) not in visited:
                if dfs(new_state, path + [move]):
                    return True
        return False
    dfs(start_state, [])
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

# Thuật toán di truyền cải tiến
def genetic_algorithm_solve(start_state, goal_state, population_size=200, max_generations=1000, mutation_rate=0.05):
    moves_list = list(MOVES.keys())

    def fitness(moves_seq):
        new_state = apply_moves(start_state, moves_seq)
        if new_state is None:
            return float('inf')  # Phạt nặng nếu chuỗi không hợp lệ
        manhattan = manhattan_distance(new_state, goal_state)
        length_penalty = len(moves_seq) * 0.1  # Phạt nhẹ cho chuỗi dài
        return manhattan + length_penalty

    def smart_initialize():
        current_state = [row[:] for row in start_state]
        chromosome = []
        for _ in range(random.randint(5, 40)):  # Độ dài ngẫu nhiên từ 5 đến 40
            valid_moves = [move for _, move in generate_states(current_state)]
            if not valid_moves:
                break
            # Ưu tiên nước đi giảm khoảng cách Manhattan
            best_move = min(valid_moves, key=lambda m: manhattan_distance(
                apply_moves(current_state, [m]), goal_state) if apply_moves(current_state, [m]) else float('inf'))
            chromosome.append(best_move)
            current_state = apply_moves(current_state, [best_move])
            if current_state is None:
                break
        return chromosome

    # Khởi tạo dân số
    population = [smart_initialize() for _ in range(population_size)]

    for generation in range(max_generations):
        # Đánh giá fitness
        fitness_values = [fitness(chromosome) for chromosome in population]
        min_fitness = min(fitness_values)
        if min_fitness < 0.1:  # Gần với trạng thái mục tiêu
            best_idx = fitness_values.index(min_fitness)
            best_chromosome = population[best_idx]
            final_state = apply_moves(start_state, best_chromosome)
            if final_state and manhattan_distance(final_state, goal_state) == 0:
                return best_chromosome

        # Chọn lọc: Tournament selection
        new_population = []
        for _ in range(population_size):
            tournament = random.sample(population, max(3, population_size // 10))
            best = min(tournament, key=fitness)
            new_population.append(best.copy())

        # Lai ghép: Two-point crossover
        for i in range(0, population_size, 2):
            if i + 1 < population_size and random.random() < 0.8:
                parent1, parent2 = new_population[i], new_population[i + 1]
                if len(parent1) > 2 and len(parent2) > 2:
                    point1 = random.randint(1, min(len(parent1), len(parent2)) - 1)
                    point2 = random.randint(point1, min(len(parent1), len(parent2)))
                    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
                    new_population[i], new_population[i + 1] = child1, child2

        # Đột biến
        for chromosome in new_population:
            if random.random() < mutation_rate:
                if random.random() < 0.5 and len(chromosome) > 1:
                    # Thay đổi một nước đi
                    idx = random.randint(0, len(chromosome) - 1)
                    chromosome[idx] = random.choice(moves_list)
                else:
                    # Thêm hoặc xóa nước đi
                    if random.random() < 0.5 and len(chromosome) < 40:
                        idx = random.randint(0, len(chromosome))
                        chromosome.insert(idx, random.choice(moves_list))
                    elif len(chromosome) > 5:
                        idx = random.randint(0, len(chromosome) - 1)
                        del chromosome[idx]

        population = new_population

    # Trả về giải pháp tốt nhất nếu không tìm được chính xác
    fitness_values = [fitness(chromosome) for chromosome in population]
    best_idx = fitness_values.index(min(fitness_values))
    return population[best_idx]

def and_or_solve(start_state, goal_state, limit=100):
    def and_or(state, path, depth):
        if state == goal_state:
            return path
        if depth > limit:
            return None
        state_tuple = tuple(map(tuple, state))
        for move in MOVES.keys():
            new_state = apply_move(state, move)
            if new_state:
                new_state_tuple = tuple(map(tuple, new_state))
                if new_state_tuple not in path_set:
                    path_set.add(new_state_tuple)
                    result = and_or(new_state, path + [move], depth + 1)
                    if result:
                        return result
                    path_set.remove(new_state_tuple)
        return None

    def apply_move(state, move):
        blank_x, blank_y = find_blank(state)
        dx, dy = MOVES[move]
        new_x, new_y = blank_x + dx, blank_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = [row[:] for row in state]
            new_state[blank_x][blank_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[blank_x][blank_y]
            return new_state
        return None

    path_set = set([tuple(map(tuple, start_state))])
    solution = and_or(start_state, [], 0)
    return solution

def sensorless_bfs_solve(goal_state):
    from collections import deque
    all_states = generate_all_possible_states()
    start_belief = frozenset(tuple(map(tuple, s)) for s in all_states)
    queue = deque()
    queue.append((start_belief, []))
    visited = set([start_belief])
    
    while queue:
        current_belief, path = queue.popleft()
        
        if len(current_belief) == 1:
            single_state = list(current_belief)[0]
            if single_state == tuple(map(tuple, goal_state)):
                return path
        
        for action in MOVES.keys():
            next_belief = set()
            for st_tuple in current_belief:
                st_list = [list(row) for row in st_tuple]
                new_st = apply_action_sensorless(st_list, action)
                if new_st is not None:
                    next_belief.add(tuple(map(tuple, new_st)))
            
            if not next_belief:
                continue
            
            next_belief = frozenset(next_belief)
            if next_belief not in visited:
                visited.add(next_belief)
                queue.append((next_belief, path + [action]))
    
    return None

def apply_action_sensorless(state, move):
    blank_x, blank_y = find_blank(state)
    dx, dy = MOVES[move]
    new_x, new_y = blank_x + dx, blank_y + dy
    if 0 <= new_x < 3 and 0 <= new_y < 3:
        new_state = [row[:] for row in state]
        new_state[blank_x][blank_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[blank_x][blank_y]
        return new_state
    else:
        return None

def generate_all_possible_states():
    all_states = []
    for perm in itertools.permutations(range(9)):
        mat = [list(perm[0:3]), list(perm[3:6]), list(perm[6:9])]
        all_states.append(mat)
    return all_states

def sensorless_bfs_solve_wrapper(goal_state):
    return sensorless_bfs_solve(goal_state)

def general_problem_solver(start_state, goal_state):
    def find_differences(state, goal):
        differences = []
        for i in range(3):
            for j in range(3):
                if state[i][j] != goal[i][j]:
                    differences.append((i, j, state[i][j], goal[i][j]))
        return differences

    def apply_operator(state, operator):
        blank_x, blank_y = find_blank(state)
        dx, dy = MOVES[operator]
        new_x, new_y = blank_x + dx, blank_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = [row[:] for row in state]
            new_state[blank_x][blank_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[blank_x][blank_y]
            return new_state
        return None

    def means_ends_analysis(state, goal, path):
        if state == goal:
            return path
        differences = find_differences(state, goal)
        if not differences:
            return path
        diff = differences[0]
        for move in MOVES.keys():
            new_state = apply_operator(state, move)
            if new_state and new_state not in path:
                new_path = means_ends_analysis(new_state, goal, path + [move])
                if new_path:
                    return new_path
        return None

    return means_ends_analysis(start_state, goal_state, [])

# -------------------------------------
# GIAO DIỆN CHÍNH (Tkinter)
# -------------------------------------
class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("1000x800")
        self.root.configure(bg='#FFF8F8') 

        self.initial_state = None
        self.goal_state = None
        self.state = None
        self.solution = None

        self.main_frame = tk.Frame(self.root, bg='#FFF8F8')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame, bg='#FFF8F8', bd=2, relief=tk.RIDGE)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.center_frame = tk.Frame(self.main_frame, bg='#FFF8F8', bd=2, relief=tk.RIDGE)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.right_frame = tk.Frame(self.main_frame, bg='#FFF8F8', bd=2, relief=tk.RIDGE)
        self.right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.create_left_panel()
        self.create_center_panel()
        self.create_right_panel()

    def create_left_panel(self):
        instruction_label = tk.Label(self.left_frame, text="Input states using 0-8 (0 is blank)", bg='#FFF8F8', font=('Arial', 10, 'italic'))
        instruction_label.pack(pady=5)

        init_frame = tk.LabelFrame(self.left_frame, text="Initial State", bg='#FFF8F8', font=('Arial', 12, 'bold'), fg='black')
        init_frame.pack(padx=5, pady=5)
        self.init_entries = [[None]*3 for _ in range(3)]
        init_defaults = [['2','6','5'],['8','0','7'],['4','3','1']]
        for i in range(3):
            for j in range(3):
                e = tk.Entry(init_frame, width=2, font=('Arial', 20), justify='center', bg='white', fg='black')
                e.grid(row=i, column=j, padx=10, pady=10)
                e.insert(0, init_defaults[i][j])
                self.init_entries[i][j] = e

        goal_frame = tk.LabelFrame(self.left_frame, text="Goal State", bg='#FFF8F8', font=('Arial', 12, 'bold'), fg='black')
        goal_frame.pack(padx=5, pady=5)
        self.goal_entries = [[None]*3 for _ in range(3)]
        goal_defaults = [['1','2','3'],['4','5','6'],['7','8','0']]
        for i in range(3):
            for j in range(3):
                e = tk.Entry(goal_frame, width=2, font=('Arial', 20), justify='center', bg='white', fg='black')
                e.grid(row=i, column=j, padx=10, pady=10)
                e.insert(0, goal_defaults[i][j])
                self.goal_entries[i][j] = e

        update_button = tk.Button(self.left_frame, text="Update States", font=('Arial', 12, 'bold'), bg='#FFB6C1', fg='black', command=self.update_states)
        update_button.pack(padx=5, pady=5, fill=tk.X)

        algo_frame = tk.LabelFrame(self.left_frame, text="Algorithm Selection", bg='#FFF8F8', font=('Arial', 12, 'bold'), fg='black')
        algo_frame.pack(padx=5, pady=5, fill=tk.X)
        self.algo_var = tk.StringVar()
        self.algo_var.set("BFS")
        algo_options = ["BFS", "UCS", "Greedy", "A*", "DFS", "IDS", "IDA*", 
                        "Simple Hill Climbing", "Steepest Hill Climbing", 
                        "Stochastic Hill Climbing", "Simulated Annealing", "Beam Search",
                        "Genetic Algorithm", "AND-OR Graph Search", "Sensorless BFS",
                        "General Problem Solver"]
        algo_menu = tk.OptionMenu(algo_frame, self.algo_var, *algo_options)
        algo_menu.config(font=('Arial', 12), bg='#FFB6C1', fg='black')
        algo_menu.pack(padx=5, pady=5, fill=tk.X)

        solve_button = tk.Button(self.left_frame, text="Solve", font=('Arial', 12, 'bold'), bg='#FFB6C1', fg='black', command=lambda: self.start_solver_thread(self.algo_var.get()))
        solve_button.pack(padx=5, pady=5, fill=tk.X)

        reset_button = tk.Button(self.left_frame, text="Reset", font=('Arial', 12, 'bold'), bg='#FFB6C1', fg='black', command=self.reset_board)
        reset_button.pack(padx=5, pady=5, fill=tk.X)

    def create_center_panel(self):
        board_label = tk.Label(self.center_frame, text="8-Puzzle Board", bg='#FFF8F8', font=('Arial', 14, 'bold'))
        board_label.pack(pady=5)
        self.board_frame = tk.Frame(self.center_frame, bg='#FFF8F8')
        self.board_frame.pack(pady=5)
        self.buttons = [[None]*3 for _ in range(3)]
        self.create_board()
        self.status_label = tk.Label(self.center_frame, text="", bg='#FFF8F8', font=('Arial', 12))
        self.status_label.pack(pady=5)

    def create_right_panel(self):
        solution_label = tk.Label(self.right_frame, text="Solution Steps:", bg='#FFF8F8', font=('Arial', 12, 'bold'))
        solution_label.pack(pady=5)
        solution_frame = tk.Frame(self.right_frame, bg='#FFF8F8')
        solution_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.solution_text = tk.Text(solution_frame, height=20, width=20, font=('Arial', 12), bg='white', fg='black')
        self.solution_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(solution_frame, command=self.solution_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.solution_text.config(yscrollcommand=scrollbar.set)

    def create_board(self):
        if self.state is None:
            self.state = [[int(e.get()) for e in row] for row in self.init_entries]
            self.initial_state = [row[:] for row in self.state]
        for widget in self.board_frame.winfo_children():
            widget.destroy()
        self.buttons = [[None]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                btn = tk.Button(self.board_frame,
                                text=str(val) if val != 0 else '',
                                font=('Arial', 24, 'bold'),
                                width=4, height=2,
                                bg='#FFB6C1', fg='black')
                btn.grid(row=i, column=j, padx=8, pady=8)
                self.buttons[i][j] = btn

    def update_states(self):
        try:
            new_init = [[int(self.init_entries[i][j].get()) for j in range(3)] for i in range(3)]
            new_goal = [[int(self.goal_entries[i][j].get()) for j in range(3)] for i in range(3)]
        except ValueError:
            messagebox.showerror("Error", "Please enter integers from 0 to 8.")
            return
        
        if sorted(sum(new_init, [])) != list(range(9)) or sorted(sum(new_goal, [])) != list(range(9)):
            messagebox.showerror("Error", "Invalid state. Must contain unique numbers from 0 to 8.")
            return
        
        self.initial_state = [row[:] for row in new_init]
        self.goal_state = [row[:] for row in new_goal]
        self.state = [row[:] for row in new_init]
        self.create_board()
        self.status_label.config(text="States updated.")

    def start_solver_thread(self, algo):
        self.solution = None
        self.status_label.config(text="Solving...")
        self.solution_text.config(state='normal')
        self.solution_text.delete(1.0, tk.END)
        self.solution_text.config(state='disabled')

        if not self.goal_state:
            self.goal_state = [[1,2,3],[4,5,6],[7,8,0]]
        
        def run_solver():
            if algo == "BFS":
                self.solution = bfs_solve(self.initial_state, self.goal_state)
            elif algo == "UCS":
                self.solution = ucs_solve(self.initial_state, self.goal_state)
            elif algo == "Greedy":
                self.solution = greedy_solve(self.initial_state, self.goal_state)
            elif algo == "A*":
                self.solution = astar_solve(self.initial_state, self.goal_state)
            elif algo == "DFS":
                self.solution = dfs_solve(self.initial_state, self.goal_state)
            elif algo == "IDS":
                self.solution = ids_solve(self.initial_state, self.goal_state)
            elif algo == "IDA*":
                self.solution = ida_star_solve(self.initial_state, self.goal_state)
            elif algo == "Simple Hill Climbing":
                self.solution = simple_hill_climbing_solve(self.initial_state, self.goal_state)
            elif algo == "Steepest Hill Climbing":
                self.solution = steepest_hill_climbing_solve(self.initial_state, self.goal_state)
            elif algo == "Stochastic Hill Climbing":
                self.solution = stochastic_hill_climbing_solve(self.initial_state, self.goal_state)
            elif algo == "Simulated Annealing":
                self.solution = simulated_annealing_solve(self.initial_state, self.goal_state)
            elif algo == "Beam Search":
                self.solution = beam_search_solve(self.initial_state, self.goal_state)
            elif algo == "Genetic Algorithm":
                self.solution = genetic_algorithm_solve(self.initial_state, self.goal_state)
            elif algo == "AND-OR Graph Search":
                self.solution = and_or_solve(self.initial_state, self.goal_state)
            elif algo == "Sensorless BFS":
                self.solution = sensorless_bfs_solve_wrapper(self.goal_state)
            elif algo == "General Problem Solver":
                self.solution = general_problem_solver(self.initial_state, self.goal_state)

        t = threading.Thread(target=run_solver)
        t.start()
        self.root.after(100, lambda: self.check_thread(t))

    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread(thread))
        else:
            if self.solution is not None:
                self.status_label.config(text="Solution found!")
                self.solution_text.config(state='normal')
                self.solution_text.delete(1.0, tk.END)
                for i, move in enumerate(self.solution):
                    self.solution_text.insert(tk.END, f"Step {i+1}: {move}\n")
                self.solution_text.config(state='disabled')
                self.animate_solution(self.solution)
            else:
                self.status_label.config(text="No solution found.")
                self.solution_text.config(state='normal')
                self.solution_text.delete(1.0, tk.END)
                self.solution_text.insert(tk.END, "No solution found.")
                self.solution_text.config(state='disabled')

    def reset_board(self):
        if self.initial_state:
            self.state = [row[:] for row in self.initial_state]
            self.create_board()
            self.status_label.config(text="Board reset to initial state.")
            self.solution_text.config(state='normal')
            self.solution_text.delete(1.0, tk.END)
            self.solution_text.config(state='disabled')

    def animate_solution(self, solution):
        if self.algo_var.get() == "Sensorless BFS":
            self.state = [row[:] for row in self.goal_state]
        else:
            self.state = [row[:] for row in self.initial_state]
        self.create_board()

        def step(index):
            if index < len(solution):
                move = solution[index]
                for new_state, move_candidate in generate_states(self.state):
                    if move_candidate == move:
                        self.state = new_state
                        self.create_board()
                        self.status_label.config(text=f"Move {index+1}/{len(solution)}: {move}")
                        self.root.after(500, step, index + 1)
                        break
            else:
                self.status_label.config(text="Completed!")
        step(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()