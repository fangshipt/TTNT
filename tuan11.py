import tkinter as tk
from tkinter import messagebox
import heapq
import threading
from collections import deque
import random
import math

# -------------------------------------
# HÀM HỖ TRỢ CHO 8-PUZZLE
# -------------------------------------
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

MOVES = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

def generate_states(state):
    blank_x, blank_y = find_blank(state)
    new_states = []
    for move, (dx, dy) in MOVES.items():
        new_x, new_y = blank_x + dx, blank_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = [row[:] for row in state]
            new_state[blank_x][blank_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[blank_x][blank_y]
            new_states.append((new_state, move))
    return new_states

def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_i = (val - 1) // 3
                goal_j = (val - 1) % 3
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

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
    h = manhattan_distance(start_state)
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
                new_h = manhattan_distance(new_state)
                heapq.heappush(pq, (new_h, new_state, path + [move]))
    return None

def astar_solve(start_state, goal_state):
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    h = manhattan_distance(start_state)
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
            new_h = manhattan_distance(new_state)
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
    bound = manhattan_distance(start_state)
    def search(g, bound, path, moves):
        nonlocal solution
        state = path[-1]
        f = g + manhattan_distance(state)
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
    current_h = manhattan_distance(current_state)
    path = []
    if current_state == goal_state:
        return path
    while True:
        found_better = False
        for new_state, move in generate_states(current_state):
            new_h = manhattan_distance(new_state)
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
    current_h = manhattan_distance(current_state)
    path = []
    if current_state == goal_state:
        return path
    while True:
        best_neighbor = None
        best_move = None
        best_h = current_h
        for new_state, move in generate_states(current_state):
            new_h = manhattan_distance(new_state)
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
        current_h = manhattan_distance(current_state)
        path = []
        steps = 0
        while steps < max_steps:
            if current_state == goal_state:
                return path
            better_neighbors = []
            for new_state, move in generate_states(current_state):
                new_h = manhattan_distance(new_state)
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
    current_cost = manhattan_distance(current_state)
    path = []
    T = T_init
    steps = 0
    while T > 1 and steps < max_steps:
        if current_state == goal_state:
            return path
        neighbors = generate_states(current_state)
        new_state, move = random.choice(neighbors)
        new_cost = manhattan_distance(new_state)
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

# -------------------------------------
# THUẬT TOÁN BEAM SEARCH
# -------------------------------------
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
        new_beam.sort(key=lambda x: manhattan_distance(x[0]))
        beam = new_beam[:beam_width]
    return None
def and_or_solve(start_state, goal_state, limit=50):
    """
    OR-SEARCH có cycle‑check và cắt theo độ sâu.
    Trả về list of moves hoặc None nếu không tìm được.
    """
    def and_or(state, explored, depth):
        # Nếu đã tới đích
        if state == goal_state:
            return []
        # Nếu quá sâu → cắt
        if depth > limit:
            return None
        # cycle‑check
        state_t = tuple(map(tuple, state))
        if state_t in explored:
            return None
        # đánh dấu đã thăm
        new_explored = explored.union({state_t})
        # thử từng action
        for new_state, move in generate_states(state):
            result = and_or(new_state, new_explored, depth + 1)
            if result is not None:
                return [move] + result
        return None

    return and_or(start_state, set(), 0)

# -------------------------------------
# GIAO DIỆN CHÍNH (Tkinter)
# -------------------------------------
class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("1000x600")
        self.root.configure(bg='#FFF8F8')
        
        self.initial_state = None
        self.goal_state = None
        self.state = None
        
        self.snapshots = []
        self.solution = None
        
        self.main_frame = tk.Frame(self.root, bg='#FFF8F8')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.left_frame = tk.Frame(self.main_frame, bg='#FFF8F8', bd=2, relief=tk.RIDGE)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.center_frame = tk.Frame(self.main_frame, bg='#FFF8F8', bd=2, relief=tk.RIDGE)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_frame = tk.Frame(self.main_frame, bg='#FFF8F8', bd=2, relief=tk.RIDGE)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_left_panel()
        self.create_center_panel()
        self.create_right_panel()

    # ----------------------------
    # Panel trái: Nhập Initial & Goal + chọn thuật toán
    # ----------------------------
    def create_left_panel(self):
        init_frame = tk.LabelFrame(self.left_frame, text="Initial State",
                                   bg='#FFF8F8', font=('Arial', 12, 'bold'), fg='black')
        init_frame.pack(padx=5, pady=5)
        self.init_entries = [[None]*3 for _ in range(3)]
        init_defaults = [['2','6','5'],
                         ['8','0','7'],
                         ['4','3','1']]
        for i in range(3):
            for j in range(3):
                e = tk.Entry(init_frame, width=3, font=('Arial', 14), justify='center')
                e.grid(row=i, column=j, padx=3, pady=3)
                e.insert(0, init_defaults[i][j])
                self.init_entries[i][j] = e

        goal_frame = tk.LabelFrame(self.left_frame, text="Goal State",
                                   bg='#FFF8F8', font=('Arial', 12, 'bold'), fg='black')
        goal_frame.pack(padx=5, pady=5)
        self.goal_entries = [[None]*3 for _ in range(3)]
        goal_defaults = [['1','2','3'],
                         ['4','5','6'],
                         ['7','8','0']]
        for i in range(3):
            for j in range(3):
                e = tk.Entry(goal_frame, width=3, font=('Arial', 14), justify='center')
                e.grid(row=i, column=j, padx=3, pady=3)
                e.insert(0, goal_defaults[i][j])
                self.goal_entries[i][j] = e

        update_button = tk.Button(self.left_frame, text="Update States", font=('Arial', 12, 'bold'),
                                  bg='#FFB6C1', fg='black', command=self.update_states)
        update_button.pack(padx=5, pady=5, fill=tk.X)
        
        algo_frame = tk.LabelFrame(self.left_frame, text="Algorithm Selection",
                                   bg='#FFF8F8', font=('Arial', 12, 'bold'), fg='black')
        algo_frame.pack(padx=5, pady=5, fill=tk.X)
        
        self.algo_var = tk.StringVar()
        self.algo_var.set("BFS")
        algo_options = ["BFS", "UCS", "Greedy", "A*", "DFS", "IDS", "IDA*", 
                "Simple Hill Climbing", "Steepest Hill Climbing", 
                "Stochastic Hill Climbing", "Simulated Annealing", "Beam Search",
                "AND-OR Graph Search"]
        algo_menu = tk.OptionMenu(algo_frame, self.algo_var, *algo_options)
        algo_menu.config(font=('Arial', 12), bg='#FFB6C1', fg='black')
        algo_menu.pack(padx=5, pady=5, fill=tk.X)
        
        solve_button = tk.Button(self.left_frame, text="Solve", font=('Arial', 12, 'bold'),
                                 bg='#FFB6C1', fg='black', command=lambda: self.start_solver_thread(self.algo_var.get()))
        solve_button.pack(padx=5, pady=5, fill=tk.X)

    # ----------------------------
    # Panel giữa: Hiển thị Board
    # ----------------------------
    def create_center_panel(self):
        board_label = tk.Label(self.center_frame, text="8-Puzzle Board",
                               bg='#FFF8F8', font=('Arial', 14, 'bold'))
        board_label.pack(pady=5)
        self.board_frame = tk.Frame(self.center_frame, bg='#FFF8F8')
        self.board_frame.pack(pady=5)
        self.buttons = [[None]*3 for _ in range(3)]
        self.create_board()

    # ----------------------------
    # Panel phải: Hiển thị Steps (Snapshots)
    # ----------------------------
    def create_right_panel(self):
        step_label = tk.Label(self.right_frame, text="Steps",
                              bg='#FFF8F8', font=('Arial', 14, 'bold'))
        step_label.pack(pady=5)
        self.snapshot_canvas = tk.Canvas(self.right_frame, bg='#FFF8F8')
        self.snapshot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.right_frame, orient="vertical", command=self.snapshot_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.snapshot_canvas.configure(yscrollcommand=scrollbar.set)
        self.snapshot_frame = tk.Frame(self.snapshot_canvas, bg='#FFF8F8')
        self.snapshot_canvas.create_window((0,0), window=self.snapshot_frame, anchor="nw")
        self.snapshot_frame.bind("<Configure>", lambda e: self.snapshot_canvas.configure(
            scrollregion=self.snapshot_canvas.bbox("all")))

    # ----------------------------
    # TẠO/UPDATE BOARD
    # ----------------------------
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
                                width=2, height=1,
                                bg='#FFB6C1', fg='black')
                btn.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = btn

    # ----------------------------
    # UPDATE STATES
    # ----------------------------
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
        for widget in self.snapshot_frame.winfo_children():
            widget.destroy()
        self.snapshots = []

    # ----------------------------
    # VẼ SNAPSHOT
    # ----------------------------
    def draw_snapshot(self, state, step):
        snap = tk.Frame(self.snapshot_frame, bd=1, relief="solid", bg='#FFF8F8')
        snap.grid(row=step // 4, column=step % 4, padx=5, pady=5)
        tk.Label(snap, text=f"Step {step}", font=('Arial', 10, 'bold'), bg='#FFF8F8').grid(row=0, column=0, columnspan=3)
        for i in range(3):
            for j in range(3):
                val = state[i][j]
                cell = tk.Label(snap,
                                text=str(val) if val != 0 else '',
                                font=('Arial', 12),
                                width=2, height=1,
                                bg='#FFB6C1', fg='black', relief="ridge")
                cell.grid(row=i+1, column=j, padx=2, pady=2)
        self.snapshots.append(snap)

    # ----------------------------
    # CHẠY THUẬT TOÁN TRONG THREAD
    # ----------------------------
    def start_solver_thread(self, algo):
        self.solution = None
        for widget in self.snapshot_frame.winfo_children():
            widget.destroy()
        self.snapshots = []
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
            elif algo == "AND-OR Graph Search":
                self.solution = and_or_solve(self.initial_state, self.goal_state, limit=100)

        t = threading.Thread(target=run_solver)
        t.start()
        self.root.after(100, lambda: self.check_thread(t))

    # ----------------------------
    # KIỂM TRA THREAD
    # ----------------------------
    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread(thread))
        else:
            if self.solution is not None:
                self.animate_solution(self.solution)
            else:
                messagebox.showinfo("Result", "No solution found.")

    # ----------------------------
    # ANIMATION
    # ----------------------------
    def animate_solution(self, solution):
        self.draw_snapshot(self.state, 0)
        def step(index):
            if index < len(solution):
                move = solution[index]
                for new_state, move_candidate in generate_states(self.state):
                    if move_candidate == move:
                        self.state = new_state
                        self.create_board()
                        self.draw_snapshot(self.state, index + 1)
                        self.root.after(500, step, index + 1)
                        break
            else:
                messagebox.showinfo("Done", f"Completed {index} moves.")
        step(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
