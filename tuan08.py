import tkinter as tk
from tkinter import messagebox
from collections import deque
import heapq
import threading

GOAL_STATE = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

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

def bfs_solve(start_state):
    from collections import deque
    queue = deque([(start_state, [])])
    visited = set()
    visited.add(tuple(map(tuple, start_state)))
    while queue:
        current_state, path = queue.popleft()
        if current_state == GOAL_STATE:
            return path
        for new_state, move in generate_states(current_state):
            state_tuple = tuple(map(tuple, new_state))
            if state_tuple not in visited:
                visited.add(state_tuple)
                queue.append((new_state, path + [move]))
    return None

def ucs_solve(start_state):
    import heapq
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    heapq.heappush(pq, (0, start_state, []))
    visited = {initial_tuple: 0}
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == GOAL_STATE:
            return path
        for new_state, move in generate_states(state):
            new_cost = cost + 1
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited or new_cost < visited[new_tuple]:
                visited[new_tuple] = new_cost
                heapq.heappush(pq, (new_cost, new_state, path + [move]))
    return None

def greedy_solve(start_state):
    import heapq
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    h = manhattan_distance(start_state)
    heapq.heappush(pq, (h, start_state, []))
    visited = {initial_tuple}
    while pq:
        h, state, path = heapq.heappop(pq)
        if state == GOAL_STATE:
            return path
        for new_state, move in generate_states(state):
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited:
                visited.add(new_tuple)
                new_h = manhattan_distance(new_state)
                heapq.heappush(pq, (new_h, new_state, path + [move]))
    return None

def astar_solve(start_state):
    import heapq
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    h = manhattan_distance(start_state)
    g = 0
    f = g + h
    heapq.heappush(pq, (f, g, start_state, []))
    visited = {initial_tuple: f}
    while pq:
        f, g, state, path = heapq.heappop(pq)
        if state == GOAL_STATE:
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

def dfs_solve(start_state):
    solution = None
    visited = set()
    def dfs(state, path):
        nonlocal solution
        if state == GOAL_STATE:
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

def ids_solve(start_state, max_depth=50):
    solution = None
    def dls(state, path, depth, visited):
        nonlocal solution
        if state == GOAL_STATE:
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

def ida_star_solve(start_state):
    solution = None
    bound = manhattan_distance(start_state)
    def search(g, bound, path, moves):
        nonlocal solution
        state = path[-1]
        f = g + manhattan_distance(state)
        if f > bound:
            return f
        if state == GOAL_STATE:
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

class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")

        self.root.configure(bg='#FFEBEE')
        
        # Initial state
        self.initial_state = None
        self.state = None
        
        # The final solution (list of moves)
        self.solution = None
        
        self.left_frame = tk.Frame(self.root, bg='#FFEBEE')
        self.right_frame = tk.Frame(self.root, bg='#FFEBEE')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        self.create_left_panel()
        self.create_right_panel()

   
    def create_left_panel(self):
        # Frame cho việc nhập trạng thái ban đầu
        input_frame = tk.LabelFrame(
            self.left_frame, 
            text="Enter initial state (0 = blank)",
            bg='#FFEBEE', 
            font=('Arial', 12, 'bold'), 
            fg='black', 
            bd=3
        )
        input_frame.pack(padx=5, pady=5, fill=tk.BOTH)
        
        self.input_entries = [[None]*3 for _ in range(3)]
        default_values = [['2','6','5'],
                          ['8','0','7'],
                          ['4','3','1']]
        for i in range(3):
            for j in range(3):
                e = tk.Entry(input_frame, width=3, font=('Arial', 14), justify='center')
                e.grid(row=i, column=j, padx=5, pady=5)
                e.insert(0, default_values[i][j])
                self.input_entries[i][j] = e
        
        update_button = tk.Button(
            input_frame, 
            text="Update", 
            font=('Arial', 12, 'bold'), 
            bg='#FFB6C1',  
            fg='black',
            command=self.update_initial_state
        )
        update_button.grid(row=3, column=0, columnspan=3, pady=5)
        
        algo_frame = tk.LabelFrame(
            self.left_frame, 
            text="Algorithms", 
            bg='#FFEBEE', 
            font=('Arial', 12, 'bold'), 
            fg='black', 
            bd=3
        )
        algo_frame.pack(padx=5, pady=5, fill=tk.BOTH)
        
        tk.Button(algo_frame, text="BFS",   bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("BFS")).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(algo_frame, text="UCS",   bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("UCS")).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(algo_frame, text="Greedy",bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("Greedy")).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(algo_frame, text="A*",    bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("A*")).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(algo_frame, text="DFS",   bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("DFS")).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(algo_frame, text="IDS",   bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("IDS")).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(algo_frame, text="IDA*",  bg='#FFB6C1', fg='black',
                  font=('Arial', 12, 'bold'),
                  command=lambda: self.start_solver_thread("IDA*")).pack(padx=5, pady=5, fill=tk.X)

    def create_right_panel(self):
        # Puzzle board
        self.board_frame = tk.LabelFrame(
            self.right_frame, 
            text="8-Puzzle Board",
            bg='#FFEBEE', 
            font=('Arial', 12, 'bold'), 
            fg='black', 
            bd=3
        )
        self.board_frame.pack(pady=5)
        
        self.buttons = [[None]*3 for _ in range(3)]
        
        # Frame cho Steps
        steps_frame = tk.LabelFrame(
            self.right_frame, 
            text="Steps", 
            bg='#FFEBEE', 
            font=('Arial', 12, 'bold'), 
            fg='black', 
            bd=3
        )
        steps_frame.pack(pady=10, fill=tk.BOTH)
        
        self.steps_label = tk.Label(
            steps_frame, 
            text="0", 
            bg='#FFEBEE', 
            font=('Arial', 16, 'bold'), 
            fg='red'
        )
        self.steps_label.pack(padx=10, pady=10)
        
        self.create_board()

    def create_board(self):
        if self.state is None:
            self.state = [[int(e.get()) for e in row] for row in self.input_entries]
            self.initial_state = [row[:] for row in self.state]
        
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                btn = tk.Button(
                    self.board_frame, 
                    text=str(val) if val != 0 else '',
                    font=('Arial', 20), 
                    width=4, 
                    height=2, 
                    bg='#FFB6C1',  
                    fg='black'
                )
                btn.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = btn

    def update_initial_state(self):
        try:
            new_state = [[int(self.input_entries[i][j].get()) for j in range(3)] for i in range(3)]
        except ValueError:
            messagebox.showerror("Error", "Please enter integers from 0 to 8.")
            return
        
        flat = [num for row in new_state for num in row]
        if sorted(flat) != list(range(9)):
            messagebox.showerror("Error", "Invalid state. Must contain unique numbers from 0 to 8.")
            return
        
        self.initial_state = [row[:] for row in new_state]
        self.state = [row[:] for row in new_state]
        self.update_board()

    def update_board(self):
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                self.buttons[i][j].config(text=str(val) if val != 0 else '')

    def start_solver_thread(self, algo):
        self.solution = None
        def run_solver():
            if algo == "BFS":
                self.solution = bfs_solve(self.initial_state)
            elif algo == "UCS":
                self.solution = ucs_solve(self.initial_state)
            elif algo == "Greedy":
                self.solution = greedy_solve(self.initial_state)
            elif algo == "A*":
                self.solution = astar_solve(self.initial_state)
            elif algo == "DFS":
                self.solution = dfs_solve(self.initial_state)
            elif algo == "IDS":
                self.solution = ids_solve(self.initial_state)
            elif algo == "IDA*":
                self.solution = ida_star_solve(self.initial_state)
        
        t = threading.Thread(target=run_solver)
        t.start()
        self.root.after(100, lambda: self.check_thread(t))

    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread(thread))
        else:
            if self.solution:
                self.steps_label.config(text="0")
                self.animate_solution(self.solution)
            else:
                messagebox.showinfo("Result", "No solution found.")

    def animate_solution(self, solution):
        def step(index):
            self.steps_label.config(text=str(index))
            if index < len(solution):
                # Áp dụng nước đi tiếp theo
                for new_state, move in generate_states(self.state):
                    if move == solution[index]:
                        self.state = new_state
                        self.update_board()
                        self.root.after(500, step, index + 1)
                        break
            else:
                messagebox.showinfo("Done", f"Completed {index} moves.")
        step(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
