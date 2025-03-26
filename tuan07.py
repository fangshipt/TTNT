import tkinter as tk
from tkinter import messagebox
from collections import deque
import heapq

# Trạng thái đích của bài toán 8-puzzle
GOAL_STATE = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# Các hướng di chuyển: UP, DOWN, LEFT, RIGHT
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

#BFS
def bfs_solve(start_state):
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

#Uniform Cost Search
def ucs_solve(start_state):
    pq = []
    initial_tuple = tuple(map(tuple, start_state))
    heapq.heappush(pq, (0, start_state, []))
    visited = {initial_tuple: 0}
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == GOAL_STATE:
            return path
        for new_state, move in generate_states(state):
            new_cost = cost + 1  # Giả sử mỗi bước có chi phí = 1
            new_tuple = tuple(map(tuple, new_state))
            if new_tuple not in visited or new_cost < visited[new_tuple]:
                visited[new_tuple] = new_cost
                heapq.heappush(pq, (new_cost, new_state, path + [move]))
    return None

#Greedy Search
def greedy_solve(start_state):
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

#Hàm heuristic: khoảng cách Manhattan
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

#Informed Search
def astar_solve(start_state):
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

class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.initial_state = None # Sẽ được cập nhật từ input của người dùng
        self.state = None
        self.buttons = [[None]*3 for _ in range(3)]
        self.create_ui()
    
    def create_ui(self):
        self.root.configure(bg='#FFEBEE')
        input_frame = tk.Frame(self.root, bg='#FFEBEE')
        input_frame.grid(row=0, column=0, columnspan=3, pady=15, padx=15)
        tk.Label(
            input_frame, 
            text="Nhập trạng thái ban đầu (0 đại diện cho ô trống):", 
            bg='#FFEBEE', 
            font=('Arial', 14)
        ).grid(row=0, column=0, columnspan=3, pady=(0, 10))
        self.input_entries = [[None]*3 for _ in range(3)]
        default_values = [['2','6','5'],
                          ['8','0','7'],
                          ['4','3','1']]
        for i in range(3):
            for j in range(3):
                e = tk.Entry(
                    input_frame, 
                    width=4, 
                    font=('Arial', 16), 
                    justify='center'
                )
                e.grid(row=i+1, column=j, padx=10, pady=10)
                e.insert(0, default_values[i][j])
                self.input_entries[i][j] = e
        update_button = tk.Button(
            input_frame, 
            text="Cập nhật trạng thái", 
            font=('Arial', 14), 
            bg='#FFB6C1', 
            command=self.update_initial_state
        )
        update_button.grid(row=4, column=0, columnspan=3, pady=10)
        control_frame = tk.Frame(self.root, bg='#FFEBEE')
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.algo_var = tk.StringVar(value="BFS")
        algorithms = [
            ("BFS", "BFS"), 
            ("Uniform Cost Search", "UCS"), 
            ("Greedy Search", "Greedy"), 
            ("Informed Search", "A*")
        ]
        col = 0
        for text, mode in algorithms:
            rb = tk.Radiobutton(
                control_frame, 
                text=text, 
                variable=self.algo_var, 
                value=mode, 
                bg='#FFEBEE', 
                font=('Arial', 12)
            )
            rb.grid(row=0, column=col, padx=5)
            col += 1
        
        solve_button = tk.Button(
            control_frame, 
            text="Solve", 
            font=('Arial', 12), 
            bg='#FFB6C1', 
            command=self.solve
        )
        solve_button.grid(row=0, column=col, padx=5)
        
        # Frame hiển thị trạng thái
        self.board_frame = tk.Frame(self.root, bg='#FFEBEE')
        self.board_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.create_board()
    
    def create_board(self):
        if self.state is None:
            self.state = [[int(e.get()) for e in row] for row in self.input_entries]
            self.initial_state = [row[:] for row in self.state]
        for i in range(3):
            for j in range(3):
                btn = tk.Button(
                    self.board_frame, 
                    text=str(self.state[i][j]) if self.state[i][j] != 0 else '',
                    font=('Arial', 20), 
                    width=5, 
                    height=2, 
                    bg='#FFB6C1', 
                    fg='black'
                )
                btn.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = btn
    
    def update_initial_state(self):
        try:
            new_state = [
                [int(self.input_entries[i][j].get()) for j in range(3)] 
                for i in range(3)
            ]
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập số nguyên từ 0 đến 8.")
            return
        
        # Kiểm tra tính hợp lệ: chứa các số từ 0 đến 8 duy nhất
        flat = [num for row in new_state for num in row]
        if sorted(flat) != list(range(9)):
            messagebox.showerror("Lỗi", "Trạng thái không hợp lệ. Phải chứa các số từ 0 đến 8 duy nhất.")
            return
        
        self.initial_state = [row[:] for row in new_state]
        self.state = [row[:] for row in new_state]
        self.update_board()
    
    def update_board(self):
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                self.buttons[i][j].config(text=str(val) if val != 0 else '')
    
    def animate_solution(self, solution):
        def step(index):
            if index < len(solution):
                for new_state, move in generate_states(self.state):
                    if move == solution[index]:
                        self.state = new_state
                        self.update_board()
                        self.root.after(500, step, index + 1)
                        break
        step(0)
    
    def solve(self):
        if self.initial_state is None:
            messagebox.showerror("Lỗi", "Vui lòng cập nhật trạng thái ban đầu trước khi giải!")
            return
        
        algo = self.algo_var.get()
        if algo == "BFS":
            solution = bfs_solve(self.initial_state)
        elif algo == "UCS":
            solution = ucs_solve(self.initial_state)
        elif algo == "Greedy":
            solution = greedy_solve(self.initial_state)
        elif algo == "A*":
            solution = astar_solve(self.initial_state)
        else:
            solution = None
        
        if solution:
            self.animate_solution(solution)
        else:
            messagebox.showinfo("Kết quả", "Không tìm thấy lời giải!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
