import tkinter as tk
import threading
import time
import random
from collections import defaultdict

class ANDORGraphSearch:
    def __init__(self, root, return_to_menu_callback=None):
        self.root = root
        self.root.title("AND-OR Search Puzzle Tree")
        self.root.geometry("1400x900")
        self.root.configure(bg='#FFF8F8')  # Màu nền đồng bộ với Puzzle
        self.stop_flag = False
        self.solution_found = False
        self.start_time = 0
        self.goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        self.initial_state = [1, 2, 3, 4, 5, 6, 0, 7, 8]
        self.visited = set()
        self.node_positions = {}
        self.node_id = 0
        self.depth = 0
        self.path = []
        self.return_to_menu_callback = return_to_menu_callback

        # Main frame
        self.main_frame = tk.Frame(root, bg='#FFF8F8')
        self.main_frame.pack(fill='both', expand=True)

        # Top frame for current and goal states
        self.top_frame = tk.Frame(self.main_frame, bg='#FFF8F8')
        self.top_frame.pack(fill='x', pady=10)

        # Current state frame
        self.frame_curr = tk.LabelFrame(self.top_frame, text="Current State", 
                                      font=('Arial', 12, 'bold'), bg='#FFF8F8')
        self.frame_curr.pack(side='left', padx=20)
        self.curr_buttons = []
        for i in range(9):
            btn = tk.Label(self.frame_curr, text='-', width=4, height=2,
                         font=('Arial', 16), relief='solid', bg='white')
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.curr_buttons.append(btn)

        # Goal state frame
        self.frame_goal = tk.LabelFrame(self.top_frame, text="Goal State", 
                                      font=('Arial', 12, 'bold'), bg='#FFF8F8')
        self.frame_goal.pack(side='left', padx=20)
        for i, val in enumerate(self.goal):
            lbl = tk.Label(self.frame_goal, text=str(val) if val != 0 else ' ', width=4, height=2,
                         font=('Arial', 16), relief='solid', bg='#e0e0e0')
            lbl.grid(row=i//3, column=i%3, padx=5, pady=5)

        # Control frame
        self.ctrl_frame = tk.Frame(self.main_frame, bg='#FFF8F8')
        self.ctrl_frame.pack(fill='x', pady=10)
        
        # Các nút điều khiển với màu sắc đồng bộ
        tk.Button(self.ctrl_frame, text="Random Puzzle", command=self.random_puzzle, 
                 font=('Arial', 12), bg='#FFB6C1', relief=tk.RAISED).pack(side='left', padx=10)
        tk.Button(self.ctrl_frame, text="Build Tree", command=self.start, 
                 font=('Arial', 12), bg='#FFB6C1', relief=tk.RAISED).pack(side='left', padx=10)
        tk.Button(self.ctrl_frame, text="Stop", command=self.stop, 
                 font=('Arial', 12), bg='#FFB6C1', relief=tk.RAISED).pack(side='left', padx=10)
        tk.Button(self.ctrl_frame, text="Exit", command=root.destroy, 
                 font=('Arial', 12), bg='#FF9999', relief=tk.RAISED).pack(side='left', padx=10)

        # Search tree frame
        self.frame_tree = tk.LabelFrame(self.main_frame, text="Search Tree", 
                                      font=('Arial', 12, 'bold'), bg='#FFF8F8')
        self.frame_tree.pack(fill='both', expand=True, pady=10)

        # Canvas with scrollbars
        self.canvas_frame = tk.Frame(self.frame_tree)
        self.canvas_frame.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient='vertical', command=self.canvas.yview)
        self.v_scroll.pack(side='right', fill='y')
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.h_scroll = tk.Scrollbar(self.frame_tree, orient='horizontal', command=self.canvas.xview)
        self.h_scroll.pack(side='bottom', fill='x')
        self.canvas.configure(xscrollcommand=self.h_scroll.set)

        # Status labels
        self.status_label = tk.Label(self.main_frame, text="Idle", font=('Arial', 12), bg='#FFF8F8')
        self.status_label.pack(pady=5)
        self.time_label = tk.Label(self.main_frame, text="Time: Not started", 
                                 font=('Arial', 14, 'bold'), fg='blue', bg='#FFF8F8')
        self.time_label.pack(pady=5)

        self.update_current(self.initial_state)

    def update_current(self, state):
        for i, v in enumerate(state):
            self.curr_buttons[i].config(text=str(v) if v != 0 else ' ')
        self.root.update()

    def draw_puzzle_grid(self, state, x, y, is_goal=False, is_current=False):
        state_str = str(state)
        if state_str not in self.node_positions:
            self.node_id += 1
            cell_size = 30
            padding = 5
            fill_color = '#FFD1DC'  # Màu hồng nhạt đồng bộ
            if is_goal:
                fill_color = '#FF9999'  # Màu hồng đậm cho goal
            elif is_current:
                fill_color = '#FFB6C1'  # Màu trung bình cho current
            
            grid = self.canvas.create_rectangle(x - cell_size * 1.5 - padding, y - cell_size * 1.5 - padding,
                                             x + cell_size * 1.5 + padding, y + cell_size * 1.5 + padding,
                                             outline='black' if not is_goal else '#FF6666', width=2,
                                             fill=fill_color)
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    value = state[idx] if idx < len(state) else 0
                    text = str(value) if value != 0 else ' '
                    self.canvas.create_rectangle(
                        x - cell_size * 1.5 + j * cell_size, y - cell_size * 1.5 + i * cell_size,
                        x - cell_size * 1.5 + (j + 1) * cell_size, y - cell_size * 1.5 + (i + 1) * cell_size,
                        fill=fill_color, outline='black'
                    )
                    self.canvas.create_text(
                        x - cell_size * 1.5 + j * cell_size + cell_size // 2,
                        y - cell_size * 1.5 + i * cell_size + cell_size // 2,
                        text=text, font=('Arial', 12, 'bold')
                    )
            self.node_positions[state_str] = (grid, x, y)
            if self.depth > 0 and len(self.node_positions) > 1:
                parent_str = list(self.node_positions.keys())[-2]
                parent_x, parent_y = self.node_positions[parent_str][1:]
                self.canvas.create_line(parent_x, parent_y + cell_size * 1.5 + padding, 
                                      x, y - cell_size * 1.5 - padding, 
                                      arrow=tk.LAST, width=2)
            self.canvas.config(scrollregion=self.canvas.bbox('all'))
        self.root.update()

    def exit_to_menu(self):
        if self.return_to_menu_callback:
            self.main_frame.pack_forget()  # Hide current UI
            self.return_to_menu_callback()  # Call the menu callback
        else:
            self.root.destroy()

    def update_current(self, state):
        for i, v in enumerate(state):
            self.curr_buttons[i].config(text=str(v) if v != 0 else ' ')
        self.root.update()

    def draw_puzzle_grid(self, state, x, y, is_goal=False, is_current=False):
        state_str = str(state)
        if state_str not in self.node_positions:
            self.node_id += 1
            cell_size = 30
            padding = 5
            fill_color = 'lightgreen'
            if is_goal:
                fill_color = '#ffcccc'  # Light red for goal
            elif is_current:
                fill_color = '#ccffcc'  # Light green for current
            
            grid = self.canvas.create_rectangle(x - cell_size * 1.5 - padding, y - cell_size * 1.5 - padding,
                                             x + cell_size * 1.5 + padding, y + cell_size * 1.5 + padding,
                                             outline='black' if not is_goal else 'red', width=2,
                                             fill=fill_color)
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    value = state[idx] if idx < len(state) else 0
                    text = str(value) if value != 0 else ' '
                    self.canvas.create_rectangle(
                        x - cell_size * 1.5 + j * cell_size, y - cell_size * 1.5 + i * cell_size,
                        x - cell_size * 1.5 + (j + 1) * cell_size, y - cell_size * 1.5 + (i + 1) * cell_size,
                        fill=fill_color, outline='black'
                    )
                    self.canvas.create_text(
                        x - cell_size * 1.5 + j * cell_size + cell_size // 2,
                        y - cell_size * 1.5 + i * cell_size + cell_size // 2,
                        text=text, font=('Arial', 12, 'bold')
                    )
            self.node_positions[state_str] = (grid, x, y)
            if self.depth > 0 and len(self.node_positions) > 1:
                parent_str = list(self.node_positions.keys())[-2]
                parent_x, parent_y = self.node_positions[parent_str][1:]
                self.canvas.create_line(parent_x, parent_y + cell_size * 1.5 + padding, 
                                      x, y - cell_size * 1.5 - padding, 
                                      arrow=tk.LAST, width=2)
            self.canvas.config(scrollregion=self.canvas.bbox('all'))
        self.root.update()

    def random_puzzle(self):
        self.initial_state = list(range(9))
        random.shuffle(self.initial_state)
        self.update_current(self.initial_state)
        self.visited.clear()
        self.node_positions.clear()
        self.node_id = 0
        self.depth = 0
        self.canvas.delete('all')
        self.status_label.config(text="New puzzle generated")
        self.time_label.config(text="Time: Not started", fg='blue')

    def start(self):
        self.stop_flag = False
        self.solution_found = False
        self.start_time = time.time()
        self.visited.clear()
        self.node_positions.clear()
        self.node_id = 0
        self.depth = 0
        self.path = []
        self.canvas.delete('all')
        self.time_label.config(text="Time: Running...", fg='blue')
        threading.Thread(target=self.and_or_search, daemon=True).start()

    def stop(self):
        self.stop_flag = True
        self.time_label.config(text="Time: Stopped", fg='red')

    def is_goal(self, state):
        return state == self.goal

    def get_successors(self, state):
        successors = []
        zero_idx = state.index(0)
        row, col = divmod(zero_idx, 3)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in moves:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_idx = new_row * 3 + new_col
                new_state = state.copy()
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                successors.append(new_state)
        return successors

    def and_or_search(self):
        self.visited.clear()
        self.draw_puzzle_grid(self.initial_state, 600, 50, is_current=True)
        self.depth = 0

        def solve(state, depth):
            if self.stop_flag:
                return False
                
            if self.is_goal(state):
                self.solution_found = True
                self.path.append(state)
                self.draw_puzzle_grid(state, 600 + (self.node_id % 5 - 2) * 200, 
                                     50 + depth * 100, is_goal=True)
                return True

            state_str = str(state)
            if state_str in self.visited:
                return False

            self.visited.add(state_str)
            self.update_current(state)
            self.status_label.config(text=f"Exploring state: {state}")
            time.sleep(0.5)  # Reduced sleep time for better responsiveness

            successors = self.get_successors(state)
            if not successors:
                return False

            for next_state in successors:
                self.depth = depth + 1
                x_pos = 600 + ((self.node_id % 10) - 5) * 150  # Better node distribution
                y_pos = 50 + self.depth * 100
                self.draw_puzzle_grid(next_state, x_pos, y_pos)
                
                if solve(next_state, depth + 1):
                    self.path.append(next_state)
                    return True

            return False

        success = solve(self.initial_state, 0)
        elapsed = time.time() - self.start_time
        
        if success:
            self.status_label.config(text=f"Goal found in {len(self.path)} steps!")
            self.time_label.config(text=f"Solution found in {elapsed:.2f} seconds", fg='green')
            # Highlight solution path
            for i, state in enumerate(self.path):
                x, y = self.node_positions[str(state)][1], self.node_positions[str(state)][2]
                self.canvas.create_oval(x-20, y-20, x+20, y+20, outline='blue', width=3)
        else:
            self.status_label.config(text="No solution found")
            self.time_label.config(text=f"Search completed in {elapsed:.2f} seconds (no solution)", fg='red')

# Example of how to use this class with a menu callback
