import tkinter as tk
from tkinter import messagebox
import threading
import time
from algorithm import *
# Giao diện 8-Puzzle
class PuzzleApp:
    def __init__(self, root, algo, main_menu):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("1000x750")
        self.root.configure(bg='#FFF8F8')
        self.algo = algo
        self.main_menu = main_menu
        self.initial_state = None
        self.goal_state = None
        self.state = None
        self.solution = None
        self.solving_time = 0.0

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
        tk.Label(self.left_frame, text="Input states (0-8, 0 is blank)", bg='#FFF8F8', font=('Arial', 10, 'italic')).pack(pady=5)

        init_frame = tk.LabelFrame(self.left_frame, text="Initial State", bg='#FFF8F8', font=('Arial', 12, 'bold'))
        init_frame.pack(padx=5, pady=5)
        self.init_entries = [[tk.Entry(init_frame, width=2, font=('Arial', 20), justify='center') for j in range(3)] for i in range(3)]
        init_defaults = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
        for i in range(3):
            for j in range(3):
                self.init_entries[i][j].grid(row=i, column=j, padx=10, pady=10)
                self.init_entries[i][j].insert(0, init_defaults[i][j])

        goal_frame = tk.LabelFrame(self.left_frame, text="Goal State", bg='#FFF8F8', font=('Arial', 12, 'bold'))
        goal_frame.pack(padx=5, pady=5)
        self.goal_entries = [[tk.Entry(goal_frame, width=2, font=('Arial', 20), justify='center') for j in range(3)] for i in range(3)]
        goal_defaults = [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 0]]
        for i in range(3):
            for j in range(3):
                self.goal_entries[i][j].grid(row=i, column=j, padx=10, pady=10)
                self.goal_entries[i][j].insert(0, goal_defaults[i][j])

        tk.Button(self.left_frame, text="Update States", font=('Arial', 12, 'bold'), bg='#FFB6C1', command=self.update_states).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(self.left_frame, text="Solve", font=('Arial', 12, 'bold'), bg='#FFB6C1', command=self.start_solver_thread).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(self.left_frame, text="Reset", font=('Arial', 12, 'bold'), bg='#FFB6C1', command=self.reset_board).pack(padx=5, pady=5, fill=tk.X)
        tk.Button(self.left_frame, text="Exit to Menu", font=('Arial', 12, 'bold'), bg='#FFB6C1', command=self.exit_to_menu).pack(padx=5, pady=5, fill=tk.X)
        tk.Label(self.left_frame, text=f"Algorithm: {self.algo}", bg='#FFF8F8', font=('Arial', 12, 'italic')).pack(pady=5)

    def exit_to_menu(self):
        self.root.destroy()
        self.main_menu.root.deiconify()

    def create_center_panel(self):
        tk.Label(self.center_frame, text="8-Puzzle Board", bg='#FFF8F8', font=('Arial', 14, 'bold')).pack(pady=5)
        self.board_frame = tk.Frame(self.center_frame, bg='#FFF8F8')
        self.board_frame.pack(pady=5)
        self.buttons = [[None]*3 for _ in range(3)]
        self.create_board()
        self.status_label = tk.Label(self.center_frame, text="", bg='#FFF8F8', font=('Arial', 12))
        self.status_label.pack(pady=5)

    def create_right_panel(self):
        tk.Label(self.right_frame, text="Solution Steps:", bg='#FFF8F8', font=('Arial', 12, 'bold')).pack(pady=5)
        solution_frame = tk.Frame(self.right_frame, bg='#FFF8F8')
        solution_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.solution_text = tk.Text(solution_frame, height=20, width=20, font=('Arial', 12))
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
        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                btn = tk.Button(self.board_frame, text=str(val) if val != 0 else '', font=('Arial', 24, 'bold'),
                                width=4, height=2, bg='#FFB6C1')
                btn.grid(row=i, column=j, padx=8, pady=8)
                self.buttons[i][j] = btn

    def update_states(self):
        try:
            new_init = [[int(self.init_entries[i][j].get().strip()) for j in range(3)] for i in range(3)]
            new_goal = [[int(self.goal_entries[i][j].get().strip()) for j in range(3)] for i in range(3)]
        except ValueError:
            messagebox.showerror("Error", "Please enter integers from 0 to 8.")
            return
        if sorted(sum(new_init, [])) != list(range(9)) or sorted(sum(new_goal, [])) != list(range(9)):
            messagebox.showerror("Error", "Invalid state. Must contain unique numbers from 0 to 8.")
            return
        self.initial_state = new_init
        self.goal_state = new_goal
        self.state = [row[:] for row in new_init]
        self.create_board()
        self.status_label.config(text="States updated.")

    def start_solver_thread(self):
        self.solution = None
        self.status_label.config(text="Solving...")
        self.solution_text.delete(1.0, tk.END)

        if not self.goal_state:
            self.goal_state = [[1,2,3],[4,5,6],[7,8,0]]

        def run_solver():
            start_time = time.time()
            if self.algo == "BFS":
                self.solution = bfs_solve(self.initial_state, self.goal_state)
            elif self.algo == "UCS":
                self.solution = ucs_solve(self.initial_state, self.goal_state)
            elif self.algo == "Greedy":
                self.solution = greedy_solve(self.initial_state, self.goal_state)
            elif self.algo == "A*":
                self.solution = astar_solve(self.initial_state, self.goal_state)
            elif self.algo == "DFS":
                self.solution = dfs_solve(self.initial_state, self.goal_state)
            elif self.algo == "IDS":
                self.solution = ids_solve(self.initial_state, self.goal_state)
            elif self.algo == "IDA*":
                self.solution = ida_star_solve(self.initial_state, self.goal_state)
            elif self.algo == "Simple Hill Climbing":
                self.solution = simple_hill_climbing_solve(self.initial_state, self.goal_state)
            elif self.algo == "Steepest Hill Climbing":
                self.solution = steepest_hill_climbing_solve(self.initial_state, self.goal_state)
            elif self.algo == "Stochastic Hill Climbing":
                self.solution = stochastic_hill_climbing_solve(self.initial_state, self.goal_state)
            elif self.algo == "Simulated Annealing":
                self.solution = simulated_annealing_solve(self.initial_state, self.goal_state)
            elif self.algo == "Beam Search":
                self.solution = beam_search_solve(self.initial_state, self.goal_state)
            elif self.algo == "Genetic Algorithm":
                self.solution = genetic_algorithm_solve(self.initial_state, self.goal_state)
            elif self.algo == "Q‑Learning":
                self.solution = q_learning_solve(self.initial_state, self.goal_state,
                                                episodes=5000, max_steps_per_episode=200,
                                                alpha=0.1, gamma=0.9, epsilon=0.1)
            end_time = time.time()
            self.solving_time = end_time - start_time

        t = threading.Thread(target=run_solver)
        t.start()
        self.root.after(100, lambda: self.check_thread(t))

    def check_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.check_thread(thread))
        else:
            if self.solution:
                self.status_label.config(text=f"Solution found in {self.solving_time:.4f} seconds!")
                self.root.update_idletasks()
                for i, move in enumerate(self.solution):
                    self.solution_text.insert(tk.END, f"Step {i+1}: {move}\n")
                self.root.after(1000, lambda: self.animate_solution(self.solution))

            else:
                self.status_label.config(text=f"No solution found after {self.solving_time:.4f} seconds.")
                self.root.update_idletasks()
                self.solution_text.insert(tk.END, "No solution found.")

    def reset_board(self):
        if self.initial_state:
            self.state = [row[:] for row in self.initial_state]
            self.create_board()
            self.status_label.config(text="Board reset.")
            self.solution_text.delete(1.0, tk.END)

    def animate_solution(self, solution):
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
                        self.root.after(200, step, index + 1)
                        break
            else:
                self.status_label.config(text=f"Completed in {self.solving_time:.4f} seconds!")
        step(0)