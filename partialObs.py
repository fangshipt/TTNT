import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame
from collections import deque
import threading
import time
from itertools import permutations

# Constants
MOVES = {'up': -3, 'down': 3, 'left': -1, 'right': 1}
GRID_SIZE = 3
OBSERVATION = {0: 1, 1: 2, 2: 3}

INITIAL_BELIEF_STATES = [
    (1, 2, 0, 4, 5, 3, 6, 7, 8),
    (1, 0, 2, 4, 5, 3, 6, 7, 8),
    (0, 1, 2, 4, 5, 3, 6, 7, 8),
]

def generate_goal_candidates():
    all_states = permutations(range(9))
    valid_goals = []
    for state in all_states:
        if all(state[pos] == val for pos, val in OBSERVATION.items()):
            valid_goals.append(state)
    return valid_goals

GOAL_CANDIDATES = generate_goal_candidates()

class PartialObsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Belief State Search With Partial Observation")
        self.root.geometry("1200x850")
        self.root.configure(bg='#FFF8F8')
        self.build_ui()

    def build_ui(self):
        ctrl = tk.Frame(self.root, bg='#FFF8F8')
        ctrl.pack(fill='x', pady=8)
        self.run_btn = tk.Button(ctrl, text="Run", command=self.start_search,
                                 font=('Arial', 12, 'bold'), bg='#FFB6C1')
        self.run_btn.pack(side='left', padx=8)
        self.time_lbl = tk.Label(ctrl, text="Time: 0.00s", bg='#FFF8F8', font=('Arial', 12))
        self.time_lbl.pack(side='left', padx=12)
        self.step_lbl = tk.Label(ctrl, text=f"Belief Steps: 0 / {len(INITIAL_BELIEF_STATES)}",
                                 bg='#FFF8F8', font=('Arial', 12))
        self.step_lbl.pack(side='left', padx=12)
        self.status_lbl = tk.Label(ctrl, text="", bg='#FFF8F8', font=('Arial', 12, 'italic'), fg='#D32F2F')
        self.status_lbl.pack(side='left', padx=12)
        canvas = Canvas(self.root, height=180, bg='#FFF8F8', highlightthickness=0)
        hbar = Scrollbar(self.root, orient='horizontal', command=canvas.xview)
        canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side='bottom', fill='x')
        canvas.pack(side='top', fill='x', padx=10)
        self.belief_frame = Frame(canvas, bg='#FFF8F8')
        canvas.create_window((0, 0), window=self.belief_frame, anchor='nw')
        self.draw_beliefs(INITIAL_BELIEF_STATES)
        self.belief_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))
        container = tk.Frame(self.root, bg='#FFF8F8')
        container.pack(fill='both', expand=True, padx=10, pady=10)
        obs_frame = tk.LabelFrame(container, text="Partial Observation",
                                   bg='#FFF8F8', font=('Arial', 14, 'bold'), fg='#333')
        obs_frame.pack(side='left', padx=10, pady=10)
        self.obs_cells = []
        for i in range(GRID_SIZE * GRID_SIZE):
            r, c = divmod(i, GRID_SIZE)
            val = OBSERVATION.get(i, None)
            lbl = tk.Label(
                obs_frame,
                text=str(val) if val is not None else ' ',
                width=4,
                height=2,
                relief='ridge' if val is not None else 'flat',
                font=('Arial', 16),
                bg='#FFF9C4' if val is not None else '#FFFFFF'
            )
            lbl.grid(row=r, column=c, padx=4, pady=4)
            self.obs_cells.append(lbl)
        self.goal_frame = tk.LabelFrame(container, text="Matching Goals",
                                         bg='#FFF8F8', font=('Arial', 14, 'bold'), fg='#333')
        self.goal_frame.pack(side='left', padx=20, pady=10)
        self.goal_labels = []
    def draw_beliefs(self, beliefs):
        for widget in self.belief_frame.winfo_children():
            widget.destroy()
        for idx, state in enumerate(beliefs):
            f = Frame(self.belief_frame, bd=2, relief='ridge', bg='#FFF8F8')
            f.grid(row=0, column=idx, padx=8)
            for i, v in enumerate(state):
                r, c = divmod(i, GRID_SIZE)
                lbl = tk.Label(
                    f,
                    text=str(v) if v else ' ',
                    width=3,
                    height=2,
                    font=('Arial', 14),
                    bg='#E0F7FA',
                    relief='ridge'
                )
                lbl.grid(row=r, column=c, padx=2, pady=2)

    def update_belief_states(self, beliefs, observation):
        return [state for state in beliefs if all(state[pos] == val for pos, val in observation.items())]
    def get_next_states(self, state):
        blank_idx = state.index(0)
        r, c = divmod(blank_idx, GRID_SIZE)
        next_states = []
        for move_name, move in MOVES.items():
            new_idx = blank_idx + move
            new_r, new_c = divmod(new_idx, GRID_SIZE)
            if move_name == 'up' and r == 0:
                continue
            if move_name == 'down' and r == GRID_SIZE - 1:
                continue
            if move_name == 'left' and c == 0:
                continue
            if move_name == 'right' and c == GRID_SIZE - 1:
                continue
            if 0 <= new_idx < GRID_SIZE * GRID_SIZE:
                new_state = list(state)
                new_state[blank_idx], new_state[new_idx] = new_state[new_idx], new_state[blank_idx]
                next_states.append(tuple(new_state))
        return next_states

    def belief_state_search(self):
        start_time = time.time()
        iteration_count = 0
        current_beliefs = self.update_belief_states(INITIAL_BELIEF_STATES, OBSERVATION)
        if current_beliefs:
            return current_beliefs[0], start_time

        self.root.after(0, lambda: self.step_lbl.config(text=f"Belief Steps: {iteration_count} / 10"))
        time.sleep(0.5)

        queue = deque([(INITIAL_BELIEF_STATES, [])])
        visited = set()
        while queue:
            beliefs, actions = queue.popleft()
            iteration_count += 1
            for state in beliefs:
                if any(all(state[pos] == val for pos, val in OBSERVATION.items()) for val in GOAL_CANDIDATES):
                    return state, start_time

            new_beliefs = set()
            for state in beliefs:
                for nxt in self.get_next_states(state):
                    new_beliefs.add(nxt)

            new_beliefs = list(new_beliefs)
            new_beliefs = self.update_belief_states(new_beliefs, OBSERVATION)
            if not new_beliefs:
                continue
            belief_tuple = tuple(sorted(new_beliefs))
            if belief_tuple in visited:
                continue
            visited.add(belief_tuple)
            queue.append((new_beliefs, actions + [f"Step {iteration_count}"]))
            self.root.after(0, lambda b=new_beliefs: self.draw_beliefs(b))
            self.root.after(0, lambda: self.step_lbl.config(text=f"Belief Steps: {iteration_count}"))
            time.sleep(0.1)
        return None, start_time

    def start_search(self):
        self.status_lbl.config(text="")
        self.run_btn.config(state='disabled')
        threading.Thread(target=self.run_search).start()

    def run_search(self):
        result, start_time = self.belief_state_search()
        elapsed = time.time() - start_time
        self.root.after(0, lambda: self.time_lbl.config(text=f"Time: {elapsed:.2f}s"))
        if result:
            self.root.after(1000, lambda: self.draw_beliefs([result]))
            self.root.after(2000, lambda: self.step_lbl.config(text="Belief Steps: Final"))
            self.root.after(3000, lambda: self.show_goal_candidates())
            self.root.after(4000, lambda: self.status_lbl.config(text="Solution Found! Goal(s) Revealed."))
        else:
            self.root.after(0, lambda: self.status_lbl.config(text="No solution found."))
        self.root.after(2000, lambda: self.run_btn.config(state='normal'))

    def show_goal_candidates(self):
        for widget in self.goal_frame.winfo_children():
            widget.destroy()
        self.goal_frame.pack(side='left', padx=10, pady=10)
        for i in range(min(3, len(GOAL_CANDIDATES))):
            frame = Frame(self.goal_frame, bd=2, relief='groove', bg='#FFF8F8')
            frame.grid(row=0, column=i, padx=8)
            for j, val in enumerate(GOAL_CANDIDATES[i]):
                r, c = divmod(j, GRID_SIZE)
                lbl = tk.Label(frame, text=str(val), width=4, height=2,
                               font=('Arial', 14), fg='#333', bg='#FFFFFF')
                lbl.grid(row=r, column=c, padx=3, pady=3)
            self.goal_labels.append(frame)

