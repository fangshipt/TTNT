import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame
from collections import deque
import threading
import time
from itertools import permutations
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MOVES = {'up': -3, 'down': 3, 'left': -1, 'right': 1}
GRID_SIZE = 3
MAX_BELIEF_DISPLAY = 10

INITIAL_BELIEF_STATES = random.sample(list(permutations(range(9))), 100) 
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
class NoObservationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Belief State Search With No Observation")
        self.root.geometry("1200x850")
        self.root.configure(bg='#FFF8F8')
        self.step_times = []
        self.build_ui()

    def build_ui(self):
        ctrl = tk.Frame(self.root, bg='#FFF8F8')
        ctrl.pack(fill='x', pady=8)      
        self.run_btn = tk.Button(ctrl, text="Run", command=self.start_search,
                               font=('Arial', 12, 'bold'), bg='#FFB6C1')
        self.run_btn.pack(side='left', padx=8)
        self.time_lbl = tk.Label(ctrl, text="Total Time: 0.00s", bg='#FFF8F8', font=('Arial', 12))
        self.time_lbl.pack(side='left', padx=12)
        self.step_lbl = tk.Label(ctrl, text="Belief Steps: 0", bg='#FFF8F8', font=('Arial', 12))
        self.step_lbl.pack(side='left', padx=12)
        self.status_lbl = tk.Label(ctrl, text="", bg='#FFF8F8', font=('Arial', 12, 'italic'), fg='#D32F2F')
        self.status_lbl.pack(side='left', padx=12)
        self.step_time_lbl = tk.Label(ctrl, text="Step Time: 0.00s", bg='#FFF8F8', font=('Arial', 12))
        self.step_time_lbl.pack(side='left', padx=12)
        canvas = Canvas(self.root, height=180, bg='#FFF8F8', highlightthickness=0)
        hbar = Scrollbar(self.root, orient='horizontal', command=canvas.xview)
        canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side='bottom', fill='x')
        canvas.pack(side='top', fill='x', padx=10)
        self.belief_frame = Frame(canvas, bg='#FFF8F8')
        canvas.create_window((0, 0), window=self.belief_frame, anchor='nw')
        self.draw_beliefs(INITIAL_BELIEF_STATES[:MAX_BELIEF_DISPLAY])  
        self.belief_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))

    def draw_beliefs(self, beliefs):
        for widget in self.belief_frame.winfo_children():
            widget.destroy()
        for idx, state in enumerate(beliefs[:MAX_BELIEF_DISPLAY]):
            f = Frame(self.belief_frame, bd=2, relief='ridge', bg='#FFF8F8')
            f.grid(row=0, column=idx, padx=8)
            for i, v in enumerate(state):
                r, c = divmod(i, GRID_SIZE)
                lbl = tk.Label(
                    f, text=str(v) if v else ' ',
                    width=3, height=2,
                    font=('Arial', 14),
                    bg='#E1F5FE',
                    relief='ridge'
                )
                lbl.grid(row=r, column=c, padx=2, pady=2)

    def get_next_states(self, state):
        try:
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
        except Exception as e:
            logger.error(f"Error in get_next_states: {e}")
            return []

    def belief_state_search(self):
        start_time = time.time()
        iteration_count = 0
        queue = deque([(INITIAL_BELIEF_STATES, [])])
        visited = set()
        
        logger.info("Starting belief state search")
        logger.info(f"Initial belief states count: {len(INITIAL_BELIEF_STATES)}")

        if GOAL_STATE in INITIAL_BELIEF_STATES:
            logger.info("Goal state found in initial beliefs, proceeding to simulate steps")
            self.status_lbl.config(text="Goal in initial states, simulating...")
        
        while queue:
            step_start = time.time()
            iteration_count += 1
            
            beliefs, actions = queue.popleft()
            
            self.root.after(0, lambda b=beliefs[:MAX_BELIEF_DISPLAY]: self.draw_beliefs(b))
            self.root.after(0, lambda: self.step_lbl.config(text=f"Belief Steps: {iteration_count}"))
            
            if GOAL_STATE in beliefs and iteration_count > 1:  
                step_time = time.time() - step_start
                self.step_times.append(step_time)
                logger.info(f"Goal found at step {iteration_count}, step time: {step_time:.2f}s")
                return beliefs, actions, start_time

            new_beliefs = set()
            for state in beliefs:
                next_states = self.get_next_states(state)
                new_beliefs.update(next_states)

            new_beliefs = list(new_beliefs)
            belief_tuple = tuple(sorted(new_beliefs))
            
            if belief_tuple in visited:
                continue
            visited.add(belief_tuple)

            queue.append((new_beliefs, actions + [f"Step {iteration_count}"]))
            
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            logger.info(f"Step {iteration_count} completed, states: {len(new_beliefs)}, time: {step_time:.2f}s")
            
            self.root.after(0, lambda: self.step_time_lbl.config(text=f"Step Time: {step_time:.2f}s"))
            self.root.update()  
            
            time.sleep(0.01)  

        logger.warning("No solution found")
        return None, [], start_time

    def start_search(self):
        try:
            self.status_lbl.config(text="Searching...")
            self.run_btn.config(state='disabled')
            self.step_times = []
            threading.Thread(target=self.run_search, daemon=True).start()
        except Exception as e:
            logger.error(f"Error starting search: {e}")
            self.status_lbl.config(text="Error occurred!")

    def run_search(self):
        try:
            result, actions, start_time = self.belief_state_search()
            elapsed = time.time() - start_time
            
            self.root.after(0, lambda: self.time_lbl.config(text=f"Total Time: {elapsed:.2f}s"))
            if result:
                self.root.after(0, lambda: self.draw_beliefs([GOAL_STATE]))
                self.root.after(0, lambda: self.status_lbl.config(text=f"Solution Found! Steps: {len(actions)}"))
                logger.info(f"Search completed. Total time: {elapsed:.2f}s")
                logger.info(f"Total steps: {len(actions)}")
                logger.info(f"Average step time: {sum(self.step_times)/len(self.step_times) if self.step_times else 0:.2f}s")
            else:
                self.root.after(0, lambda: self.status_lbl.config(text="No solution found"))
                
        except Exception as e:
            logger.error(f"Error in run_search: {e}")
            self.root.after(0, lambda: self.status_lbl.config(text="Search failed"))
        
        finally:
            self.root.after(0, lambda: self.run_btn.config(state='normal'))

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = NoObservationApp(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")