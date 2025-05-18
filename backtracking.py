import tkinter as tk
import threading
import time
from algorithm import *


class Backtracking:
    def __init__(self, root):
        self.root = root
        self.root.title("Backtracking Demo")
        self.root.geometry("1000x750")
        self.root.configure(bg='#FFF8F8')  
        self.stop_flag = False
        self.solution_found = False
        self.start_time = 0
        self.goal = list(range(9))  # [0,1,2,3,4,5,6,7,8]

        self.frame_curr = tk.LabelFrame(root, text="Current", font=('Arial', 16, 'bold'), bg='#FFF8F8', fg='black')
        self.frame_curr.pack(pady=15)
        self.curr_buttons = []
        for i in range(9):
            btn = tk.Label(self.frame_curr, text='-', width=6, height=2,
                           font=('Arial', 18, 'bold'), relief='solid', bg='#FFB6C1', fg='black')
            btn.grid(row=i//3, column=i%3, padx=10, pady=10)
            self.curr_buttons.append(btn)

        self.frame_goal = tk.LabelFrame(root, text="Goal State", font=('Arial', 16, 'bold'), bg='#FFF8F8', fg='black')
        self.frame_goal.pack(pady=15)
        for i, val in enumerate(self.goal):
            lbl = tk.Label(self.frame_goal, text=str(val), width=6, height=2,
                           font=('Arial', 18, 'bold'), relief='solid', bg='#FFB6C1', fg='black')
            lbl.grid(row=i//3, column=i%3, padx=10, pady=10)

        ctrl = tk.Frame(root, bg='#FFF8F8')
        ctrl.pack(pady=15)
        btn_font = ('Arial', 14, 'bold')
        btn_kwargs = {'font': btn_font, 'bg': '#FFB6C1', 'fg': 'black', 'width': 15}
        tk.Button(ctrl, text="Start Solving", command=self.start, **btn_kwargs).pack(side='left', padx=10)
        tk.Button(ctrl, text="Stop", command=self.stop, **btn_kwargs).pack(side='left', padx=10)
        tk.Button(ctrl, text="Close", command=root.destroy, **btn_kwargs).pack(side='left', padx=10)

        self.status_label = tk.Label(root, text="Idle", font=('Arial', 14), bg='#FFF8F8', fg='black')
        self.status_label.pack(pady=10)

    def update_current(self, state, failed_idx=None):
        for i, v in enumerate(state):
            self.curr_buttons[i].config(text=str(v))
            if i == failed_idx:
                self.curr_buttons[i].config(bg='red')
            else:
                self.curr_buttons[i].config(bg='#FFB6C1')
        self.root.update_idletasks()

    def start(self):
        self.stop_flag = False
        self.solution_found = False
        self.start_time = time.time()
        threading.Thread(target=self.backtrack, args=([],)).start()

    def stop(self):
        self.stop_flag = True

    def backtrack(self, partial):
        if self.stop_flag:
            return
        self.update_current(partial + ['-']*(9-len(partial)))
        self.status_label.config(text=f"Try {partial}")
        time.sleep(0.5)
        if len(partial) == 9:
            if partial == self.goal:
                self.solution_found = True
                self.update_current(partial)
                elapsed = time.time() - self.start_time
                self.status_label.config(text=f"Goal found in {elapsed:.4f} seconds!")
            else:
                self.status_label.config(text="Not goal")
            return

        order = [0, 1, 2, 4, 8, 3, 5, 6, 7]
        for num in order:
            if self.stop_flag or self.solution_found:
                return
            if num in partial:
                continue
            if partial and num <= partial[-1]:
                self.update_current(partial + [num] + ['-']*(8-len(partial)), failed_idx=len(partial))
                time.sleep(0.1)
                continue
            self.backtrack(partial + [num])
