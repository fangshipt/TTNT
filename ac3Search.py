import tkinter as tk
import threading
import time

class AC3Search:
    def __init__(self, domains, order=None):
        self.domains = domains
        self.order = order or list(range(len(domains)))

    def get_constraint(self, Vi, Vj):
        return lambda a, b: a != b

    def run(self, update_callback=None, stop_flag=lambda: False, delay=0.1):
        arcs = []
        for i in range(len(self.order)):
            for j in range(i + 1, len(self.order)):
                vi = self.order[i]
                vj = self.order[j]
                arcs.append((vi, vj, self.get_constraint(vi, vj)))
                arcs.append((vj, vi, self.get_constraint(vj, vi)))

        queue = arcs.copy()
        while queue and not stop_flag():
            Xi, Xj, constraint = queue.pop(0)
            Di = self.domains[Xi]
            Dj = self.domains[Xj]
            to_remove = [x for x in Di if not any(constraint(x, y) for y in Dj)]
            if to_remove:
                for x in to_remove:
                    Di.remove(x)
                for Xk in range(len(self.domains)):
                    if Xk != Xi:
                        queue.append((Xk, Xi, self.get_constraint(Xk, Xi)))
            if update_callback:
                update_callback(self.domains, (Xi, Xj))
            time.sleep(delay)
        return self.domains

class Backtracking:
    def __init__(self, domains, goal=None, order=None):
        self.domains = domains
        self.order = order or list(range(len(domains)))
        self.goal = goal
        self.solution = None
        self.stop_flag = False

    def backtrack(self, partial, update_callback=None, delay=0.1):
        if self.stop_flag:
            return None
        if update_callback:
            update_callback(partial, None)
        time.sleep(delay)
        if len(partial) == len(self.domains):
            if not self.goal or partial == self.goal:
                self.solution = list(partial)
            return self.solution
        var_idx = len(partial)
        var = self.order[var_idx]
        for val in sorted(self.domains[var]):
            if self.stop_flag:
                return None
            if val in partial:
                continue
            result = self.backtrack(partial + [val], update_callback, delay)
            if result:
                return result
        return None

class AC3CSP:
    def __init__(self, root, main_menu=None):
        self.root = root
        self.main_menu = main_menu
        self.root.title("8-Puzzle CSP Demo")
        self.root.geometry("1200x850")
        self.root.configure(bg='#FFF8F8')

        self.order = [0, 1, 2, 4, 8, 3, 5, 6, 7]
        self.goal = list(range(9))
        self.stop_flag = False
        self.domains = [set(range(9)) for _ in range(9)]

        self.setup_ui()

    def setup_ui(self):
        domain_frame = tk.LabelFrame(self.root, text="Domains", font=('Arial',16,'bold'), bg='#FFF8F8')
        domain_frame.pack(pady=10)
        self.domain_labels = []
        for i in range(9):
            lbl = tk.Label(domain_frame,
                           text=','.join(map(str, sorted(self.domains[i]))),
                           width=14, font=('Arial',14), relief='solid', bg='#FFB6C1')
            lbl.grid(row=i//3, column=i%3, padx=10, pady=10)
            self.domain_labels.append(lbl)

        bt_frame = tk.LabelFrame(self.root, text="Backtracking", font=('Arial',16,'bold'), bg='#FFF8F8')
        bt_frame.pack(pady=10)
        self.curr_labels = []
        for i in range(9):
            lbl = tk.Label(bt_frame, text='-', width=6, font=('Arial',18,'bold'), relief='solid', bg='#FFB6C1')
            lbl.grid(row=i//3, column=i%3, padx=10, pady=10)
            self.curr_labels.append(lbl)

        ctrl_frame = tk.Frame(self.root, bg='#FFF8F8')
        ctrl_frame.pack(pady=15)
        btn_font = ('Arial',14,'bold')
        btn_kwargs = {'font': btn_font, 'bg':'#FFB6C1', 'width':15}
        tk.Button(ctrl_frame, text="AC-3 + Backtrack", command=self.run_all, **btn_kwargs).pack(side='left', padx=10)
        tk.Button(ctrl_frame, text="Stop", command=self.stop, **btn_kwargs).pack(side='left', padx=10)
        tk.Button(ctrl_frame, text="Exit to Menu", command=self.exit_to_menu, **btn_kwargs).pack(side='left', padx=10)

        self.status = tk.Label(self.root, text="Idle", font=('Arial',14), bg='#FFF8F8')
        self.status.pack(pady=10)
        self.time_label = tk.Label(self.root, text="Time: -", font=('Arial',14), bg='#FFF8F8')
        self.time_label.pack(pady=10)

    def update_ac3(self, domains, arc):
        for i, d in enumerate(domains):
            self.domain_labels[i].config(text=','.join(map(str, sorted(d))))
        self.status.config(text=f"AC-3 processing arc {arc}")
        self.root.update_idletasks()

    def update_bt(self, partial, _):
        for i, lbl in enumerate(self.curr_labels):
            lbl.config(text=str(partial[i]) if i < len(partial) else '-')
        self.status.config(text=f"Backtracking: {partial}")
        self.root.update_idletasks()

    def run_all(self):
        self.stop_flag = False
        self.domains = [set(range(9)) for _ in range(9)]
        for lbl in self.domain_labels:
            lbl.config(text=','.join(map(str, sorted(set(range(9))))))
        for lbl in self.curr_labels:
            lbl.config(text='-')
        self.time_label.config(text="Time: -")

        ac3 = AC3Search(self.domains, self.order)
        bt = Backtracking(self.domains, self.goal, self.order)

        def task():
            start = time.time()
            ac3.run(update_callback=self.update_ac3, stop_flag=lambda: self.stop_flag, delay=0.1)
            if not self.stop_flag:
                bt.backtrack([], update_callback=self.update_bt, delay=0.1)
            end = time.time()
            self.time_label.config(text=f"Time: {end - start:.2f} seconds")

        threading.Thread(target=task).start()

    def stop(self):
        self.stop_flag = True

    def exit_to_menu(self):
        self.root.destroy()
        if self.main_menu:
            try:
                self.main_menu.deiconify()
            except Exception:
                pass

