import tkinter as tk
from backtracking import Backtracking
from puzzleBasic import PuzzleApp
from ac3Search import AC3CSP
from partialObs import PartialObsApp
from noObs import NoObservationApp
from andOr import ANDORGraphSearch

# Lớp Menu chính
class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Menu")
        window_width = 1000
        window_height = 700
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#FFF8F8')
        self.create_menu()
    def create_menu(self):
        tk.Label(self.root, text="Choose an Algorithm", bg='#FFF8F8',
                 font=('Arial', 18, 'bold')).pack(pady=20)
        menu_frame = tk.Frame(self.root, bg='#FFF8F8')
        menu_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        algo_options = [
            "BFS", "DFS", "UCS", "IDS", "A*", "IDA*", "Greedy",
            "Simple Hill Climbing", "Steepest Hill Climbing", "Stochastic Hill Climbing",
            "Simulated Annealing", "Beam Search", "Genetic Algorithm",
            "Backtracking", "AC-3", "AND-OR Graph Search",
            "Q‑Learning", "Search with Partial Ob", "Search with No Ob"
        ]
        cols = 5
        for idx, algo in enumerate(algo_options):
            row = idx // cols
            col = idx % cols
            btn = tk.Button(
                menu_frame,
                text=algo,
                font=('Arial', 12, 'bold'),
                width=12,
                height=1,
                bg='#FFB6C1',
                fg='black',
                relief='flat',
                borderwidth=0,
                activebackground='#FF9999',
                cursor='hand2'
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            btn.configure(
                highlightthickness=0,
                border=0,
                highlightbackground='#FFF8F8'
            )
            btn.bind('<Enter>', lambda e, b=btn: b.configure(bg='#FF9999'))
            btn.bind('<Leave>', lambda e, b=btn: b.configure(bg='#FFD1DC'))
            btn.configure(command=lambda a=algo: self.open_puzzle(a))
        for c in range(cols):
            menu_frame.grid_columnconfigure(c, weight=1)
        total_rows = (len(algo_options) + cols - 1) // cols
        for r in range(total_rows):
            menu_frame.grid_rowconfigure(r, weight=1)
    def open_puzzle(self, algo):
        self.root.withdraw()
        puzzle_root = tk.Toplevel()
        puzzle_root.protocol("WM_DELETE_WINDOW", lambda: self.on_puzzle_close(puzzle_root))
        if algo == "Backtracking":
            Backtracking(puzzle_root)
        elif algo == "AC-3":
            AC3CSP(puzzle_root)
        elif algo == "Search with Partial Ob":
            PartialObsApp(puzzle_root)
        elif algo == "Search with No Ob":
            NoObservationApp(puzzle_root)
        elif algo == "AND-OR Graph Search":
            ANDORGraphSearch(puzzle_root)
        else:
            PuzzleApp(puzzle_root, algo, self)

    def on_puzzle_close(self, puzzle_root):
        puzzle_root.destroy()
        self.root.deiconify()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()