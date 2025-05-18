import threading
import time
import ttkbootstrap as tb
from ttkbootstrap import Style
from tkinter import Toplevel
from backtracking import Backtracking
from puzzlebasic import PuzzleApp
from ac3Search import CSPDemoApp
from partialObs import PartialObsApp

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Menu Algorithms")
        self.root.geometry("1500x1000")

        # Khởi theme và config chung
        self.style = Style(theme="flatly")
        # Nền chính giữ nguyên
        self.root.configure(bg='#FFF8F8')
        # Cấu hình font chung cho button (lớn hơn & đậm hơn)
        self.style.configure('Custom.TButton', font=('Helvetica', 13, 'bold'))
        # Đặt màu cho Custom.TButton: nền trắng, viền & chữ hồng
        self.style.configure('Custom.TButton',
                             background='white',
                             foreground='#FFB6C1',
                             bordercolor='#FFB6C1',
                             focusthickness=0,
                             focuscolor='white')
        # Hover/active effect: nền hồng, chữ trắng, viền hồng
        self.style.map('Custom.TButton',
                       background=[('active', '#FFB6C1'), ('hover', '#FFB6C1')],
                       foreground=[('active', 'white'), ('hover', 'white')],
                       bordercolor=[('active', '#FF9999'), ('hover', '#FF9999')])

        self.create_menu()

    def create_menu(self):
        header = tb.Label(
            self.root,
            text="Choose an Algorithm",
            font=("Helvetica", 24, "bold"),
            bootstyle="primary",
            padding=20,
            foreground='#333333',
            background='#FFF8F8'
        )
        header.pack(pady=20)

        menu_frame = tb.Frame(self.root, bootstyle="secondary", padding=20)
        menu_frame.pack(fill=tb.BOTH, expand=True, padx=30, pady=10)

        algo_options = [
            "BFS", "DFS", "UCS", "IDS", "A*", "IDA*", "Greedy",
            "Simple Hill Climbing", "Steepest Hill Climbing", "Stochastic Hill Climbing",
            "Simulated Annealing", "Beam Search", "Genetic Algorithm",
            "Backtracking", "AC-3", "AND-OR Graph Search",
            "Q-Learning", "Search with Partial Ob"
        ]

        cols = 3
        for idx, algo in enumerate(algo_options):
            row = idx // cols
            col = idx % cols
            btn = tb.Button(
                menu_frame,
                text=algo,
                width=25,
                style='Custom.TButton',
                cursor='hand2',
                command=lambda a=algo: self.open_puzzle(a)
            )
            btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Căn đều grid
        for c in range(cols):
            menu_frame.columnconfigure(c, weight=1)
        total_rows = (len(algo_options) + cols - 1) // cols
        for r in range(total_rows):
            menu_frame.rowconfigure(r, weight=1)

    def open_puzzle(self, algo):
        self.root.withdraw()
        puzzle_root = Toplevel(self.root)
        puzzle_root.protocol("WM_DELETE_WINDOW", lambda: self.on_puzzle_close(puzzle_root))
        if algo == "Backtracking":
            Backtracking(puzzle_root)
        elif algo == "AC-3":
            CSPDemoApp(puzzle_root)
        elif algo == "Search with Partial Ob":
            PartialObsApp(puzzle_root)
        else:
            PuzzleApp(puzzle_root, algo, self)

    def on_puzzle_close(self, puzzle_root):
        puzzle_root.destroy()
        self.root.deiconify()

if __name__ == "__main__":
    app = tb.Window(themename="flatly")
    MainMenu(app)
    app.mainloop()