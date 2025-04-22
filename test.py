import tkinter as tk
from tkinter import messagebox
from collections import deque
import heapq
import itertools
import random
import math

# --------------------- Helper Functions ---------------------
MOVES = {
    'UP': (-1, 0), 'DOWN': (1, 0),
    'LEFT': (0, -1), 'RIGHT': (0, 1)
}

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None


def generate_states(state):
    x, y = find_blank(state)
    for move, (dx, dy) in MOVES.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            yield new_state, move


def manhattan_distance(state, goal=None):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                gi, gj = ((val - 1) // 3, (val - 1) % 3)
                dist += abs(i - gi) + abs(j - gj)
    return dist

# --------------------- Search Algorithms ---------------------

def bfs_solve(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = {tuple(map(tuple, start_state))}
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path
        for nxt, mv in generate_states(state):
            t = tuple(map(tuple, nxt))
            if t not in visited:
                visited.add(t)
                queue.append((nxt, path + [mv]))
    return None


def ucs_solve(start_state, goal_state):
    pq = []
    heapq.heappush(pq, (0, start_state, []))
    visited = {tuple(map(tuple, start_state)): 0}
    while pq:
        cost, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for nxt, mv in generate_states(state):
            new_cost = cost + 1
            t = tuple(map(tuple, nxt))
            if t not in visited or new_cost < visited[t]:
                visited[t] = new_cost
                heapq.heappush(pq, (new_cost, nxt, path + [mv]))
    return None


def greedy_solve(start_state, goal_state):
    pq = []
    heapq.heappush(pq, (manhattan_distance(start_state), start_state, []))
    visited = {tuple(map(tuple, start_state))}
    while pq:
        h, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for nxt, mv in generate_states(state):
            t = tuple(map(tuple, nxt))
            if t not in visited:
                visited.add(t)
                heapq.heappush(pq, (manhattan_distance(nxt), nxt, path + [mv]))
    return None


def astar_solve(start_state, goal_state):
    pq = []
    start_t = tuple(map(tuple, start_state))
    heapq.heappush(pq, (manhattan_distance(start_state), 0, start_state, []))
    best = {start_t: 0}
    while pq:
        f, g, state, path = heapq.heappop(pq)
        if state == goal_state:
            return path
        for nxt, mv in generate_states(state):
            ng = g + 1
            t = tuple(map(tuple, nxt))
            if t not in best or ng < best[t]:
                best[t] = ng
                heapq.heappush(pq, (ng + manhattan_distance(nxt), ng, nxt, path + [mv]))
    return None


def dfs_solve(start_state, goal_state):
    solution = None
    visited = set()
    def dfs(state, path):
        nonlocal solution
        if state == goal_state:
            solution = path
            return True
        visited.add(tuple(map(tuple, state)))
        for nxt, mv in generate_states(state):
            if tuple(map(tuple, nxt)) not in visited:
                if dfs(nxt, path + [mv]):
                    return True
        return False
    dfs(start_state, [])
    return solution


def ids_solve(start_state, goal_state, max_depth=50):
    def dls(state, path, depth, visited):
        if state == goal_state:
            return path
        if depth == 0:
            return None
        visited.add(tuple(map(tuple, state)))
        for nxt, mv in generate_states(state):
            if tuple(map(tuple, nxt)) not in visited:
                res = dls(nxt, path + [mv], depth-1, visited)
                if res is not None:
                    return res
        visited.remove(tuple(map(tuple, state)))
        return None
    for depth in range(max_depth+1):
        res = dls(start_state, [], depth, set())
        if res is not None:
            return res
    return None


def ida_star_solve(start_state, goal_state):
    solution = None
    bound = manhattan_distance(start_state)
    def search(path, g, bound, moves):
        nonlocal solution
        state = path[-1]
        f = g + manhattan_distance(state)
        if f > bound:
            return f
        if state == goal_state:
            solution = moves
            return -1
        min_t = float('inf')
        for nxt, mv in generate_states(state):
            if nxt not in path:
                path.append(nxt)
                t = search(path, g+1, bound, moves+[mv])
                if t == -1:
                    return -1
                if t < min_t:
                    min_t = t
                path.pop()
        return min_t
    path = [start_state]
    while True:
        t = search(path, 0, bound, [])
        if t == -1:
            return solution
        if t == float('inf'):
            return None
        bound = t


def simple_hill_climbing_solve(start_state, goal_state):
    current = [row[:] for row in start_state]
    cur_h = manhattan_distance(current)
    path = []
    while True:
        neighbors = list(generate_states(current))
        better = [(n,h) for n, m in neighbors for h in [manhattan_distance(n)] if h < cur_h]
        if not better:
            return path if current==goal_state else None
        next_state, _ = min(better, key=lambda x: x[1])
        move = [mv for ns,mv in neighbors if ns==next_state][0]
        current = next_state
        cur_h = manhattan_distance(current)
        path.append(move)


def steepest_hill_climbing_solve(start_state, goal_state):
    current = [row[:] for row in start_state]
    path = []
    while True:
        neighbors = [(n,m,manhattan_distance(n)) for n,m in generate_states(current)]
        best = min(neighbors, key=lambda x: x[2])
        if best[2] >= manhattan_distance(current):
            return path if current==goal_state else None
        current = best[0]; path.append(best[1])


def stochastic_hill_climbing_solve(start_state, goal_state, max_restarts=50, max_steps=1000):
    for _ in range(max_restarts):
        current = [row[:] for row in start_state]
        path = []
        for _ in range(max_steps):
            if current == goal_state:
                return path
            better = [(n,m) for n,m in generate_states(current) if manhattan_distance(n)<manhattan_distance(current)]
            if not better: break
            nxt, mv = random.choice(better)
            current = nxt; path.append(mv)
    return None


def simulated_annealing_solve(start_state, goal_state, T=1e5, cooling=0.95, max_steps=10000):
    current = [row[:] for row in start_state]
    path = []
    temp = T
    for _ in range(max_steps):
        if current==goal_state: return path
        nxt, mv = random.choice(list(generate_states(current)))
        delta = manhattan_distance(nxt)-manhattan_distance(current)
        if delta<0 or random.random()<math.exp(-delta/temp):
            current = nxt; path.append(mv)
        temp *= cooling
        if temp<1: break
    return None


def beam_search_solve(start_state, goal_state, beam_width=3):
    beam = [(start_state, [])]
    visited = {tuple(map(tuple,start_state))}
    while beam:
        new_beam = []
        for state,path in beam:
            if state==goal_state: return path
            for nxt,mv in generate_states(state):
                t=tuple(map(tuple,nxt))
                if t not in visited:
                    visited.add(t)
                    new_beam.append((nxt,path+[mv]))
        new_beam.sort(key=lambda x: manhattan_distance(x[0]))
        beam = new_beam[:beam_width]
    return None


def genetic_algorithm_solve(start_state, goal_state, pop_size=200, chrom_len=30, max_gen=1000, mut_rate=0.1):
    moves_list = list(MOVES.keys())
    def apply_moves(state, seq):
        s=[row[:] for row in state]
        for mv in seq:
            x,y=find_blank(s)
            dx,dy=MOVES[mv]
            nx,ny=x+dx,y+dy
            if 0<=nx<3:
                s[x][y],s[nx][ny]=s[nx][ny],s[x][y]
        return s
    def fitness(seq): return manhattan_distance(apply_moves(start_state,seq))
    pop=[[random.choice(moves_list) for _ in range(chrom_len)] for _ in range(pop_size)]
    for gen in range(max_gen):
        for chrom in pop:
            if fitness(chrom)==0:
                path=[]; s=[row[:] for row in start_state]
                for mv in chrom:
                    for nxt,mv2 in generate_states(s):
                        if mv2==mv:
                            s=nxt; path.append(mv); break
                    if s==goal_state: return path
        pop2=[]
        def tour(): return min(random.sample(pop,3), key=fitness)
        while len(pop2)<pop_size:
            p1,p2=tour(),tour()
            pt=random.randint(1,chrom_len-1)
            child=p1[:pt]+p2[pt:]
            child=[random.choice(moves_list) if random.random()<mut_rate else g for g in child]
            pop2.append(child)
        pop=pop2
    return None


def and_or_solve(start_state, goal_state, limit=50):
    def ao(state, explored, depth):
        if state==goal_state: return []
        if depth>limit or tuple(map(tuple,state)) in explored: return None
        explored_new=explored|{tuple(map(tuple,state))}
        for nxt,mv in generate_states(state):
            res=ao(nxt,explored_new,depth+1)
            if res is not None: return [mv]+res
        return None
    return ao(start_state,set(),0)


def sensorless_bfs_solve(goal_state):
    all_states=[list(map(list,p)) for p in itertools.permutations(range(9))]
    belief=frozenset(tuple(map(tuple,s)) for s in all_states)
    queue=deque([(belief,[])])
    visited={belief}
    while queue:
        b,path=queue.popleft()
        if len(b)==1 and list(b)[0]==tuple(map(tuple,goal_state)): return path
        for mv in MOVES:
            new=set()
            for st in b:
                s=[list(r) for r in st]
                nx=apply_action_sensorless(s,mv)
                if nx: new.add(tuple(map(tuple,nx)))
            if new:
                nb=frozenset(new)
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb,path+[mv]))
    return None

def apply_action_sensorless(state, move):
    x,y=find_blank(state)
    dx,dy=MOVES[move]
    if 0<=x+dx<3 and 0<=y+dy<3:
        s=[row[:] for row in state]
        s[x][y],s[x+dx][y+dy]=s[x+dx][y+dy],s[x][y]
        return s
    return None


def general_problem_solver(start_state, goal_state):
    """
    Defines the 8-puzzle problem using standard components
    (actions, transition, goal_test, step_cost) and then
    calls an existing search algorithm (currently DFS) to solve it.
    """
    # 1. Define Actions function
    def actions(s):
        # Returns a list of possible move strings ('UP', 'DOWN', ...) from state s
        return [move for (_, move) in generate_states(s)]

    # 2. Define Transition Model function
    def transition(s, a):
        # Returns the state resulting from taking action a in state s
        # Note: generate_states returns [(new_state, move), ...]
        for new_s, move in generate_states(s):
            if move == a:
                return new_s
        return None # Should not happen if 'a' came from actions(s)

# Map algorithm names to functions
def get_solver(name):
    return {
        'BFS': bfs_solve,
        'UCS': ucs_solve,
        'Greedy': greedy_solve,
        'A*': astar_solve,
        'DFS': dfs_solve,
        'IDS': ids_solve,
        'IDA*': ida_star_solve,
        'Simple Hill Climbing': simple_hill_climbing_solve,
        'Steepest Hill Climbing': steepest_hill_climbing_solve,
        'Stochastic Hill Climbing': stochastic_hill_climbing_solve,
        'Simulated Annealing': simulated_annealing_solve,
        'Beam Search': beam_search_solve,
        'Genetic Algorithm': genetic_algorithm_solve,
        'AND-OR Search': and_or_solve,
        'Sensorless BFS': sensorless_bfs_solve,
        'General': general_problem_solver
    }.get(name)

# --------------------- GUI Application ---------------------
class PuzzleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Thiết lập kích thước khung chính to hơn
        self.geometry('800x600')
        self.title('8-Puzzle Solver')
        self.start_state = [[2,6,5],[8,0,7],[4,3,1]]
        self.goal_state = [[1,2,3],[4,5,6],[7,8,0]]
        self._build_ui()

    def _build_ui(self):
        left = tk.Frame(self, padx=10, pady=10, bg='#fdeada', width=250)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)  # Ngăn việc tự động co lại theo nội dung

        tk.Label(left, text='Initial State', bg='#fdeada', font=('Arial', 12, 'bold')).pack()
        self.start_entries = self._make_grid(left, self.start_state)

        tk.Label(left, text='Goal State', bg='#fdeada', font=('Arial', 12, 'bold')).pack(pady=(10,0))
        self.goal_entries = self._make_grid(left, self.goal_state)

        tk.Button(left, text='Update States', bg='#f4cccc', command=self.update_states).pack(pady=5, fill=tk.X)

        tk.Label(left, text='Algorithm Selection', bg='#fdeada', font=('Arial', 12, 'bold')).pack(pady=(20,0))
        self.alg_var = tk.StringVar(value='BFS')
        choices = ['BFS','UCS','Greedy','A*','DFS','IDS','IDA*','Simple Hill Climbing',
                'Steepest Hill Climbing','Stochastic Hill Climbing','Simulated Annealing',
                'Beam Search','Genetic Algorithm','AND-OR Search','Sensorless BFS','General']
        tk.OptionMenu(left, self.alg_var, *choices).pack(fill=tk.X, pady=5)

        tk.Button(left, text='Solve', bg='#f4cccc', command=self.solve).pack(fill=tk.X, pady=(10,0))

        # Board display on the right
        right = tk.Frame(self, padx=10, pady=10, bg='#fdeada')
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right, text='8-Puzzle Board', font=('Arial', 16, 'bold'), bg='#fdeada')\
            .grid(row=0, column=0, columnspan=3)

        self.tile_labels = []
        for i in range(3):
            row = []
            for j in range(3):
                lbl = tk.Label(right, text='', font=('Arial', 24), width=4, height=2,
                            bg='#f4cccc', relief='raised')
                lbl.grid(row=i+1, column=j, padx=5, pady=5)
                row.append(lbl)
            self.tile_labels.append(row)
        self._update_board(self.start_state)

    def _make_grid(self, parent, data):
        entries = []
        grid = tk.Frame(parent, bg='#fdeada')
        grid.pack()
        for i in range(3):
            row = []
            for j in range(3):
                e = tk.Entry(grid, width=4, justify='center', bg='#fdeada', highlightbackground='#fdeada')
                e.grid(row=i, column=j, padx=2, pady=2)
                e.insert(0, str(data[i][j]))
                row.append(e)
            entries.append(row)
        return entries

    def update_states(self):
        try:
            self.start_state = [[int(e.get()) for e in row] for row in self.start_entries]
            self.goal_state = [[int(e.get()) for e in row] for row in self.goal_entries]
            self._update_board(self.start_state)
        except ValueError:
            messagebox.showerror('Error','Please enter valid integers 0-8')

    def _update_board(self, state):
        for i in range(3):
            for j in range(3):
                val=state[i][j]
                self.tile_labels[i][j]['text']='' if val==0 else str(val)

    def solve(self):
        name = self.alg_var.get()
        solver = get_solver(name)
        if not solver:
            messagebox.showerror('Error','Unsupported algorithm')
            return
        path = solver(self.start_state, self.goal_state)
        if path is None:
            messagebox.showinfo('Result','No solution found')
        else:
            # Bỏ showinfo ở đây, thay thành callback truyền vào animate
            self._animate_solution(path, on_complete=lambda:
                messagebox.showinfo('Result', f"Algorithm {name} completed in {len(path)} moves.")
            )

    def _animate_solution(self, moves, delay=300, on_complete=None):
        state = [row[:] for row in self.start_state]
        def step(idx):
            if idx >= len(moves):
                # Khi đã chạy hết moves thì gọi callback
                if on_complete:
                    on_complete()
                return
            mv = moves[idx]
            x, y = find_blank(state)
            dx, dy = MOVES[mv]
            nx, ny = x+dx, y+dy
            state[x][y], state[nx][ny] = state[nx][ny], state[x][y]
            self._update_board(state)
            # đệ quy với delay
            self.after(delay, lambda: step(idx+1))
        step(0)

if __name__=='__main__':
    app=PuzzleApp()
    app.mainloop()
