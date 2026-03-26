"""
Labyrinth RL Training Server
────────────────────────────
• Runs real Tabular Q-Learning and DQN training in Python
• Streams every episode's data live to the browser via WebSocket
• Serves the HTML dashboard on http://localhost:8765
• Zero external dependencies (pure stdlib + numpy)

Usage:  python server.py
Then open:  http://localhost:8765
"""

import sys, os, json, time, random, math, threading, struct, hashlib, base64
import socket, select, queue
from collections import deque, defaultdict
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler


# ═══════════════════════════════════════════════════════
#  MAZE ENVIRONMENT
# ═══════════════════════════════════════════════════════

TRAP_COUNTS = {"5x5": 1, "8x8": 3, "10x10": 5}
GRID_DIMS   = {"5x5": (5,5), "8x8": (8,8), "10x10": (10,10)}

def generate_maze(grid_size: str) -> np.ndarray:
    """
    Maze generator that works directly on the pixel grid (no logical/pixel
    distinction) so every cell — including borders and odd-indexed edges —
    can be either wall or passage.

    Algorithm: Random-walk flood (variant of recursive backtracker) where
    we start from a random free cell and repeatedly carve to unvisited
    neighbours, choosing randomly among all frontier cells (Prim-like).
    This produces full-grid randomness with no fixed borders.

    Guarantees
    ──────────
    • Start and goal placed in random quadrants (top-left / bottom-right)
    • Always BFS-solvable (retry if not)
    • Exactly n_traps traps placed OFF the shortest path
    • trap count scales with grid size
    """
    from collections import deque

    rows, cols = GRID_DIMS[grid_size]
    n_traps    = TRAP_COUNTS[grid_size]
    rng        = random.Random()   # fresh → different every call

    # ── 1. Start with ~40% random open cells, guarantee connectivity ────
    # We use a cleaner approach: random spanning tree via iterative DFS
    # on EVERY cell (not just even-indexed ones).

    # Begin: all cells are walls
    grid = np.ones((rows, cols), dtype=int)

    # Pick a random starting pixel, open it
    start_r = rng.randint(0, rows-1)
    start_c = rng.randint(0, cols-1)
    grid[start_r, start_c] = 0

    # Stack-based DFS; each step opens a neighbour 1 step away
    # (not 2), so there's no parity constraint
    stack = [(start_r, start_c)]
    visited = {(start_r, start_c)}

    while stack:
        r, c = stack[-1]
        # find unvisited neighbours (1 step, all 4 directions)
        nbrs = []
        for dr, dc in rng.sample([(-1,0),(1,0),(0,-1),(0,1)], 4):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in visited:
                nbrs.append((nr,nc))
        if nbrs:
            nr, nc = rng.choice(nbrs)
            grid[nr, nc] = 0
            visited.add((nr,nc))
            stack.append((nr,nc))
        else:
            stack.pop()

    # ── 2. Randomly wall back ~30% of cells to create obstacles ─────────
    # Keep a guaranteed open path by only walling cells that are not
    # articulation points (i.e. removing them doesn't disconnect the maze).
    free_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r,c]==0]
    rng.shuffle(free_cells)
    target_walls = int(len(free_cells) * 0.35)

    def still_connected(g, sr, sc, gr, gc):
        """Quick BFS connectivity check."""
        if g[sr,sc]==1 or g[gr,gc]==1: return False
        vis = set(); q = deque([(sr,sc)])
        vis.add((sr,sc))
        while q:
            r,c=q.popleft()
            if (r,c)==(gr,gc): return True
            for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and g[nr,nc]==0 and (nr,nc) not in vis:
                    vis.add((nr,nc)); q.append((nr,nc))
        return False

    # We'll wall cells, but first pick start/goal so we can connectivity-check
    # Quadrant free cells
    def quadrant_free(g, rmin, rmax, cmin, cmax):
        return [(r,c) for r in range(rmin,rmax) for c in range(cmin,cmax) if g[r,c]==0]

    def pick_start_goal(g):
        tl = quadrant_free(g, 0, max(1,rows//2), 0, max(1,cols//2))
        br = quadrant_free(g, rows//2, rows, cols//2, cols)
        if not tl: tl = [(r,c) for r in range(rows) for c in range(cols) if g[r,c]==0]
        if not br: br = [(r,c) for r in range(rows) for c in range(cols) if g[r,c]==0]
        sr,sc = rng.choice(tl)
        candidates = [(r,c) for r,c in br if (r,c)!=(sr,sc)]
        if not candidates: candidates = br
        gr_,gc_ = rng.choice(candidates)
        return (sr,sc),(gr_,gc_)

    (sr,sc),(gr,gc) = pick_start_goal(grid)

    walled = 0
    for cell in free_cells:
        if walled >= target_walls: break
        r, c = cell
        if (r,c)==(sr,sc) or (r,c)==(gr,gc): continue
        grid[r,c] = 1
        if not still_connected(grid, sr, sc, gr, gc):
            grid[r,c] = 0   # revert — would disconnect
        else:
            walled += 1

    # ── 3. BFS shortest path ────────────────────────────────────────────
    def bfs_path(src, dst):
        parent = {src: None}
        q = deque([src])
        while q:
            r,c = q.popleft()
            if (r,c)==dst: break
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]!=1 and (nr,nc) not in parent:
                    parent[(nr,nc)]=(r,c); q.append((nr,nc))
        if dst not in parent: return None
        path,cur=set(),dst
        while cur: path.add(cur); cur=parent.get(cur)
        return path

    path_cells = bfs_path((sr,sc),(gr,gc))
    if path_cells is None:
        return generate_maze(grid_size)   # very rare; retry

    # ── 4. Place start, goal, traps ──────────────────────────────────────
    grid[sr, sc] = 2
    grid[gr, gc] = 3

    trap_candidates = [
        (r,c) for r in range(rows) for c in range(cols)
        if grid[r,c]==0 and (r,c) not in path_cells
    ]
    rng.shuffle(trap_candidates)
    for cell in trap_candidates[:n_traps]:
        grid[cell] = 4

    return grid


class LabyrinthEnv:
    # Procedurally generated — refreshed each time generate_maze() is called.
    # The server keeps the "current" maze per size so train() uses the same
    # maze the browser just received.
    _current: dict = {}   # grid_size → np.ndarray, set by TrainingServer

    # 0=free 1=wall 2=start 3=goal 4=trap
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]   # up down left right

    def __init__(self, grid_size="5x5", max_steps=3000, grid=None):
        if grid is not None:
            self.grid = grid.copy()
        else:
            # fallback: generate fresh (shouldn't normally happen)
            self.grid = generate_maze(grid_size)
        self.rows, self.cols = self.grid.shape
        self.max_steps = max_steps
        self.start = tuple(map(int, np.argwhere(self.grid == 2)[0]))
        self.goal  = tuple(map(int, np.argwhere(self.grid == 3)[0]))
        self.traps = [tuple(map(int, p)) for p in np.argwhere(self.grid == 4)]
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.done = False
        self.path = [self.pos]
        return self._state()

    def _state(self):
        s = np.zeros(self.rows * self.cols, dtype=np.float32)
        s[self.pos[0]*self.cols + self.pos[1]] = 1.0
        return s

    def state_id(self):
        return self.pos[0]*self.cols + self.pos[1]

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        r, c = self.pos
        nr, nc = r+dr, c+dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr,nc] != 1:
            self.pos = (nr, nc)
        self.steps += 1
        self.path.append(self.pos)

        if self.pos == self.goal:
            reward, done = +10.0, True
        elif self.pos in self.traps:
            reward, done = -5.0, True
        elif self.steps >= self.max_steps:
            reward, done = -1.0, True
        else:
            dist = abs(self.pos[0]-self.goal[0]) + abs(self.pos[1]-self.goal[1])
            reward = -0.05 - 0.01*(dist/(self.rows+self.cols))
            done = False

        self.done = done
        return self._state(), reward, done

    @property
    def state_size(self): return self.rows * self.cols
    @property
    def action_size(self): return 4


# ═══════════════════════════════════════════════════════
#  TABULAR Q-LEARNING
# ═══════════════════════════════════════════════════════

class TabularQ:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.q = defaultdict(lambda: np.zeros(action_size))
        self.lr = 0.15
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.992
        self.name = "Tabular Q-Learning"

    def act(self, sid, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q[sid]))

    def learn(self, s, a, r, sn, done):
        target = r if done else r + self.gamma * np.max(self.q[sn])
        self.q[s][a] += self.lr * (target - self.q[s][a])
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def policy_map(self, rows, cols):
        arrows = ['↑','↓','←','→']
        out = []
        for r in range(rows):
            row = []
            for c in range(cols):
                q = self.q[r*cols+c]
                row.append(arrows[int(np.argmax(q))] if np.any(q != 0) else '·')
            out.append(row)
        return out

    def q_values_map(self, rows, cols):
        """Max Q-value per cell for heat-map coloring"""
        out = []
        for r in range(rows):
            row = []
            for c in range(cols):
                q = self.q[r*cols+c]
                row.append(float(np.max(q)))
            out.append(row)
        return out


# ═══════════════════════════════════════════════════════
#  DQN  (pure numpy, FC: state→64→32→actions)
# ═══════════════════════════════════════════════════════

class FCNet:
    def __init__(self, in_size, out_size):
        h1, h2 = 64, 32
        self.W1 = np.random.randn(in_size, h1) * np.sqrt(2/in_size)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2)     * np.sqrt(2/h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, out_size)* np.sqrt(2/h2)
        self.b3 = np.zeros(out_size)

    def forward(self, x):
        self._a0 = x
        self._z1 = x @ self.W1 + self.b1;  self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2; self._a2 = np.maximum(0, self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3

    def copy_from(self, other):
        self.W1,self.b1 = other.W1.copy(),other.b1.copy()
        self.W2,self.b2 = other.W2.copy(),other.b2.copy()
        self.W3,self.b3 = other.W3.copy(),other.b3.copy()

    def sgd_step(self, x, y_target, lr=0.001):
        """Single batch backprop"""
        y_pred = self.forward(x)
        delta3 = (y_pred - y_target) * 2 / x.shape[0]
        np.clip(delta3, -1, 1, out=delta3)

        dW3 = self._a2.T @ delta3;  db3 = delta3.sum(0)
        d2  = (delta3 @ self.W3.T) * (self._z2 > 0)
        dW2 = self._a1.T @ d2;      db2 = d2.sum(0)
        d1  = (d2 @ self.W2.T) * (self._z1 > 0)
        dW1 = self._a0.T @ d1;      db1 = d1.sum(0)

        for dW,db in [(dW1,db1),(dW2,db2),(dW3,db3)]:
            np.clip(dW,-1,1,out=dW); np.clip(db,-1,1,out=db)

        self.W1 -= lr*dW1; self.b1 -= lr*db1
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        self.W3 -= lr*dW3; self.b3 -= lr*db3


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.gamma = 0.99
        self.lr = 0.002
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.992
        self.batch = 32
        self.target_update = 20
        self._step = 0
        self.memory = deque(maxlen=3000)
        self.online = FCNet(state_size, action_size)
        self.target = FCNet(state_size, action_size)
        self.target.copy_from(self.online)
        self.name = "DQN"

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        q = self.online.forward(state[np.newaxis,:])[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, sn, done):
        self.memory.append((s, a, r, sn, done))

    def replay(self):
        if len(self.memory) < self.batch:
            return
        batch = random.sample(self.memory, self.batch)
        S  = np.array([b[0] for b in batch])
        A  = [b[1] for b in batch]
        R  = [b[2] for b in batch]
        Sn = np.array([b[3] for b in batch])
        D  = [b[4] for b in batch]

        Q_pred = self.online.forward(S)
        Q_next = self.target.forward(Sn)

        target = Q_pred.copy()
        for i in range(self.batch):
            t = R[i] if D[i] else R[i] + self.gamma*np.max(Q_next[i])
            target[i, A[i]] = t

        self.online.sgd_step(S, target, lr=self.lr)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        self._step += 1
        if self._step % self.target_update == 0:
            self.target.copy_from(self.online)


# ═══════════════════════════════════════════════════════
#  TRAINING LOOP  (runs in a thread, pushes to queue)
# ═══════════════════════════════════════════════════════

def train(agent_type: str, episodes: int, out_q: queue.Queue, stop_evt: threading.Event,
          grid_size: str = "5x5", grid: np.ndarray = None):
    env = LabyrinthEnv(grid_size=grid_size, grid=grid)

    if agent_type == "tabular":
        agent = TabularQ(env.state_size, env.action_size)
    else:
        agent = DQNAgent(env.state_size, env.action_size)

    for ep in range(1, episodes+1):
        if stop_evt.is_set():
            break

        state = env.reset()
        sid   = env.state_id()
        total_r = 0.0
        ep_path = [list(env.pos)]

        while not env.done:
            if stop_evt.is_set():
                break

            if agent_type == "tabular":
                action = agent.act(sid)
            else:
                action = agent.act(state)

            next_state, reward, done = env.step(action)
            next_sid = env.state_id()

            if agent_type == "tabular":
                agent.learn(sid, action, reward, next_sid, done)
            else:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

            state   = next_state
            sid     = next_sid
            total_r += reward
            ep_path.append(list(env.pos))

        success = env.pos == env.goal

        msg = {
            "type":    "episode",
            "agent":   agent_type,
            "ep":      ep,
            "total":   episodes,
            "reward":  round(total_r, 3),
            "steps":   env.steps,
            "success": success,
            "epsilon": round(agent.eps, 4),
            "path":    ep_path,
            "pos":     list(env.pos),
            "goal":    list(env.goal),
            "traps":   [list(t) for t in env.traps],
            "grid_size": grid_size,
        }

        # attach policy map for tabular so browser can render it
        if agent_type == "tabular":
            msg["policy"]  = agent.policy_map(env.rows, env.cols)
            msg["q_values"]= agent.q_values_map(env.rows, env.cols)

        out_q.put(json.dumps(msg))

    # final done signal
    out_q.put(json.dumps({"type":"done","agent":agent_type}))


# ═══════════════════════════════════════════════════════
#  HAND-ROLLED WEBSOCKET SERVER  (RFC 6455, stdlib only)
# ═══════════════════════════════════════════════════════

class WSClient:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.alive = True

    def handshake(self, raw_request: bytes) -> bool:
        try:
            lines = raw_request.decode().split('\r\n')
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    k, v = line.split(':', 1)
                    headers[k.strip().lower()] = v.strip()
            key = headers.get('sec-websocket-key','')
            magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
            accept = base64.b64encode(
                hashlib.sha1((key+magic).encode()).digest()
            ).decode()
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            )
            self.conn.sendall(response.encode())
            return True
        except Exception as e:
            return False

    def send_text(self, text: str):
        try:
            payload = text.encode('utf-8')
            n = len(payload)
            if n <= 125:
                header = struct.pack('BB', 0x81, n)
            elif n <= 65535:
                header = struct.pack('!BBH', 0x81, 126, n)
            else:
                header = struct.pack('!BBQ', 0x81, 127, n)
            self.conn.sendall(header + payload)
        except:
            self.alive = False

    def recv_text(self) -> str | None:
        try:
            raw = self.conn.recv(4096)
            if not raw:
                self.alive = False
                return None
            b1, b2 = raw[0], raw[1]
            masked  = bool(b2 & 0x80)
            length  = b2 & 0x7F
            idx = 2
            if length == 126:
                length = struct.unpack('!H', raw[idx:idx+2])[0]; idx+=2
            elif length == 127:
                length = struct.unpack('!Q', raw[idx:idx+8])[0]; idx+=8
            if masked:
                masks = raw[idx:idx+4]; idx+=4
                data  = bytes(b ^ masks[i%4] for i,b in enumerate(raw[idx:idx+length]))
            else:
                data = raw[idx:idx+length]
            opcode = b1 & 0x0F
            if opcode == 8:    # close
                self.alive = False
                return None
            return data.decode('utf-8') if data else None
        except:
            self.alive = False
            return None


class TrainingServer:
    def __init__(self, host='127.0.0.1', port=8765):
        self.host = host
        self.port = port
        self.clients: list[WSClient] = []
        self.clients_lock = threading.Lock()
        self.train_queue: queue.Queue = queue.Queue()
        self.stop_evt = threading.Event()
        self.train_thread: threading.Thread | None = None
        # current generated maze per grid_size — regenerated on each set_maze
        self.current_grids: dict = {}

    def _get_or_generate(self, grid_size: str) -> np.ndarray:
        if grid_size not in self.current_grids:
            self.current_grids[grid_size] = generate_maze(grid_size)
        return self.current_grids[grid_size]

    def _regenerate(self, grid_size: str) -> np.ndarray:
        self.current_grids[grid_size] = generate_maze(grid_size)
        return self.current_grids[grid_size]

    # ── broadcast to all live WS clients ──────────────────────────────
    def broadcast(self, msg: str):
        with self.clients_lock:
            dead = []
            for c in self.clients:
                c.send_text(msg)
                if not c.alive:
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)

    # ── drain train_queue and broadcast ───────────────────────────────
    def _drain_loop(self):
        while True:
            try:
                msg = self.train_queue.get(timeout=0.05)
                self.broadcast(msg)
            except queue.Empty:
                pass

    def _handle_ws(self, client: WSClient):
        # send a freshly generated maze on first connect
        self._send_maze(client, "5x5", regenerate=False)

        while client.alive:
            msg = client.recv_text()
            if msg is None:
                break
            try:
                data = json.loads(msg)
                cmd  = data.get("cmd")
                if cmd == "train":
                    agent_type = data.get("agent", "tabular")
                    episodes   = int(data.get("episodes", 400))
                    grid_size  = data.get("grid_size", "5x5")
                    # use the maze currently stored (already sent to browser)
                    grid = self._get_or_generate(grid_size)
                    # stop any running training
                    self.stop_evt.set()
                    if self.train_thread and self.train_thread.is_alive():
                        self.train_thread.join(timeout=2)
                    self.stop_evt.clear()
                    self.train_thread = threading.Thread(
                        target=train,
                        args=(agent_type, episodes, self.train_queue, self.stop_evt, grid_size, grid),
                        daemon=True
                    )
                    self.train_thread.start()

                elif cmd == "set_maze":
                    grid_size = data.get("grid_size", "5x5")
                    # always regenerate → new random maze every switch
                    self._send_maze(client, grid_size, regenerate=True)

                elif cmd == "stop":
                    self.stop_evt.set()

            except json.JSONDecodeError:
                pass

    def _send_maze(self, client: WSClient, grid_size: str, regenerate: bool = True):
        if regenerate:
            maze = self._regenerate(grid_size)
        else:
            maze = self._get_or_generate(grid_size)
        start = list(map(int, np.argwhere(maze == 2)[0]))
        goal  = list(map(int, np.argwhere(maze == 3)[0]))
        traps = [list(map(int, p)) for p in np.argwhere(maze == 4)]
        client.send_text(json.dumps({
            "type":      "maze",
            "grid":      maze.tolist(),
            "start":     start,
            "goal":      goal,
            "traps":     traps,
            "grid_size": grid_size,
        }))

    # ── raw TCP server that upgrades HTTP→WS ──────────────────────────
    def _tcp_server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port+1))   # WS on port+1 (8766)
        srv.listen(16)
        print(f"  WebSocket listening on ws://{self.host}:{self.port+1}")
        while True:
            conn, addr = srv.accept()
            # read HTTP upgrade request
            raw = conn.recv(4096)
            client = WSClient(conn, addr)
            if client.handshake(raw):
                with self.clients_lock:
                    self.clients.append(client)
                t = threading.Thread(target=self._handle_ws, args=(client,), daemon=True)
                t.start()
            else:
                conn.close()

    def start(self):
        # drain thread
        threading.Thread(target=self._drain_loop, daemon=True).start()
        # WS thread
        threading.Thread(target=self._tcp_server, daemon=True).start()


# ═══════════════════════════════════════════════════════
#  HTTP SERVER  (serves dashboard.html)
# ═══════════════════════════════════════════════════════

class DashHandler(BaseHTTPRequestHandler):
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")

    def do_GET(self):
        try:
            with open(self.html_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass  # silence access log


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    HTTP_PORT = 8765
    WS_PORT   = 8766

    ts = TrainingServer(port=HTTP_PORT)
    ts.start()

    httpd = HTTPServer(('127.0.0.1', HTTP_PORT), DashHandler)
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  Labyrinth RL Training Server        ║")
    print(f"  ║  Dashboard → http://localhost:{HTTP_PORT}  ║")
    print(f"  ║  WebSocket → ws://localhost:{WS_PORT}   ║")
    print(f"  ╚══════════════════════════════════════╝\n")
    print("  Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")