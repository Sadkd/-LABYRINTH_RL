# Labyrinth RL — Live Training Dashboard

A real-time reinforcement learning system that lets you watch agents learn to solve procedurally generated mazes live.

This project combines a pure Python RL backend with a modern browser dashboard, connected via a custom WebSocket server — all with zero heavy frameworks.

## Features
### 🎮 Interactive Maze Environment
Procedurally generated mazes (every run is different)
Grid sizes:
* 5×5
* 8×8
* 10×10

### 🤖 Reinforcement Learning Agents
* 🔹 Tabular Q-Learning
    * Q-table (state × action)
    * Fast convergence on small grids
    * Fully interpretable (policy map)
  
* 🔸 Deep Q-Network (DQN)
    * Pure NumPy neural network (state → 64 → 32 → actions)
    * Experience replay buffer
    * Target network updates
    * Scales better to larger environments

### 📡 Real-Time Training
* Live episode streaming via WebSocket
* Per-step visualization

### 📊 Live Analytics
* Smoothed reward curves
* Steps per episode
* Exploration rate (ε)
* Success rate & averages

### 🗺 Policy Visualization
* Arrow-based greedy policy
* Q-value heatmap
* Updates every episode

### 📋 Episode Logs
* Goal / trap / timeout detection
* Scrollable training history
* Color-coded outcomes

## ⚙️ Setup & Usage
1. Install Requirements
Only dependency:      pip install numpy
2. Run the Server:    python server.py

You’ll see:
*  Dashboard → http://localhost:8765
*  WebSocket → ws://localhost:8766
3. Open the Dashboard go to:
*  http://localhost:8765
4. Start Training, select:
*  Grid size
*  Episode count
*  Choose agent:
    ▶ Tabular Q
    ▶ DQN
, then watch learning happen in real time

## 🧠 Core Components

### 🎮 Environment (LabyrinthEnv)
* State: one-hot encoded position
* Actions: up, down, left, right
* Rewards:
    * +10 → goal
    * -5 → trap
    * -1 → timeout
small negative step penalty (encourages efficiency)

### 📚 Tabular Q-Learning
  * ε-greedy exploration
  * Learning rate: 0.15
  * Discount: 0.99
  * ε decay: 0.992

### 🧠 DQN 
Fully connected network:
  * input → 64 → 32 → output
  * Replay buffer: 3000
  * Batch size: 32
  * Target network sync every 20 steps
  * Gradient clipping for stability

### 🔌 WebSocket Server (Custom Implementation)
No external libraries — built from scratch using:
  * socket
  * struct
  * hashlib
  * base64

Handles:
* Handshake (RFC 6455)
* Frame encoding/decoding
* Multi-client broadcasting

### 🌐 HTTP Server
  * Serves dashboard.html
  * Runs on localhost:8765
  * Built with http.server

### ⚡ Performance Notes
* Tabular Q converges very fast on small grids
* DQN is slower but:
    * More generalizable
    * Handles larger state spaces better
* All training runs in a separate thread, keeping UI smooth
  
## 🔮 Future Improvements
  * Add more algorithms (SARSA, PPO, A2C)
  * Save/load trained models
  * Multi-agent side-by-side training

## 📜 License
MIT License — free to use, modify, and distribute.

