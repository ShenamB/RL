import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

# ---------- PAGE ----------
st.set_page_config(page_title="RL Grid World", layout="wide")
st.title("🧠 RL Grid World (Q-Learning)")
st.markdown(
    """
This app demonstrates a **Q-learning agent** navigating a grid world.

- The agent selects actions based on learned Q-values  
- Goal: reach the target cell with maximum reward  
- You can change grid size and observe behavior  

"""
)
# ---------- SIDEBAR ----------
with st.sidebar:
    size = st.selectbox("Grid Size", [3, 4, 5])
    speed = st.slider("Speed", 0.1, 1.5, 0.5)

    start = st.button("▶️ Start")
    reset = st.button("🔄 Reset")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model(size):
    path = f"models/q_table_{size}x{size}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

Q = load_model(size)

SIZE = size
GOAL = SIZE * SIZE - 1

actions_map = {
    0: "⬆️ Up",
    1: "⬇️ Down",
    2: "⬅️ Left",
    3: "➡️ Right"
}

# ---------- SESSION STATE ----------
if "state" not in st.session_state:
    st.session_state.state = 0

if "step" not in st.session_state:
    st.session_state.step = 0

if "running" not in st.session_state:
    st.session_state.running = False

if "path" not in st.session_state:
    st.session_state.path = []

if "prev_size" not in st.session_state:
    st.session_state.prev_size = size

# ---------- HANDLE GRID SIZE CHANGE ----------
if st.session_state.prev_size != size:
    st.session_state.state = 0
    st.session_state.step = 0
    st.session_state.running = False
    st.session_state.path = []
    st.session_state.prev_size = size

# ---------- ENV ----------
def get_next_state(state, action):
    row, col = divmod(state, SIZE)

    if action == 0:
        row = max(row - 1, 0)
    elif action == 1:
        row = min(row + 1, SIZE - 1)
    elif action == 2:
        col = max(col - 1, 0)
    elif action == 3:
        col = min(col + 1, SIZE - 1)

    return row * SIZE + col

# ---------- GRID ----------
def render_grid(state):
    cells = []

    for i in range(SIZE * SIZE):
        if i == state:
            cells.append('<div class="cell agent">A</div>')
        elif i == GOAL:
            cells.append('<div class="cell goal">G</div>')
        elif i in st.session_state.path:
            cells.append('<div class="cell visited">•</div>')
        else:
            cells.append('<div class="cell"></div>')

    html = f"""
    <div class="grid">{''.join(cells)}</div>

    <style>
    
    .grid {{
        display: grid;
    grid-template-columns: repeat({SIZE}, 80px);
    gap: 8px;
    justify-content: center;
    }}
    .cell {{
        width:80px;
        height:80px;
        border:1px solid #ddd;
        border-radius:10px;
    }}
    .agent {{
        background:green;
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:bold;
    }}
    .goal {{
        background:red;
        color:white;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:bold;
    }}
    .visited {{
        background:#e3f2fd;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:18px;
    }}
    </style>
    """

    height = int(SIZE * 110)
    st.components.v1.html(html, height=height, scrolling=False)

# ---------- RESET ----------
if reset:
    st.session_state.state = 0
    st.session_state.step = 0
    st.session_state.running = False
    st.session_state.path = []
    st.rerun()

# ---------- START ----------
if start:
    st.session_state.running = True

# ---------- LOOP ----------
placeholder = st.empty()

if st.session_state.running:

    while st.session_state.state != GOAL and st.session_state.step < 20:

        state = st.session_state.state

        # SAFETY (extra protection)
        if state >= SIZE * SIZE:
            st.session_state.state = 0
            break

        action = np.argmax(Q[state])

        # Track path
        st.session_state.path.append(state)

        with placeholder.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### 🧭 Step {st.session_state.step}")
                render_grid(state)

            with col2:
                st.markdown("### 🎯 Action")
                st.markdown(f"**{actions_map[action]}**")

                st.markdown("### 📊 Q-values")

                q_df = pd.DataFrame(
                Q[state].reshape(1, -1),
                columns=["Up", "Down", "Left", "Right"]
                )

                st.dataframe(q_df, use_container_width=True)

                st.markdown("### 🧠 Best Action")
                st.markdown(f"**{actions_map[np.argmax(Q[state])]}**")

        next_state = get_next_state(state, action)

        st.session_state.state = next_state
        st.session_state.step += 1

        time.sleep(speed)

    if st.session_state.state == GOAL:
        st.success("🎯 Goal reached!")
    else:
        st.warning("Stopped.")