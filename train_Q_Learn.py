import numpy as np
import pickle
import os

# ---------- SETTINGS ----------
SIZE = 5   # 🔁 Change to 3, 4, 5 and run separately
EPISODES = 2000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

GOAL = SIZE * SIZE - 1

# ---------- Q TABLE ----------
Q = np.zeros((SIZE * SIZE, 4))

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

# ---------- TRAINING ----------
for episode in range(EPISODES):
    state = np.random.randint(0, SIZE * SIZE)

    for _ in range(100):
        if np.random.rand() < EPSILON:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(Q[state])

        next_state = get_next_state(state, action)

        reward = 1 if next_state == GOAL else -0.01

        Q[state][action] += ALPHA * (
            reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

        if state == GOAL:
            break

    if episode % 200 == 0:
        print(f"Episode {episode}")

# ---------- SAVE MODEL ----------
os.makedirs("models", exist_ok=True)

file_path = f"models/q_table_{SIZE}x{SIZE}.pkl"

with open(file_path, "wb") as f:
    pickle.dump(Q, f)

print(f"✅ Saved: {file_path}")

# Save model
import os

os.makedirs("models", exist_ok=True)

file_name = f"models/q_table_{SIZE}x{SIZE}.pkl"

with open(file_name, "wb") as f:
    pickle.dump(Q, f)

print(f"✅ Saved: {file_name}")