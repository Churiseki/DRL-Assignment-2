import sys
import numpy as np
import random
import os
import pickle
from tqdm import tqdm
import math
import copy
from collections import defaultdict
import numpy as np
import itertools

class Connect6Game:
    def __init__(self, size=19, agent_black=None, agent_white=None):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.agent_black = agent_black
        self.agent_white = agent_white

    def reset_board(self):
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        col_char = col_char.upper()
        if col_char >= 'J':
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        #print(f"[play_move] Color: {color}, Move: {move}", file=sys.stderr)

        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)
    def generate_move(self, color):
        if self.game_over:
            print("? Game over")
            return

        agent = self.agent_black if color.upper() == 'B' else self.agent_white

        if agent is not None:
            move = agent.select_move(self.board)
            if move is None:
                print("? No valid move")
                return
            r, c = move
            move_str = f"{self.index_to_label(c)}{r+1}"
            self.play_move(color, move_str)
            print(f"= {move_str}\n", flush=True)
        else:
            # 原本的隨機選擇行為
            empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
            selected = random.sample(empty_positions, 1)
            move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)

            self.play_move(color, move_str)
            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)
            
        #print(f"[generate_move] Agent {'Black' if color.upper() == 'B' else 'White'} selecting move...", file=sys.stderr)
        #print(f"[generate_move] Move: {move_str}", file=sys.stderr)
        return move_str

    def show_board(self):
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def simulate_between_agents(self, agent1, agent2, max_moves=300):
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        agents = {1: agent1, 2: agent2}
        current_color = 1
        move_count = 0

        while not self.game_over and move_count < max_moves:
            agent = agents[current_color]
            move = agent.select_move(self.board)
            if move is None:
                break
            row, col = move
            self.board[row, col] = current_color
            agent.record_state(self.board, move)

            winner = self.check_win()
            if winner != 0:
                #print(f"\nWinner is {'Black' if winner == 1 else 'White'}!\n")
                #print(f"[simulate] Winner is {'Black' if winner == 1 else 'White'}", file=sys.stderr)

                self.game_over = True
                agent1.update_q_table(winner)
                agent2.update_q_table(winner)
                break

            if np.all(self.board != 0):
                #print("\nDraw!\n")
                #print("[simulate] Game ended in draw or max move limit.", file=sys.stderr)

                agent1.update_q_table(0)
                agent2.update_q_table(0)
                break

            current_color = 3 - current_color
            move_count += 1

        if not self.game_over:
            agent1.update_q_table(0)
            agent2.update_q_table(0)

        # agent1.save_q_table()
        agent2.save_q_table()

    def list_commands(self):
        print("= ", flush=True)

    def process_command(self, command):
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        while True:
            print(f"recieving", file=sys.stderr)
            try:
                line = sys.stdin.readline()
                print(f"got Line: {line}", file=sys.stderr)
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

def check_win(board):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    size = board.shape[0]
    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                current_color = board[r, c]
                for dr, dc in directions:
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r, prev_c] == current_color:
                        continue
                    count = 0
                    rr, cc = r, c
                    while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == current_color:
                        count += 1
                        rr += dr
                        cc += dc
                    if count >= 6:
                        return current_color
    return 0

class NTupleQLearningAgent:
    def __init__(self, color, board_size=19, alpha=0.1, gamma=0.99, epsilon=0.1, q_table_file="q_table.pkl"):
        self.color = color
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file

        if os.path.exists(q_table_file):
            with open(q_table_file, "rb") as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = {}

        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.trajectory = []

    def convert_board(self, board):
        result = np.zeros_like(board)
        result[board == self.color] = 1
        result[(board != 0) & (board != self.color)] = 2
        return result

    def extract_lines(self, board):
        lines = []
        size = board.shape[0]
        for dr, dc in self.directions:
            for r in range(size):
                for c in range(size):
                    line = []
                    for k in range(8):
                        nr, nc = r + k * dr, c + k * dc
                        if 0 <= nr < size and 0 <= nc < size:
                            line.append(board[nr, nc])
                        else:
                            break
                    if len(line) == 8:
                        lines.append(tuple(line))
        return lines
    
    def get_q_value(self, board):
        board_conv = self.convert_board(board)
        ifwin = check_win(board_conv)
        if ifwin == 1:
            return np.inf
        elif ifwin == 2:
            return -np.inf
        lines = self.extract_lines(board_conv)
        q_total = 0.0
        for line in lines:
            q_total += self.q_table.get(line, 0.0)
        return q_total

    def select_move(self, board):
        legal_moves = list(zip(*np.where(board == 0)))
        best_move = None
        best_q = -float('inf')
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        best_moves = []
        for move in legal_moves:
            temp_board = board.copy()
            temp_board[move] = self.color
            q_val = self.get_q_value(temp_board)
            # print(f"[select_move] Evaluating move {move}, Q = {q_val}", file=sys.stderr)
            if q_val > best_q:
                best_moves = [move]
                best_q = q_val
            elif q_val == best_q:
                best_moves.append(move)
        if best_moves:
            best_move = random.choice(best_moves)
            # print(f"[select_move] Best moves: {best_moves}, Q = {best_q}", file=sys.stderr)
        #print(f"[select_move] Selected best move: {best_move}, Q = {best_q}", file=sys.stderr)
        if best_move is not None:
            return best_move
        else:
            #print(f"[select_move] No best move found, selecting random move", file=sys.stderr)
            return random.choice(legal_moves)
        return best_move if best_move is not None else random.choice(legal_moves)

    def record_state(self, board, move):
        row, col = move
        board_conv = self.convert_board(board)
        size = board.shape[0]
        related_tuples = []

        for dr, dc in self.directions:
            for offset in range(-7, 1):  # 8個點的起點
                tuple_cells = []
                for k in range(8):
                    nr = row + (offset + k) * dr
                    nc = col + (offset + k) * dc
                    if 0 <= nr < size and 0 <= nc < size:
                        tuple_cells.append((nr, nc))
                    else:
                        break
                if len(tuple_cells) == 8:
                    tuple_values = tuple(board_conv[nr, nc] for nr, nc in tuple_cells)
                    related_tuples.append(tuple_values)

        self.trajectory.append(related_tuples)

    def update_q_table(self, result):
        if result == self.color:
            reward = 100.0
        elif result == 0:
            reward = 0.1
        else:
            reward = -50.0
        self.trajectory.reverse()
        for tuple_list in self.trajectory:
            for line in tuple_list:
                old_q = self.q_table.get(line, 0.0)
                new_q = old_q + self.alpha * (reward - old_q)
                self.q_table[line] = new_q
            reward *= self.gamma

        self.trajectory.clear()


    def save_q_table(self):
        with open(self.q_table_file, "wb") as f:
            pickle.dump(self.q_table, f)



def is_near_stone(board, pt, distance=1):
    r, c = pt
    rows, cols = board.shape
    r_min = max(0, r - distance)
    r_max = min(rows, r + distance + 1)
    c_min = max(0, c - distance)
    c_max = min(cols, c + distance + 1)
    sub_board = board[r_min:r_max, c_min:c_max]
    return np.any(sub_board != 0)


class TDMCTSAgent:
    def __init__(self, color, value_table, board_size=19, num_simulations=100):
        self.color = color
        self.value_table = value_table
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.pending_move = None
        self.first_move_done = False  # 黑棋第一步用

    def convert_board(self, board, player):
        result = np.zeros_like(board)
        result[board == player] = 1
        result[(board != 0) & (board != player)] = 2
        return result

    def extract_related_tuples(self, board, move, player):
        related = []
        size = board.shape[0]
        board_conv = self.convert_board(board, player)
        for dr, dc in self.directions:
            for offset in range(-7, 1):
                line = []
                valid = True
                for i in range(8):
                    r = move[0] + (offset + i) * dr
                    c = move[1] + (offset + i) * dc
                    if 0 <= r < size and 0 <= c < size:
                        line.append(board_conv[r, c])
                    else:
                        valid = False
                        break
                if valid and (move[0], move[1]) in [(move[0] + k * dr, move[1] + k * dc) for k in range(8)]:
                    related.append(tuple(line))
        return related

    def evaluate_board(self, board, player):
        value = -np.inf
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != player:
                    continue
                tuples = self.extract_related_tuples(board, (r, c), player)
                for t in tuples:
                    idx = sum(cell * (3 ** i) for i, cell in enumerate(t))
                    value = max(value, self.value_table[idx])
        return value

    def generate_moves(self, board, num_moves):
        empties = list(zip(*np.where(board == 0)))
        valid = [pt for pt in empties if is_near_stone(board, pt)]

        if num_moves == 1:
            return [(pt,) for pt in valid]

        valid_pairs = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                (r1, c1), (r2, c2) = valid[i], valid[j]
                if abs(r1 - r2) <= 7 and abs(c1 - c2) <= 7:
                    valid_pairs.append((valid[i], valid[j]))
        return valid_pairs

    def simulate(self, board, move_pair):
        # print(f"[simulate] Simulating move pair: {move_pair}", file=sys.stderr)
        temp_board = board.copy()
        for r, c in move_pair:
            temp_board[r, c] = self.color
        my_value = self.evaluate_board(temp_board, self.color)
        opp_color = 3 - self.color
        opp_best = float('-inf')
        opp_moves = self.generate_moves(temp_board, 2)
        for opp_move in random.sample(opp_moves, min(2, len(opp_moves))):  # 採樣對手
            for r, c in opp_move:
                temp_board[r, c] = opp_color
            val = self.evaluate_board(temp_board, opp_color)
            opp_best = max(opp_best, val)
            for r, c in opp_move:
                temp_board[r, c] = 0
        return my_value - opp_best  # 越高越好

    def select_move(self, board):
        # print(f"[select_move] Selecting move for color {self.color}", file=sys.stderr)
        if self.pending_move:
            move = self.pending_move
            self.pending_move = None
            return move

        num_moves = 1 if (self.color == 1 and not self.first_move_done) else 2
        self.first_move_done = True if self.color == 1 else self.first_move_done

        candidates = self.generate_moves(board, num_moves)
        best_score = float('-inf')
        best_pair = None

        for move_pair in tqdm(random.sample(candidates, min(len(candidates), 40))):
            # print(f"[select_move] Evaluating move pair: {move_pair}", file=sys.stderr)
            total = 0
            for _ in range(self.num_simulations):
                total += self.simulate(board, move_pair)
            avg_score = total / self.num_simulations
            if avg_score > best_score:
                best_score = avg_score
                best_pair = move_pair

        if best_pair:
            self.pending_move = best_pair[1] if num_moves == 2 else None
            return best_pair[0]
        else:
            empties = list(zip(*np.where(board == 0)))
            return random.choice(empties)
import numpy as np

def build_fast_value_table(slow_table):
    fast_table = np.zeros(3 ** 8, dtype=np.float32)
    for tup, val in slow_table.items():
        idx = sum(cell * (3 ** i) for i, cell in enumerate(tup))
        fast_table[idx] = val
    return fast_table

if __name__ == "__main__":
    value_table = build_fast_value_table(pickle.load(open("q_table.pkl", "rb")))
    agent1 = TDMCTSAgent(color=1, value_table=value_table, board_size=10)
    agent2 = TDMCTSAgent(color=2, value_table=value_table, board_size=10)
    game = Connect6Game(size=19, agent_black=agent1, agent_white=agent2)
    game.run()

