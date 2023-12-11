from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import numpy as np
import random
import pickle
import sys
import io


# Rules on PDF

class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

class Player(ABC):
    def __init__(self, name: str) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        self.name = name
        pass

    def __str__(self) -> str:
        return f'{self.name}'

    def is_agent(self) -> bool:
        return isinstance(self, MyAgent)

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass

class Game(object):
    def __init__(self) -> None:
        self.winner: Player = None
        self._current_player_idx = 1
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self._available_moves_list: list[(tuple[int, int], Move)] = list()
        self._emojis = ['❌', '⭕️', '⚪️']

    def __str__(self) -> str:
        original_stdout = sys.stdout
        output_buffer = io.StringIO()
        sys.stdout = output_buffer

        print('╔════╤════╤════╤════╤════╗ ') 

        for r, row in enumerate(self._board):
            print('║', end=' ')
            for t, tile in enumerate(row):
                print(self._emojis[tile] , end=' ')

                if t < 4:
                    print('│', end=' ')
                else:
                    print('║', end=' ')

            if r < 4:        
                print("\n╟────┼────┼────┼────┼────╢")

        print('\n╚════╧════╧════╧════╧════╝') 

        sys.stdout = original_stdout
        captured_output = output_buffer.getvalue()
        return captured_output

    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        for x in range(self._board.shape[0]):
            if all(self._board[x, :] == self._board[x, 0]):
                return self._board[x, 0]
        for y in range(self._board.shape[0]):
            if all(self._board[:, y] == self._board[0, y]):
                return self._board[0, y]
        if all([self._board[x, x] for x in range(self._board.shape[0])] == self._board[0, 0]):
            return self._board[0, 0]
        if all([self._board[x, -x] for x in range(self._board.shape[0])] == self._board[-1, -1]):
            return self._board[0, -1]
        return -1

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            self._current_player_idx += 1
            self._current_player_idx %= len(players)
            ok = False
            # calculate all possible available moves from this state
            self.__available_moves()
            while not ok:
                from_pos, slide = players[self._current_player_idx].make_move(self)
                ok = self.__move(from_pos, slide)
                if not ok and isinstance(players[self._current_player_idx], HumanPlayer):
                    print("That's an invalid move, please reenter your move:")
            # print(self)
            winner = self.check_winner()
        
        self.winner = players[winner]
        return winner

    def single_move(self, from_pos: tuple[int, int], slide: Move) -> None:
        ok = self.__move(from_pos, slide)
        assert ok == True

    def __move(self, from_pos: tuple[int, int], slide: Move, mock: bool=False) -> bool:
        '''Perform a move'''
        assert self._current_player_idx < 2

        prev_value = deepcopy(self._board)
        acceptable = self.__take(from_pos)
        if acceptable:
            acceptable = self.__slide(from_pos, slide)
            if not acceptable:
                self._board = deepcopy(prev_value)

        if mock:
            self._board = deepcopy(prev_value)

        return acceptable

    def __take(self, from_pos: tuple[int, int]) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (from_pos[0] == 0 and from_pos[1] < 5) or (from_pos[0] == 4 and from_pos[1] < 5) or (
            from_pos[1] == 0 and from_pos[0] < 5) or (from_pos[1] == 4 and from_pos[0] < 5) and (self._board[from_pos] < 0 or self._board[from_pos] == self._current_player_idx)
        if acceptable:
            self._board[(from_pos[1], from_pos[0])] = self._current_player_idx
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        if from_pos not in SIDES:
            acceptable_top: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT)
            acceptable_bottom: bool = from_pos[1] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT)
            acceptable_left: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT)
            acceptable_right: bool = from_pos[0] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT)
        else:
            # top left
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # top right
            acceptable_right: bool = from_pos == (4, 0) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # bottom left
            acceptable_left: bool = from_pos == (0, 4) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # bottom right
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)

        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        if acceptable:
            piece = self._board[from_pos]

            if slide == Move.TOP:
                column = [row[from_pos[0]] for row in self._board]
                rotated_column = column[-1:] + column[:-1]
                for i, row in enumerate(self._board):
                    row[from_pos[0]] = rotated_column[i]
            elif slide == Move.BOTTOM:
                column = [row[from_pos[0]] for row in self._board]
                rotated_column = column[1:] + column[:1]
                for i, row in enumerate(self._board):
                    row[from_pos[0]] = rotated_column[i]
            elif slide == Move.LEFT:
                self._board[from_pos[1]] = np.concatenate((self._board[from_pos[1]][-1:], self._board[from_pos[1]][:-1]), axis=None)
            elif slide == Move.RIGHT:
                self._board[from_pos[1]] = np.concatenate((self._board[from_pos[1]][1:], self._board[from_pos[1]][:1]), axis=None)

        return acceptable

    def __available_moves(self) -> None:
        self._available_moves_list.clear()
        for x in range(5):
            for slide in Move:
                ok = self.__move((0, x), slide, mock=True)
                if ok:
                    self._available_moves_list.append(((0, x), slide))
                ok = self.__move((4, x), slide, mock=True)
                if ok:
                    self._available_moves_list.append(((4, x), slide))
                
                if x != 0 and x != 4:
                    ok = self.__move((x, 0), slide, mock=True)
                    if ok:
                        self._available_moves_list.append(((x, 0), slide))     
                    ok = self.__move((x, 4), slide, mock=True)
                    if ok:
                        self._available_moves_list.append(((x, 4), slide))

    def get_available_moves(self) -> list[(tuple[int, int], Move)]:
        '''return the possible moves in the current position'''
        return self._available_moves_list
        
    def get_hash(self) -> str:
        '''hashes the state of the board'''
        return str(self._board.reshape(5 * 5))

class RandomPlayer(Player):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MyAgent(Player):
    def __init__(self, name: str, exp_rate=0.3) -> None:
        super().__init__(name)
        self._states = list()
        self._state_values = dict()
        self._lr = 0.2
        self._exp_rate = exp_rate
        self._decay_gamma = 0.9

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        from_pos, move = self.__choose_action(game)
        return from_pos, move

    def feed_reward(self, reward: float) -> None:
        for st in reversed(self._states):
            if self._states_value.get(st) is None:
                self._states_value[st] = 0
            self._states_value[st] += self._lr * (self._decay_gamma * reward - self._states_value[st])
            reward = self._states_value[st]

    def __choose_action(self, game: Game) -> tuple[tuple[int, int], Move]:
        possible_moves = game.get_available_moves()
        if np.random.uniform(0, 1) <= self._exp_rate:
            # take random action
            idx = np.random.choice(len(possible_moves))
            action = possible_moves[idx]
        else:
            value_max = -999
            for pm in possible_moves:
                from_pos, move = pm
                next_state = deepcopy(game)
                next_state.single_move(from_pos, move)
                next_hash = game.get_hash(next_state)
                value = 0 if self._states_value.get(next_hash) is None else self._states_value.get(next_hash)
                if value >= value_max:
                    value_max = value
                    action = pm
        
        # add state to the hash table
        self._states.append(next_hash)

        return action

    def save_policy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self._states_value, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, 'rb')
        self._states_value = pickle.load(fr)
        fr.close()

class HumanPlayer(Player):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        x = int(input("Input the x coordinate (from 0 to 4):"))
        y = int(input("Input the y coordinate (from 0 to 4):"))
        move = int(input("Input the slide method (0:top, 1:bottom, 2:left, 3:right):"))
        return (x, y), Move(move)