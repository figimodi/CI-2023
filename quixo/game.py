from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import NamedTuple
import numpy as np
import random
import pickle
import sys
import io


class Coordinates(NamedTuple):
    '''Coordinates on the board'''
    x: int
    y: int

class Slide(Enum):
    '''Type of slides, for example: TOP means that we push from the TOP to the BOTTOM'''
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

class Move(NamedTuple):
    '''Combination of coordinates and type of slide'''
    coordinates: Coordinates
    slide: Slide

class Player(ABC):
    '''Skeleton for different types of players'''

    def __init__(self, name: str) -> None:
        self.name = name
        pass

    def __str__(self) -> str:
        return f'{self.name}'

    def is_agent(self) -> bool:
        '''Tells if the object Player is an instance of the class MyAgent'''
        return isinstance(self, MyAgent)

    def is_human(self) -> bool:
        '''Tells if the object Player is an instance of the class HumanPlayer'''
        return isinstance(self, HumanPlayer)

    @abstractmethod
    def make_move(self, game: Game) -> Move:
        '''
        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass

class Game(object):
    '''
    This class represents the state of the game, along with the board,
    the current turn, the available moves in the position and the winner when
    the match is over.
    '''

    def __init__(self) -> None:
        self.winner: Player = None
        self._current_player_idx = 1
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self._available_moves_list: list[Move] = list()
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
        # Check the rows
        for x in range(self._board.shape[0]):
            if all(self._board[x, :] == self._board[x, 0]):
                return self._board[x, 0]
        
        # Check the columns
        for y in range(self._board.shape[0]):
            if all(self._board[:, y] == self._board[0, y]):
                return self._board[0, y]
        
        # Check the diagonals
        if all([self._board[x, x] for x in range(self._board.shape[0])] == self._board[0, 0]):
            return self._board[0, 0]
        if all([self._board[x, -x] for x in range(self._board.shape[0])] == self._board[-1, -1]):
            return self._board[0, -1]

        # TODO: add the case in which there are two different five-in-a-row combinations -> wins the opponent
        
        return -1

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            self._current_player_idx += 1
            self._current_player_idx %= len(players)
            valid = False
            
            # Calculate all possible available moves from the current state
            self.__available_moves()
            
            while not valid:
                coordinates, slide = players[self._current_player_idx].make_move(self)
                valid = self.__move(coordinates, slide)
                if not valid and isinstance(players[self._current_player_idx], HumanPlayer):
                    print("That's an invalid move, please reenter your move:")
            
            # Activate the following print if at least one of the player is human
            if player1.is_human() or player2.is_human:
                print(self)

            # Check if the game has a winner
            winner = self.check_winner()
        
        # Assing the winner to the class and return the corresponing ID
        self.winner = players[winner]
        return winner

    def single_move(self, coordinates: Coordinates, slide: Slide) -> None:
        '''Makes a single move on the board'''
        ok = self.__move(coordinates, slide)
        assert ok == True

    def get_available_moves(self) -> list[Move]:
        '''Return the possible moves in the current position'''
        return self._available_moves_list
        
    def get_hash(self) -> str:
        '''Hashes the state of the board'''
        return str(self._board.reshape(5 * 5))

    def reset(self) -> None:
        '''Reset the state of the board'''
        self.winner: Player = None
        self._current_player_idx = 1
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self._available_moves_list = list()
    
    def __move(self, coordinates: Coordinates, slide: Slide, mock: bool=False) -> bool:
        '''Perform a move'''
        # Save the sate as it is now
        prev_value = deepcopy(self._board)

        # Check the validity of the move
        acceptable = self.__take(coordinates)
        if acceptable:
            acceptable = self.__slide(coordinates, slide)
            if not acceptable:
                self._board = deepcopy(prev_value)

        # Allow to perform __move but without affecting the board
        if mock:
            self._board = deepcopy(prev_value)

        return acceptable

    def __take(self, coordinates: Coordinates) -> bool:
        '''Take piece  and 'flip it' facing the player symbol'''
        # Acceptable only if in border
        acceptable: bool = (coordinates[0] == 0 and coordinates[1] < 5) or (coordinates[0] == 4 and coordinates[1] < 5) or (
            coordinates[1] == 0 and coordinates[0] < 5) or (coordinates[1] == 4 and coordinates[0] < 5) and (self._board[coordinates] < 0 or self._board[coordinates] == self._current_player_idx)
        if acceptable:
            self._board[(coordinates[1], coordinates[0])] = self._current_player_idx
        return acceptable

    def __slide(self, coordinates: Coordinates, slide: Slide) -> bool:
        '''Slide the other pieces'''
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        if coordinates not in SIDES:
            acceptable_top: bool = coordinates[1] == 0 and (
                slide == Slide.BOTTOM or slide == Slide.LEFT or slide == Slide.RIGHT)
            acceptable_bottom: bool = coordinates[1] == 4 and (
                slide == Slide.TOP or slide == Slide.LEFT or slide == Slide.RIGHT)
            acceptable_left: bool = coordinates[0] == 0 and (
                slide == Slide.BOTTOM or slide == Slide.TOP or slide == Slide.RIGHT)
            acceptable_right: bool = coordinates[0] == 4 and (
                slide == Slide.BOTTOM or slide == Slide.TOP or slide == Slide.LEFT)
        else:
            # top left
            acceptable_top: bool = coordinates == (0, 0) and (
                slide == Slide.BOTTOM or slide == Slide.RIGHT)
            # top right
            acceptable_right: bool = coordinates == (4, 0) and (
                slide == Slide.BOTTOM or slide == Slide.LEFT)
            # bottom left
            acceptable_left: bool = coordinates == (0, 4) and (
                slide == Slide.TOP or slide == Slide.RIGHT)
            # bottom right
            acceptable_bottom: bool = coordinates == (4, 4) and (
                slide == Slide.TOP or slide == Slide.LEFT)

        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        
        # If the move is allowed, then slide the pieces on the corresponding column/row
        if acceptable:
            if slide == Slide.TOP:
                column = [row[coordinates[0]] for row in self._board]
                rotated_column = column[-1:] + column[:-1]
                for i, row in enumerate(self._board):
                    row[coordinates[0]] = rotated_column[i]
            elif slide == Slide.BOTTOM:
                column = [row[coordinates[0]] for row in self._board]
                rotated_column = column[1:] + column[:1]
                for i, row in enumerate(self._board):
                    row[coordinates[0]] = rotated_column[i]
            elif slide == Slide.LEFT:
                self._board[coordinates[1]] = np.concatenate((self._board[coordinates[1]][-1:], self._board[coordinates[1]][:-1]), axis=None)
            elif slide == Slide.RIGHT:
                self._board[coordinates[1]] = np.concatenate((self._board[coordinates[1]][1:], self._board[coordinates[1]][:1]), axis=None)

        return acceptable

    def __available_moves(self) -> None:
        '''Calculate all the possible moves from the current state'''
        # Clear the list containing the possible moves (for the old state)
        self._available_moves_list.clear()

        # Try all possible moves (with mock=True not to modify the board)
        for x in range(5):
            for slide in Slide:
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


class RandomPlayer(Player):
    '''
    This class contains the implementation for the Random Player.
    The make_move method is overwrite by simply choosing a random move.
    '''

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: 'Game') -> Move:
        coordinates = (random.randint(0, 4), random.randint(0, 4))
        slide = random.choice([Slide.TOP, Slide.BOTTOM, Slide.LEFT, Slide.RIGHT])
        return coordinates, slide

class MyAgent(Player):
    '''
    The class contains the implementation for the Agent using Reinforcement Learning.
    It has several methods and attributes:
    lr:           learning rate: the magnitude of the rewards given to the states
    exp_rate:     exploration rate, probability of chosing random moves and exploring new strategies
    decay_gamma:  discount, act as a discount factor
    states:       contains the path of the player in the game, storing all the states from start to finish
    state_values: dictionary that conatins (state, value) pair for each state that the agent knows
    '''
    def __init__(self, name: str, exp_rate=0.3) -> None:
        super().__init__(name)
        self._states = list()
        self._state_value = dict()
        self._lr = 0.2
        self._exp_rate = exp_rate
        self._decay_gamma = 0.9

    def make_move(self, game: Game) -> Move:
        coordinates, slide = self.__choose_action(game)
        return coordinates, slide

    def feed_reward(self, reward: float) -> None:
        '''Gives rewards to the actions perfomed in the match'''
        # Starting from the last state of the game, update the value associated to that state
        for st in reversed(self._states):
            if self._state_value.get(st) is None:
                self._state_value[st] = 0
            self._state_value[st] += self._lr * (self._decay_gamma * reward - self._state_value[st])
            reward = self._state_value[st]

    def __choose_action(self, game: Game) -> Move:
        '''Return a move, coordinates + slide'''
        # Retrieve ll possible moves from the current state (moves have already been calculated)
        possible_moves = game.get_available_moves()

        # With probability exp_rate the agent choose to play a random move to favor exploration
        if np.random.uniform(0, 1) <= self._exp_rate:
            idx = np.random.choice(len(possible_moves))
            action = possible_moves[idx]
        else:
            value_max = -999
            # For all possible moves we retrieve the value from the dictionary (if the state was already visited)
            for pm in possible_moves:
                coordinates, slide = pm
                next_state = deepcopy(game)
                next_state.single_move(coordinates, slide)
                next_hash = next_state.get_hash()
                value = 0 if self._state_value.get(next_hash) is None else self._state_value.get(next_hash)
                
                # If we get a state that has a better score, than we save the action that leads to that state
                if value >= value_max:
                    value_max = value
                    action = pm

        # Briefly update the path of the player through the game (_states.append)
        coordinates, slide = action
        next_state = deepcopy(game)
        next_state.single_move(coordinates, slide)
        next_hash = next_state.get_hash()
        self._states.append(next_hash)

        return action

    def reset_states(self) -> None:
        '''Reset the path of the player into the game (when starting a new game)'''
        self._states.clear()

    def save_policy(self) -> None:
        '''Save the dictionary of pairs (state, value) representing the knoledge of the agent'''
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self._state_value, fw)
        fw.close()

    def load_policy(self, file) -> None:
        '''Load the dictionary of pairs (state, value) representing the knoledge of the agent'''
        try:
            fr = open(file, 'rb')
            self._state_value = pickle.load(fr)
            fr.close()
        except FileNotFoundError:
            sys.exit(f"ERROR: failed to load the policy, file {file} doesn't exist")

    def set_exp_rate(self, exp_rate: float=0.3) -> None:
        '''
        Set exploration rate, usefull when we want to test and set exp_rate=0.
        i.e. only taking rational decisions for which we can estimate the outcome
        '''
        self._exp_rate = exp_rate

class HumanPlayer(Player):
    '''
    Class for representing a Human Player,
    the make_move method is overwritten with just inputs from the keyboard,
    such that an human can play
    '''
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: Game) -> Move:
        '''Asks for (x, y) and slide, that combined represents a move on the board'''
        x = int(input("Input the x coordinate (from 0 to 4):"))
        y = int(input("Input the y coordinate (from 0 to 4):"))
        slide = int(input("Input the slide type (0:top, 1:bottom, 2:left, 3:right):"))
        return (x, y), Slide(slide)