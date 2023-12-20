from abc import ABC, abstractmethod
from copy import deepcopy
from game import Game, Coordinates, Slide, Move
import numpy as np
import random
import pickle
import sys

class Player(ABC):
    '''Skeleton for different types of players.'''

    def __init__(self, name: str) -> None:
        self.name = name
        pass

    def __str__(self) -> str:
        return f'{self.name}'

    def is_RLagent(self) -> bool:
        '''Tells if the object Player is an instance of the class RLPlayer.'''
        return isinstance(self, RLPlayer)

    def is_human(self) -> bool:
        '''Tells if the object Player is an instance of the class HumanPlayer.'''
        return isinstance(self, HumanPlayer)

    @abstractmethod
    def make_move(self, game: 'Game') -> Move:
        '''
        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT.
        '''
        pass

class RandomPlayer(Player):
    '''
    This class contains the implementation for the Random Player.
    The make_move method is overwrite by simply choosing a random move.
    '''

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: 'Game') -> Move:
        '''Make the move in a random fashion.'''
        coordinates = (random.randint(0, 4), random.randint(0, 4))
        slide = random.choice([Slide.TOP, Slide.BOTTOM, Slide.LEFT, Slide.RIGHT])
        return coordinates, slide

class HumanPlayer(Player):
    '''
    Class for representing a Human Player,
    the make_move method is overwritten with just inputs from the keyboard,
    such that an human can play.
    '''
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: Game) -> Move:
        '''Asks for (row, column) and slide, that combined represents a move on the board.'''
        row = int(input("Input the row coordinate (from 0 to 4):"))
        col = int(input("Input the column coordinate (from 0 to 4):"))
        slide = int(input("Input the slide type (0:top, 1:bottom, 2:left, 3:right):"))
        return (row, col), Slide(slide)

class RLPlayer(Player):
    '''
    The class contains the implementation for the Agent using Reinforcement Learning.
    It has several methods and attributes:
    lr:           learning rate: the magnitude of the rewards given to the states.
    exp_rate:     exploration rate, probability of chosing random moves and exploring new strategies.
    decay_gamma:  discount, act as a discount factor.
    states:       contains the path of the player in the game, storing all the states from start to finish.
    state_values: dictionary that conatins (state, value) pair for each state that the agent knows.
    '''
    def __init__(self, name: str, exp_rate=0.3) -> None:
        super().__init__(name)
        self._states = list()
        self._state_value = dict()
        self._lr = 0.2
        self._exp_rate = exp_rate
        self._decay_gamma = 0.9

    def make_move(self, game: Game) -> Move:
        '''Returns the coordinates, slide tuple by choosing the best candidate move.'''
        coordinates, slide = self.__choose_action(game)
        return coordinates, slide

    def feed_reward(self, reward: float) -> None:
        '''Gives rewards to the actions perfomed in the match.'''
        # Starting from the last state of the game, update the value associated to that state
        for st in reversed(self._states):
            if self._state_value.get(st) is None:
                self._state_value[st] = 0
            self._state_value[st] += self._lr * (self._decay_gamma * reward - self._state_value[st])
            reward = self._state_value[st]

    def __choose_action(self, game: Game) -> Move:
        '''Return a move, coordinates + slide.'''
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
        '''Reset the path of the player into the game (when starting a new game).'''
        self._states.clear()

    def save_policy(self) -> None:
        '''Save the dictionary of pairs (state, value) representing the knoledge of the agent.'''
        fw = open('../policies/policy_' + str(self.name), 'wb')
        pickle.dump(self._state_value, fw)
        fw.close()

    def load_policy(self, file) -> None:
        '''Load the dictionary of pairs (state, value) representing the knoledge of the agent.'''
        try:
            fr = open(file, 'rb')
            self._state_value = pickle.load(fr)
            fr.close()
        except FileNotFoundError:
            sys.exit(f"ERROR: failed to load the policy, file {file} doesn't exist")

    def set_exp_rate(self, exp_rate: float=0.3) -> None:
        '''
        Set exploration rate, usefull when we want to test and set exp_rate=0.
        i.e. only taking rational decisions for which we can estimate the outcome.
        '''
        self._exp_rate = exp_rate

class MinMaxPlayer(Player):
    '''
    The class contains the implementation for the Agent using the Min-Max algorithm with alpha beta pruning.
    '''

    def __init__(self, name: str, max_depth: int = 3) -> None:
        super().__init__(name)
        self._MIN_VALUE = -10000
        self._MAX_VALUE = 10000
        self._max_depth = max_depth

    def make_move(self, game: Game) -> Move:
        '''Return the coordinates, slide tuple that gives the best evaluation of the posistion with a certain depth.'''
        coordinates, slide = self.__min_max_decision(game)
        return coordinates, slide

    def set_max_depth(self, max_depth: int) -> None:
        '''Change the maximum depth that the engine can reach.'''
        self._max_depth = max_depth

    def __min_max_decision(self, game: Game) -> Move:
        '''Return the best move according to the min max algorithm, with the help of alpha beta pruning.'''
        v = self._MIN_VALUE
        alpha = self._MIN_VALUE
        beta = self._MAX_VALUE

        possible_moves = game.get_available_moves()

        for pm in possible_moves:
            coordinates, slide = pm
            next_position = deepcopy(game)
            next_position.single_move(coordinates, slide)

            min_result = -self.__min_value(next_position, alpha, beta, 1)
            if min_result > v:
                v = min_result
                action = pm

            if v >= beta:
                break

            alpha = max(alpha, v)

        return action
    
    def __min_value(self, game: Game, alpha: int, beta: int, depth: int) -> int:
        '''Select the best move in the min layer of the min max algorithm.'''
        # Check if we are in a final position
        winner = game.check_winner()
        if winner is not None:
            if game.players[winner] is self:
                return 1200 - depth
            else:
                return depth - 1200

        # Check if we reached the depth limit
        if depth == self._max_depth:
            return self.__eval3(game)

        v = self._MAX_VALUE
        possible_moves = game.get_available_moves()

        for pm in possible_moves:
            coordinates, slide = pm
            next_position = deepcopy(game)
            next_position.single_move(coordinates, slide)

            v = min(v, self.__max_value(next_position, alpha, beta, depth + 1))
            
            if v <= alpha:
                return v;
                
            beta = min(beta, v);    

        return v

    def __max_value(self, game: Game, alpha: int, beta: int, depth: int) -> int:        
        '''Select the best move in the max layer of the min max algorithm.'''
        # Check if we are in a final position
        winner = game.check_winner()
        if winner is not None:
            if game.players[winner] is self:
                return 1200 - depth
            else:
                return depth - 1200

        # Check if we reached the depth limit
        if depth == self._max_depth:
            return self.__eval3(game)

        v = self._MIN_VALUE
        possible_moves = game.get_available_moves()

        for pm in possible_moves:
            coordinates, slide = pm
            next_position = deepcopy(game)
            next_position.single_move(coordinates, slide)

            v = max(v, self.__min_value(next_position, alpha, beta, depth + 1))
            
            if v >= beta:
                return v;
                
            alpha = max(alpha, v);    

        return v

    def __eval1(self, game: Game) -> float:
        '''
        First evaluation function to evaluate the position.
        It considers the cell with different weights and then it counts the ownership of all the cells.
        The player that owns the most valuable cells is the favourite in the position.
        '''
        cell_worth = np.array([
            [2, 3, 3, 3, 2],
            [3, 1, 1, 1, 3],
            [3, 1, 1, 1, 3],
            [3, 1, 1, 1, 3],
            [2, 3, 3, 3, 2],
        ])

        evaluation = 0
        for x in range(5):
            for y in range(5):
                if game.ownership_cell(self, (x, y)):
                    evaluation += cell_worth[(x, y)]
                else:
                    evaluation -= cell_worth[(x, y)]
        
        return evaluation
    
    def __check_symbol(self, game: Game, coordinates: Coordinates, added_eval: int) -> int:
        '''
        TODO: documentation
        '''
        evaluation = 0
        if game.ownership_cell(self, coordinates):
            evaluation += added_eval
        else:
            evaluation -= added_eval
        return evaluation

    def __check_for_threes_and_fours(self, game: Game, start: int, end: int, step: int) -> int:
        '''
        TODO: documentation
        '''
        evaluation = 0

        for i in  range(start, end, step):
            coordinates = (i % 5, int(i / 5))
            if game.check_sequence(i, i + 2*step, step):
                evaluation += self.__check_symbol(game, coordinates, 1)
            if i < (start + 2*step) and game.check_sequence(i, i + 3*step, step):
                evaluation += self.__check_symbol(game, coordinates, 3)

        return evaluation

    def __eval2(self, game: Game) -> float:
        '''
        TODO: documentation
        '''
        evaluation = 0
        main_diag_start = 0
        main_diag_step = 6
        secondary_diag_start = 4
        secondary_diag_step = 4

        evaluation += self.__check_for_threes_and_fours(game, main_diag_start, main_diag_start + 3*main_diag_step, main_diag_step)
        evaluation += self.__check_for_threes_and_fours(game, secondary_diag_start, secondary_diag_start + 3*secondary_diag_step, secondary_diag_step)

        for i in range(5):
            row_start = i*5
            row_step = 1
            col_start = i
            col_step = 5
            evaluation += self.__check_for_threes_and_fours(game, row_start, row_start + 3*row_step, row_step)
            evaluation += self.__check_for_threes_and_fours(game, col_start, col_start + 3*col_step, col_step)

        return evaluation

    def __eval3(self, game: Game) -> float:
        '''Just combine the two eval methods described before.'''
        return self.__eval1(game) + self.__eval2(game)
