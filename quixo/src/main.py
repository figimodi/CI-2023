import random
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from copy import copy
from game import Move, Game, MyGame, Player


class MyPlayer(Player):
    '''Skeleton for different types of players.'''

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()
        pass

    def __str__(self) -> str:
        return f'{self.name}'

    def __eq__(self, other) -> bool:
        if self.name == other.name:
            return True
        else:
            return False

    def is_RLagent(self) -> bool:
        '''Tells if the object Player is an instance of the class RLPlayer.'''
        return isinstance(self, RLPlayer)

    def is_human(self) -> bool:
        '''Tells if the object Player is an instance of the class HumanPlayer.'''
        return isinstance(self, HumanPlayer)


class RandomPlayer(MyPlayer):
    '''
    This class contains the implementation for the Random Player.
    The make_move method is overwrite by simply choosing a random move.
    '''

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class HumanPlayer(MyPlayer):
    '''
    Class for representing a Human Player,
    the make_move method is overwritten with just inputs from the keyboard,
    such that an human can play.
    '''
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''Asks for (row, column) and slide, that combined represents a move on the board.'''
        game.print()
        x = int(input("Input the column coordinate (from 0 to 4):"))
        y = int(input("Input the row coordinate (from 0 to 4):"))
        move = int(input("Input the slide type (0:top, 1:bottom, 2:left, 3:right):"))
        
        return (x, y), Move(move)


class RLPlayer(MyPlayer):
    '''
    The class contains the implementation for the Agent using Reinforcement Learning.
    It has several methods and attributes:
    lr:           learning rate: the magnitude of the rewards given to the states.
    exp_rate:     exploration rate, probability of chosing random moves and exploring new strategies.
    decay_gamma:  discount, act as a discount factor.
    states:       contains the path of the player in the game, storing all the states from start to finish.
    state_values: dictionary that conatins (state, value) pair for each state that the agent knows.
    '''
    def __init__(self, name: str, exp_rate=0.3, lr=0.2) -> None:
        super().__init__(name)
        self._states = list()
        self._state_value = defaultdict()
        self._lr = lr
        self._exp_rate = exp_rate
        self._decay_gamma = 0.9

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''Returns the coordinates, slide tuple by choosing the best candidate move.'''
        from_pos, move = self.__choose_action(game)
        return from_pos, move

    def feed_reward(self, reward: float) -> None:
        '''Gives rewards to the actions perfomed in the match.'''
        # Starting from the last state of the game, update the value associated to that state
        for st in reversed(self._states):
            if self._state_value.get(st) is None:
                self._state_value[st] = 0
            self._state_value[st] += self._lr * (self._decay_gamma * reward - self._state_value[st])
            reward = self._state_value[st]

    def __choose_action(self, game: 'MyGame') -> Move:
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
                from_pos, move = pm
                prev_value = game.get_board()
                game.single_move(from_pos, move)
                next_hash = game.get_hash()
                game.set_board(prev_value)
                value = 0 if self._state_value.get(next_hash) is None else self._state_value.get(next_hash)

                # If we get a state that has a better score, than we save the action that leads to that state
                if value >= value_max:
                    value_max = value
                    action = pm

        # Briefly update the path of the player through the game (_states.append)
        from_pos, move = action
        prev_value = game.get_board()
        game.single_move(from_pos, move)
        next_hash = game.get_hash()
        game.set_board(prev_value)
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

    def set_learning_rate(self, lr: float) -> None:
        self._lr = lr


class MinMaxPlayer(MyPlayer):
    '''
    The class contains the implementation for the Agent using the Min-Max algorithm with alpha beta pruning.
    '''

    def __init__(self, name: str, max_depth: int = 3, bot_symbol: int = 0) -> None:
        super().__init__(name)
        self._MIN_VALUE = -10000
        self._MAX_VALUE = 10000
        self._max_depth = max_depth
        self._bot_symbol = bot_symbol

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        '''Return the coordinates, slide tuple that gives the best evaluation of the posistion with a certain depth.'''
        from_pos, move = self.__min_max_decision(game)
        return from_pos, move

    def set_max_depth(self, max_depth: int) -> None:
        '''Change the maximum depth that the engine can reach.'''
        self._max_depth = max_depth

    def __min_max_decision(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        '''Return the best move according to the min max algorithm, with the help of alpha beta pruning.'''
        best_eval = self._MIN_VALUE
        alpha = self._MIN_VALUE
        beta = self._MAX_VALUE

        possible_moves = game.get_available_moves(clear=False)
        action = random.choice(possible_moves)
        for pm in possible_moves:
            from_pos, move = pm
            prev_value = copy(game.get_board())
            prev_player = game.get_current_player()
            game.single_move(from_pos, move)
            game.set_current_player(1 - prev_player)
            child_evaluation = self.__min_value(game, alpha, beta, 1)
            game.set_board(prev_value)
            game.set_current_player(prev_player)
            if child_evaluation > best_eval:
                best_eval = child_evaluation
                action = pm

            alpha = max(alpha, best_eval)

        return action
    
    def __min_value(self, game: 'MyGame', alpha: int, beta: int, depth: int) -> int:
        '''Select the best move in the min layer of the min max algorithm.'''
        # Check if we are in a final position
        winner = game.check_winner()
        if winner == self._bot_symbol:
            return self._MAX_VALUE
        elif winner != -1:
            return self._MIN_VALUE

        # Check if we reached the depth limit
        if depth == self._max_depth:
            return self.__eval3(game)

        best_eval = self._MAX_VALUE
        possible_moves = game.get_available_moves(clear=False)

        for pm in possible_moves:
            from_pos, move = pm
            prev_value = copy(game.get_board())
            prev_player = game.get_current_player()
            game.single_move(from_pos, move)
            game.set_current_player(1 - prev_player)
            best_eval = min(best_eval, self.__max_value(game, alpha, beta, depth + 1))      
            game.set_board(prev_value)  
            game.set_current_player(prev_player)
            
            beta = min(beta, best_eval)    
            if beta <= alpha:
                break
            

        return best_eval

    def __max_value(self, game: 'MyGame', alpha: int, beta: int, depth: int) -> int:        
        '''Select the best move in the max layer of the min max algorithm.'''
        # Check if we are in a final position
        winner = game.check_winner()
        if winner == self._bot_symbol:
            return self._MAX_VALUE
        elif winner != -1:
            return self._MIN_VALUE

        # Check if we reached the depth limit
        if depth == self._max_depth:
            return self.__eval3(game)

        best_eval = self._MIN_VALUE
        possible_moves = game.get_available_moves(clear=False)

        for pm in possible_moves:
            from_pos, move = pm
            prev_value = copy(game.get_board())
            prev_player = game.get_current_player()
            game.single_move(from_pos, move)
            game.set_current_player(1 - prev_player)
            best_eval = max(best_eval, self.__min_value(game, alpha, beta, depth + 1))
            game.set_board(prev_value)
            game.set_current_player(prev_player)
            
            alpha = max(alpha, best_eval)
            if beta <= alpha:
                break    

        return best_eval

    def __eval1(self, game: 'MyGame') -> float:
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
                own = game.ownership_cell(self._bot_symbol, (x, y))
                if own:
                    evaluation += cell_worth[(x, y)]
                elif own is not None:
                    evaluation -= cell_worth[(x, y)]
        
        return evaluation
    
    def __check_symbol(self, game: 'MyGame', from_pos: tuple[int, int], added_eval: int) -> int:
        '''
        TODO: documentation
        '''
        evaluation = 0
        if game.ownership_cell(self._bot_symbol, (from_pos[1], from_pos[0])):
            evaluation += added_eval
        else:
            evaluation -= added_eval
        return evaluation

    def __check_for_threes_and_fours(self, game: 'MyGame', start: int, end: int, step: int) -> int:
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

    def __eval2(self, game: 'MyGame') -> float:
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

    def __eval3(self, game: 'MyGame') -> float:
        '''Just combine the two eval methods described before.'''
        return self.__eval1(game) + self.__eval2(game)


if __name__ == '__main__':
    g = MyGame()
    player1 = MinMaxPlayer("p1")
    player2 = RandomPlayer("p2")

    # ## TRAINING RL
    # training_rounds = 8000
    testing_rounds = 10

    # # Starting the training
    # player1.load_policy('../policies/policy_p1')

    # for i in tqdm(range(training_rounds)):
    #     winner = g.play(player1, player2)

    #     # Feed rewards to the players depending on who won
    #     if winner == 0:
    #         player1.feed_reward(1)
    #     elif winner == 1:
    #         player1.feed_reward(0)

    #     # Reset the paths of the players trhough the game, to start a new one
    #     player1.reset_states()

    #     # Reset the game (board ecc...)
    #     g.reset()

    # player1.save_policy()

    # ## TESTING RL
    # player1.load_policy('../policies/policy_p1')
    # player1.set_exp_rate(0)
    # wins = [0, 0]

    # # Start testing
    # for i in tqdm(range(testing_rounds)):
    #     winner = g.play(player1, player2)
    #     wins[0] += (1 - winner)
    #     wins[1] += winner
        
    #     # Reset the game for a new match
    #     g.reset()

    # # Calculate win rate for player1
    # win_rate_p1 = (wins[0]/testing_rounds)*100

    # print(f"The results of the match {type(player1)} vs {type(player2)} are shown here:")
    # print(f"The win rate for the player1 is {win_rate_p1:.2f}% on a total of {testing_rounds} matches")

    ## TESTING MinMaxPlayer
    wins = [0, 0]

    # Start testing
    for i in tqdm(range(testing_rounds)):
        winner = g.play(player1, player2)
        wins[0] += (1 - winner)
        wins[1] += winner
        
        # Reset the game for a new match
        g.reset()

    # Calculate win rate for player1
    win_rate_p1 = (wins[0]/testing_rounds)*100

    print(f"The results of the match {type(player1)} vs {type(player2)} are shown here:")
    print(f"The win rate for the player1 is {win_rate_p1:.2f}% on a total of {testing_rounds} matches")
