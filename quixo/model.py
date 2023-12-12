from typing import NamedTuple
from tqdm import tqdm
from colorama import init, Fore
from game import Game, Move, Player, RandomPlayer, MyAgent, HumanPlayer

init(autoreset=True)

class Players(NamedTuple):
    p1: Player
    p2: Player

class Rewards(NamedTuple):
    '''Represents the rewards (in float) to give to the agents after the outcome of a game'''
    winning: float = 1.
    loosing: float = 0.

class Model(object):
    '''
    Class for training and testing the RL model.
    It also contains a method to play a single game.
    '''

    def __init__(self, player1: int, player2: int, name1: str = 'player1', name2: str = 'player2', policy1: str = None, policy2: str = None, reward_winning: float = 1., reward_losing: float = 0.) -> None:
        self._players_map = {'RandomPlayer': RandomPlayer, 'HumanPlayer': HumanPlayer, 'MyAgent': MyAgent}
        p1 = self._players_map.get(player1)(name=name1)
        p2 = self._players_map.get(player2)(name=name2)
        self._players = Players(p1, p2)
        self._rewards = Rewards(reward_winning, reward_losing) 

        if policy1 is not None:
            if self._players.p1.is_agent():
                self._player1.load_policy(policy1)
            else:
                print(Fore.YELLOW + f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent.")
            
        if policy2 is not None:
            if self._players.p2.is_agent():
                self._players.p2.load_policy(policy2)
            else:
                print(Fore.YELLOW + f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent.")

    def set_policies(self, policy1: str = None, policy2: str = None) -> None:
        '''Set policies for one or both players'''
        if policy1 is not None:
            if self._players.p1.is_agent():
                self._players.p1.load_policy(policy1)
            else:
                print(Fore.YELLOW + f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent.")
            
        if policy2 is not None:
            if self._players.p2.is_agent():
                self._players.p2.load_policy(policy2)
            else:
                print(Fore.YELLOW + f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent.")

    def training(self, rounds=50000) -> None:
        '''Train the agent(s)'''
        # Check if at least one player is an agent
        if not (self._players.p1.is_agent() or self._players.p2.is_agent()):
            sys.exit(FORE.RED + "ERROR: cannot start training with no agents.")
            return

        # Change name of second player if they have the same name (to prevent saving policies on same file and losing one configuration)
        if self._players.p1.name == self._players.p2.name:
            self._players.p2.name = "player2"   

        # Starting the training
        game = Game()
        for i in tqdm(range(rounds)):
            winner = game.play(self._players.p1, self._players.p2)

            # Feed rewards to the players depending on who won
            if winner == 0:
                if self._players.p1.is_agent():
                    self._players.p1.feed_reward(self._rewards.winning)
                if self._players.p2.is_agent():
                    self._players.p2.feed_reward(self._rewards.losing)
            elif winner == 1:
                if self._players.p1.is_agent():
                    self._players.p1.feed_reward(self._rewards.losing)
                if self._players.p2.is_agent():
                    self._players.p2.feed_reward(self._rewards.winning)

            # Reset the paths of the players trhough the game, to start a new one
            if self._players.p1.is_agent():
                self._players.p1.reset_states()
            if self._players.p2.is_agent():
                self._players.p2.reset_states()

            # Reset the game (board ecc...)
            game.reset()

        # Save the policies of the players for future use
        if self._players.p1.is_agent():
            self._players.p1.save_policy()
        if self._players.p2.is_agent():
            self._players.p2.save_policy()

    def testing(self, rounds=5000) -> None:
        '''Testing the agent(s)'''
        wins = [0, 0]

        # Set exploration rate to 0, since we don't want to explore anymore
        if self._players.p1.is_agent():
            self._players.p1.set_exp_rate(0)
        if self._players.p2.is_agent():
            self._players.p2.set_exp_rate(0)

        # Start testing
        game = Game()
        for i in tqdm(range(rounds)):
            winner = game.play(self._players.p1, self._players.p2)
            wins[0] += (1 - winner)
            wins[1] += winner
            
            # Reset the game for a new match
            game.reset()

        # Calculate win rate for player1
        win_rate_p1 = (wins[0]/rounds)*100

        print(f"The results of the match {type(self._players.p1)} vs {type(self._players.p2)} are shown here:")
        print(f"The win rate for the player1 is {win_rate_p1:.2f}% on a total of {rounds} matches")

    def single_match(self) -> None:        
        '''Method for playing a single match (To play with an Human for example)'''
        game = Game()
        game.play(self._players.p1, self._players.p2)

        print(f"{game.winner} has won!")