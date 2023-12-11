from tqdm import tqdm
from game import Game, Move, RandomPlayer, MyAgent, HumanPlayer

class Model(object):
    def __init__(self, player1: int, player2: int, name1: str='player1', name2: str='player2') -> None:
        self._players_map = {'RandomPlayer': RandomPlayer, 'HumanPlayer': HumanPlayer, 'MyAgent': MyAgent}
        self._player1 = self._players_map.get(player1)(name=name1)
        self._player2 = self._players_map.get(player2)(name=name2)
        self._players = [player1, player2]

    def training(self, rounds=1000, policy1: str=None, policy2: str=None) -> None:
        if not (self._player1.is_agent() or self._player2.is_agent()):
            print("ERROR: cannot start training with no agents.")
            return

        if self._player1.name == self._player2.name:
            self._player2.name = "player2"

        if policy1 is not None:
            if self._player1.is_agent():
                self._player1.load_policy(policy1)
            else:
                print(f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent.")
            
        if policy2 is not None:
            if self._player2.is_agent():
                self._player2.load_policy(policy2)
            else:
                print(f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent.")
            

        for i in tqdm(range(rounds)):
            game = Game()
            winner = game.play(self._player1, self._player2)

            if winner == 0:
                if self._player1.is_agent():
                    self._player1.feed_reward(1)
                if self._player2.is_agent():
                    self._player2.feed_reward(0)
            else:
                if self._player1.is_agent():
                    self._player1.feed_reward(0)
                if self._player2.is_agent():
                    self._player2.feed_reward(1)

        self._player1.save_policy()
        self._player2.save_policy()

    def testing(self, rounds=1000, policy: str=None) -> None:
        wins = [0, 0]
        if not (self._player1.is_agent() or self._player2.is_agent()):
            print("ERROR: cannot start testing with no agents.")
            return

        if policy1 is not None:
            if self._player1.is_agent():
                self._player1.load_policy(policy1)
            else:
                print(f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent.")
            
        if policy2 is not None:
            if self._player2.is_agent():
                self._player2.load_policy(policy2)
            else:
                print(f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent.")

        for i in tqdm(range(rounds)):
            game = Game()
            winner = game.play(self._player1, self._player2)
            wins[0] += (1 - winner)
            wins[1] += winner

        win_rate_p1 = (rounds - wins[1])/rounds

        print(f"The results of the match [{type(self._player1)} vs {type(self._player2)}] are shown here:")
        print(f"The win rate for the player1 is {win_rate_p1:.2f} on a total of {rounds} matches")

    def single_match(self, policy1: str=None, policy2: str=None) -> None:        
        if policy1 is not None:
            if isinstance(self._player1, MyAgent):
                self._player1.load_policy(policy1)
            else:
                print(f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent")
        if policy2 is not None:
            if isinstance(self._player2, MyAgent):
                self._player2.load_policy(policy2)
            else:
                print(f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent")

        game = Game()
        game.play(self._player1, self._player2)

        print(f"{game.winner} has won!")