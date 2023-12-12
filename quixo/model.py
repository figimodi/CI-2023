from collections import namedtuple
from tqdm import tqdm
from game import Game, Move, RandomPlayer, MyAgent, HumanPlayer

Rewards = namedtuple('Rewards', ['losing', 'winning'])

class Model(object):
    def __init__(self, player1: int, player2: int, name1: str = 'player1', name2: str = 'player2', policy1: str = None, policy2: str = None, rewards: tuple[float, float] = (0, 1)) -> None:
        self._players_map = {'RandomPlayer': RandomPlayer, 'HumanPlayer': HumanPlayer, 'MyAgent': MyAgent}
        self._player1 = self._players_map.get(player1)(name=name1)
        self._player2 = self._players_map.get(player2)(name=name2)
        self._rewards = Rewards(rewards[0], rewards[1]) 

        if policy1 is not None:
            if self._player1.is_agent():
                self._player1.load_policy(policy1)
            else:
                # TODO: color the warnings
                print(f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent.")
            
        if policy2 is not None:
            if self._player2.is_agent():
                self._player2.load_policy(policy2)
            else:
                # TODO: color the warnings
                print(f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent.")

    def set_reward_losing(self, reward: float) -> None:
        self._rewards.losing = reward

    def set_reward_winning(self, reward: float) -> None:
        self._rewards.winning = reward

    def set_policies(self, policy1: str = None, policy2: str = None) -> None:
        if policy1 is not None:
            if self._player1.is_agent():
                self._player1.load_policy(policy1)
            else:
                # TODO: color the warnings
                print(f"WARNING: policy1 was not loaded to player1, since it's not an instance of MyAgent.")
            
        if policy2 is not None:
            if self._player2.is_agent():
                self._player2.load_policy(policy2)
            else:
                # TODO: color the warnings
                print(f"WARNING: policy2 was not loaded to player2, since it's not an instance of MyAgent.")

    def training(self, rounds=50000) -> None:
        if not (self._player1.is_agent() or self._player2.is_agent()):
            # TODO: color the errors
            sys.exit("ERROR: cannot start training with no agents.")
            return

        if self._player1.name == self._player2.name:
            self._player2.name = "player2"   

        game = Game()
        for i in tqdm(range(rounds)):
            winner = game.play(self._player1, self._player2)

            if winner == 0:
                if self._player1.is_agent():
                    self._player1.feed_reward(self._rewards.winning)
                if self._player2.is_agent():
                    self._player2.feed_reward(self._rewards.losing)
            elif winner == 1:
                if self._player1.is_agent():
                    self._player1.feed_reward(self._rewards.losing)
                if self._player2.is_agent():
                    self._player2.feed_reward(self._rewards.winning)

            if self._player1.is_agent():
                self._player1.reset_states()
            if self._player2.is_agent():
                self._player2.reset_states()

            game.reset()

        if self._player1.is_agent():
            self._player1.save_policy()
        if self._player2.is_agent():
            self._player2.save_policy()

    def testing(self, rounds=5000) -> None:
        wins = [0, 0]

        if self._player1.is_agent():
            self._player1.set_exp_rate(0)

        if self._player2.is_agent():
            self._player2.set_exp_rate(0)

        game = Game()
        for i in tqdm(range(rounds)):
            winner = game.play(self._player1, self._player2)
            wins[0] += (1 - winner)
            wins[1] += winner
            game.reset()

        win_rate_p1 = (wins[0]/rounds)*100

        print(f"The results of the match {type(self._player1)} vs {type(self._player2)} are shown here:")
        print(f"The win rate for the player1 is {win_rate_p1:.2f}% on a total of {rounds} matches")

    def single_match(self) -> None:        
        game = Game()
        game.play(self._player1, self._player2)

        print(f"{game.winner} has won!")