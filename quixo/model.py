from game import Game, Move, RandomPlayer, MyAgent, HumanPlayer

class Model(object):
    def __init__(self, player1: int, player2: int, name1: str='player1', name2: str='player2') -> None:
        self._players_map = {'RandomPlayer': RandomPlayer, 'HumanPlayer': HumanPlayer, 'MyAgent': MyAgent}
        self._player1 = self._players_map.get(player1)(name=name1)
        self._player2 = self._players_map.get(player2)(name=name2)
        self._players = [player1, player2]

    def training(self, rounds=1000, policy: str=None) -> None:
        if policy is not None:
            self._player1.load_policy(policy)

        for i in tqdm(range(rounds)):
            game = Game()
            winner = game.play(self._player1, player2)

            if winner == 0:
                self._player1.feed_reward(1)
            else:
                self._player1.feed_reward(0)

        self._player1.save_policy()

    def testing(self, rounds=1000, policy: str=None) -> None:
        losses = 0

        if policy is not None:
            self._player1.load_policy(policy)

        for i in tqdm(range(rounds)):
            game = Game()
            losses += game.play(self._player1, self._player2)

        win_rate = (rounds - losses)/rounds

        print(f"The win rate for the agent is {win_rate:.2f} on a total of {rounds} matches")

    def play_against_human(self, policy: str=None) -> None:
        if policy is not None:
            self._player1.load_policy(policy)

        game = Game()
        game.play(self._player1, self._player2)

        print(f"{game.winner} has won!")