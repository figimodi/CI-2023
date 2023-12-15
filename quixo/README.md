## Setup Libraries
```py
pip install -r requirements.txt
```

## Literature
1. [RL Tic-Tac-Toe](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)
2. [Quixo Bot C++](https://github.com/DobrinTs/Quixo-Bot)
3. [RL Q-learning Tic-Tac-Toe](https://towardsdatascience.com/an-ai-agent-learns-to-play-tic-tac-toe-part-3-training-a-q-learning-rl-agent-2871cef2faf0)

## Log
1. **12th December**: By training an Agent against a Random Player for 50 thousends games, the result on test shows that the Agent wins 87.80% of the times.
The test was conducted on 1000 games against a Random Player. The resulting policy file, containing (state,value) pairs is roughly 60MB.
```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo> python .\main.py
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:50<00:00,  5.85it/s]
The results of the match <class 'game.MyAgent'> vs <class 'game.RandomPlayer'> are shown here:
The win rate for the player1 is 87.80% on a total of 1000 matches
```

2. **15th December**: Testing the MinMaxPlayer against a Random Player on 100 games, the results are the following, with the MinMaxPlayer winning 92% of the times:
```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo\src> python .\main.py
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [1:09:25<00:00, 41.66s/it]100 [10:20<58:51, 41.55s/it]
The results of the match <class 'player.MinMaxPlayer'> vs <class 'player.RandomPlayer'> are shown here:
The win rate for the player1 is 92.00% on a total of 100 matches
```