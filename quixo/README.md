## Setup Libraries
```py
pip install -r requirements.txt
```

## Literature
1. [RL Tic-Tac-Toe](https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542)

## Log
1. **12th December**: By training an Agent against a Random Player for 50 thousends games, the result on test shows that the Agent wins 87.80% of the times.
The test was conducted on 1000 games against a Random Player. The resulting policy file, containing (state,value) pairs is roughly 60MB.
```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo> python .\main.py
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:50<00:00,  5.85it/s]
The results of the match <class 'game.MyAgent'> vs <class 'game.RandomPlayer'> are shown here:
The win rate for the player1 is 87.80% on a total of 1000 matches
```