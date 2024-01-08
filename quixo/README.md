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

3. **16th December**: After the bug fix on the code this are the performance for the RL Agent (the frist 2000 matches were of training):
```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo\src> python .\main.py
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [2:50:10<00:00,  5.11s/it]
100%|
███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [1:57:05<00:00, 14.05s/it]
The results of the match <class 'player.RLPlayer'> vs <class 'player.RandomPlayer'> are shown here:
The win rate for the player1 is 63.80% on a total of 500 matches
```

4. **7th January**:
```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo\src> python .\main.py
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [1:53:13<00:00,  3.40s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [10:00<00:00,  1.20s/it]
The results of the match <class 'player.RLPlayer'> vs <class 'player.RandomPlayer'> are shown here:
The win rate for the player1 is 100.00% on a total of 500 matches
```

```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo\src> python .\main.py
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [2:34:21<00:00,  2.32s/it]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [48:17<00:00,  5.79s/it]
The results of the match <class 'player.RandomPlayer'> vs <class 'player.RLPlayer'> are shown here:
The win rate for the player1 is 30.40% on a total of 500 matches
```

5. **8th January**:
Games between RL vs RL (trained on each other)
```
PS C:\Users\grfil\OneDrive\Documenti\PoliTo\Computational Intelligence\CI-2023\quixo\src> python.exe .\main.py
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [1:07:04<00:00,  8.05s/it]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [09:27<00:00,  2.27s/it]
The results of the match <class 'player.RLPlayer'> vs <class 'player.RLPlayer'> are shown here:
The win rate for the player1 is 100.00% on a total of 250 matches
```