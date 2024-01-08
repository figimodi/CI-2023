from copy import deepcopy
from enum import Enum
from typing import NamedTuple
import numpy as np
import sys
import io

class Players(NamedTuple):
    p1: 'Player'
    p2: 'Player'

class Coordinates(NamedTuple):
    '''Coordinates on the board.'''
    x: int
    y: int

class Slide(Enum):
    '''Type of slides, for example: TOP means that we push from the TOP to the BOTTOM.'''
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

class Move(NamedTuple):
    '''Combination of coordinates and type of slide.'''
    coordinates: Coordinates
    slide: Slide

class Game(object):
    '''
    This class represents the state of the game, along with the board,
    the current turn, the available moves in the position and the winner when
    the match is over.
    '''

    def __init__(self, player1: int, player2: int, name1: str = 'player1', name2: str = 'player2') -> None:
        self.winner: int = None
        self._current_player_idx = 1
        self.players = Players(player1, player2)
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
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1.'''
        # Check the rows
        for x in range(self._board.shape[0]):
            if all(self._board[x, :] == self._board[x, 0]) and self._board[x, 0] != -1:
                return self._board[x, 0]
        
        # Check the columns
        for y in range(self._board.shape[0]):
            if all(self._board[:, y] == self._board[0, y]) and self._board[0, y] != -1:
                return self._board[0, y]
        
        # Check the diagonals
        if all([self._board[x, x] for x in range(self._board.shape[0])] == self._board[0, 0]) and self._board[0, 0] != -1:
            return self._board[0, 0]
        if all([self._board[x, -x-1] for x in range(self._board.shape[0])] == self._board[0, -1]) and self._board[0, -1] != -1:
            return self._board[0, -1]

        # TODO: add the case in which there are two different five-in-a-row combinations -> wins the opponent

        return None

    def play(self) -> int:
        '''Play the game. Returns the winning player.'''
        winner = None
        while winner is None:
            self._current_player_idx += 1
            self._current_player_idx %= len(self.players)
            valid = False
            while not valid:
                coordinates, slide = self.players[self._current_player_idx].make_move(self)
                valid = self.__move(coordinates, slide)
                if not valid and self.players[self._current_player_idx].is_human():
                    print("That's an invalid move, please reenter your move:")
            
            # Activate the following print if at least one of the player is human
            if self.players[0].is_human() or self.players[0].is_human():
                print(self)
            
            print(self)

            # Check if the game has a winner
            winner = self.check_winner()

        # Assing the winner to the class and return the corresponing ID
        self.winner = winner
        
        return winner

    def single_move(self, coordinates: Coordinates, slide: Slide) -> None:
        '''Makes a single move on the board.'''
        ok = self.__move(coordinates, slide)
        assert ok == True

    def get_available_moves(self) -> list[Move]:
        '''Return the possible moves in the current position.'''
        # Calculate all possible available moves from the current state
        self.__available_moves()
        return self._available_moves_list

    def ownership_cell(self, player: 'Player', coordinates: Coordinates) -> bool:
        '''TODO: documentation'''
        return self._board[coordinates] != -1 and player is self.players[self._board[coordinates]]
        
    def check_sequence(self, start: int, end: int, step: int) -> bool:
        '''TODO: documentation'''
        if self._board[start % 5, int(start / 5)] == -1:
            return False

        result = True
        squares_flat = [i for i in range(start, end, step)]
        squares = list(map(lambda s : (s % 5, int(s / 5)), squares_flat))
        
        for s in range(len(squares) - 1):
            if self._board[squares[s]] != self._board[squares[s + 1]]:
                return False

        return result

    def get_hash(self) -> str:
        '''Hashes the state of the board.'''
        return str(self._board.reshape(5 * 5))

    def reset(self) -> None:
        '''Reset the state of the board.'''
        self.winner = None
        self._current_player_idx = 1
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self._available_moves_list = list()
    
    def __move(self, coordinates: Coordinates, slide: Slide, mock: bool=False) -> bool:
        '''Perform a move.'''
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
        '''Take piece  and 'flip it' facing the player symbol.'''
        # Acceptable only if in border
        acceptable: bool = ((coordinates[0] == 0 and coordinates[1] < 5) or (coordinates[0] == 4 and coordinates[1] < 5) or (
            coordinates[1] == 0 and coordinates[0] < 5) or (coordinates[1] == 4 and coordinates[0] < 5)) and (self._board[coordinates] < 0 or self._board[coordinates] == self._current_player_idx)
        if acceptable:
            self._board[coordinates] = self._current_player_idx
        return acceptable

    def __slide(self, coordinates: Coordinates, slide: Slide) -> bool:
        '''Slide the other pieces.'''
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        if coordinates not in SIDES:
            acceptable_top: bool = coordinates[0] == 0 and (
                slide == Slide.BOTTOM or slide == Slide.LEFT or slide == Slide.RIGHT)
            acceptable_bottom: bool = coordinates[0] == 4 and (
                slide == Slide.TOP or slide == Slide.LEFT or slide == Slide.RIGHT)
            acceptable_left: bool = coordinates[1] == 0 and (
                slide == Slide.BOTTOM or slide == Slide.TOP or slide == Slide.RIGHT)
            acceptable_right: bool = coordinates[1] == 4 and (
                slide == Slide.BOTTOM or slide == Slide.TOP or slide == Slide.LEFT)
        else:
            # top left
            acceptable_top: bool = coordinates == (0, 0) and (
                slide == Slide.BOTTOM or slide == Slide.RIGHT)
            # top right
            acceptable_right: bool = coordinates == (4, 0) and (
                slide == Slide.TOP or slide == Slide.RIGHT)
            # bottom left
            acceptable_left: bool = coordinates == (0, 4) and (
                slide == Slide.BOTTOM or slide == Slide.LEFT)
            # bottom right
            acceptable_bottom: bool = coordinates == (4, 4) and (
                slide == Slide.TOP or slide == Slide.LEFT)

        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        
        # If the move is allowed, then slide the pieces on the corresponding column/row
        if acceptable:
            if slide == Slide.TOP or slide == Slide.BOTTOM:
                column = [row[coordinates[1]] for row in self._board]
                column = np.append(column, column[coordinates[0]])
                column = np.delete(column, coordinates[0])
                if slide == Slide.TOP:
                    column = np.concatenate((column[-1:], column[:-1]), axis=0)
                for i, row in enumerate(self._board):
                    row[coordinates[1]] = column[i]
            elif slide == Slide.LEFT or Slide.RIGHT:
                row = self._board[coordinates[0]]
                row = np.append(row, row[coordinates[1]])
                row = np.delete(row, coordinates[1])
                if slide == Slide.LEFT:
                    row = np.concatenate((row[-1:], row[:-1]), axis=0)
                self._board[coordinates[0]] = row

        return acceptable

    def __available_moves(self) -> None:
        '''Calculate all the possible moves from the current state.'''
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
