import numpy as np
import matplotlib.pyplot as plt

import aegomoku.gomoku_board as new_board
import aegomoku.tools as gt

from aegomoku.interfaces import Move


def maybe_convert(x):
    if type(x) != str:
        return x
    return ord(x)-64        


BLACK, WHITE = 0, 1


class MplBoard:
    """
    A display tools based on Matplotlib
    """
    
    def __init__(self, n, heuristics=None, disp_width=6, stones=None, suppress_move_numbers=False):
        self.N = n
        self.heuristics = heuristics
        self.suppress_move_numbers = suppress_move_numbers
        stones = stones or []
        if isinstance(stones, str):
            stones = gt.string_to_stones(stones)
        self.disp_width = disp_width
        self.stones = []
        self.current_color = BLACK
        self.cursor = -1
        for stone in stones:
            self.set(*stone)


    def _is_valid(self, index):
        """
        checks the array indexes (not the board coordinates!)
        """
        return 0 <= index[0] < self.N and 0 <= index[1] < self.N
    
 
    def ctoggle(self):
        """
        toggle current color
        """
        self.current_color = 1 - self.current_color
        return self.current_color
        
        
    def set(self, x, y=None):
        """
        x,y: 1-based indices of the board, x may be an uppercase letter
        """
        if y is None and isinstance(x, Move):
            x, y = x.x, x.y

        x = maybe_convert(x)

        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))
        if (x, y) in self.stones:
            print(self.stones)
            raise ValueError(f"Position ({x}, {y}) is occupied.")
        if not self._is_valid((self.N-y, x-1)):
            raise ValueError(f"Not a valid move: ({x}, {y}). Beyond board boundary.")
        
        self.stones.append((x, y))
        self.ctoggle()
        self.cursor = len(self.stones)-1
        
        return self


    def undo(self):
        raise NotImplementedError("Currently not supported.")

    
    def bwd(self, n=1):
        if n > 1:
            self.bwd()
            self.bwd(n-1)
            return self
            
        if self.cursor >= 0:
            self.cursor -= 1
            self.ctoggle()
        return self

    
    def fwd(self, n=1):
        if n > 1:
            self.fwd()
            self.fwd(n-1)
            return self
        if self.cursor < len(self.stones)-1:
            self.cursor += 1
            self.ctoggle()
        return self

    def display(self):
        size = self.N

        fig, axis = plt.subplots(figsize=(self.disp_width, self.disp_width))
        axis.set_xlim([0, size+1])
        axis.set_ylim([0, size+1])
        plt.xticks(range(1, size+1), [chr(asci) for asci in range(65, 65+size)])
        plt.yticks(range(1, size+1))
        axis.set_facecolor('#404080')
        xlines = [[[1, size], [y_, y_], '#E0E0E0'] for y_ in range(1, size+1)]
        ylines = [[[x_, x_], [1, size], '#E0E0E0'] for x_ in range(1, size+1)]
        ylines = np.reshape(
            np.array(xlines + ylines, dtype=object), [-1])
        axis.plot(*ylines)
        self.display_helpers(axis)

        if self.cursor >= 0:
            self.display_stones(axis)

        self.display_heuristics(axis)

    def display_helpers(self, axis):
        if self.N == 15:
            axis.scatter([4, 4, 12, 12, 8], [4, 12, 12, 4, 8], 
                         s=self.disp_width**2, c='#E0E0E0')
        elif self.N == 19:
            axis.scatter([4, 4, 4, 10, 16, 16, 16, 10, 10], 
                         [4, 10, 16, 16, 16, 10, 4, 4, 10], 
                         s=self.disp_width**2, c='#E0E0E0')
        elif self.N == 20:
            axis.scatter([6, 6, 15, 15], [6, 15, 15, 6], 
                         s=self.disp_width**2, c='#E0E0E0')

    def display_cursor(self):
        x_, y_ = self.stones[self.cursor]
        box = np.array(
            [[-0.6, 0.6, 0.6, -0.6, -0.6],
             [-0.6, -0.6, 0.6, 0.6, -0.6]])
        box = box + [[x_], [y_]]
        plt.plot(*box, color='w', zorder=30)

    def display_stones(self, axis):
        colors = ['black', 'white']
        for i in range(1, self.cursor + 2):
            x_, y_ = self.stones[i-1][0:2]
            stc = colors[i % 2]
            fgc = colors[1 - i % 2]
            axis.scatter([x_], [y_], c=stc, s=self.stones_size(), zorder=10)
            if not self.suppress_move_numbers:
                self.display_cursor()
                plt.text(x_, y_, i, color=fgc, fontsize=12, zorder=20,
                         horizontalalignment='center', verticalalignment='center')

    @staticmethod
    def heatmap(q):
        image = np.squeeze((np.log((1 + q))*99))
        image = (image / np.max(image, axis=None) * 255).astype(int)
        return image

    def display_heuristics(self, axis, cut_off=50):

        if self.heuristics is None:
            return

        position = new_board.GomokuBoard(self.N, stones=gt.stones_to_string(self.stones[:self.cursor+1]))
        if isinstance(self.heuristics, list):
            q = np.reshape(self.heuristics, (self.N, self.N))
        else:
            q, _ = self.heuristics(position.canonical_representation())

        heatmap = np.squeeze(self.heatmap(q))

        for c in range(self.N):
            for r in range(self.N):
                value = heatmap[r][c]
                x_, y_ = gt.m2b((r, c), self.N)
                if value >= cut_off:
                    color = f"#{value:02x}0000"
                    axis.scatter([x_], [y_], color=color, s=self.stones_size() * .5, zorder=10)
            
                
    def stones_size(self):
        return 120 / self.N * self.disp_width**2
