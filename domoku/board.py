import matplotlib.pyplot as plt

from domoku.data import create_nxnx4
from domoku.ddpg import NxNx4Game
from domoku.field import GomokuField
from domoku.tools import GomokuTools
from domoku.constants import *
import pandas as pd


def maybe_convert(x):
    if type(x) != str:
        return x
    return ord(x)-64        
            
            
class GomokuBoard(GomokuField):
    
    def __init__(self, n, heuristics=None, disp_width=6, stones=None):
        stones = stones or []
        GomokuField.__init__(self, n, heuristics=heuristics)
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
        if self.cursor != len(self.stones)-1:
            raise(ValueError("Cursor not at end position."))
        x, y = self.stones[-1]
        self.stones = self.stones[:-1]
        self.compute_neighbourhoods(x, y, 'u')
        self.ctoggle()
        self.cursor = len(self.stones)-1

        return self
    
    
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
        xlines = [[[1, size], [y, y], '#E0E0E0'] for y in range(1, size+1)]
        ylines = [[[x, x], [1, size], '#E0E0E0'] for x in range(1, size+1)]
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
        x, y = self.stones[self.cursor]
        box = np.array(
            [[-0.6, 0.6, 0.6, -0.6, -0.6],
             [-0.6, -0.6, 0.6, 0.6, -0.6]])
        box = box + [[x], [y]]
        plt.plot(*box, color='w', zorder=30)

    def display_stones(self, axis):
        colors = ['black', 'white']
        for i in range(1, self.cursor + 2):
            x, y = self.stones[i-1][0:2]
            stc = colors[i % 2]
            fgc = colors[1 - i % 2]
            axis.scatter([x], [y], c=stc, s=self.stones_size(), zorder=10)
            self.display_cursor()
            plt.text(x, y, i, color=fgc, fontsize=12, zorder=20,
                     horizontalalignment='center', verticalalignment='center')

    @staticmethod
    def heatmap(q):
        image = np.squeeze((np.log((1 + q.numpy()))*99))
        image = (image / np.max(image, axis=None) * 255).astype(int)
        return image

    def game(self):
        state = create_nxnx4(self.N, stones=self.stones)
        return NxNx4Game(state)

    def display_heuristics(self, axis, cut_off=50):

        if self.heuristics is None:
            return

        position = create_nxnx4(15, stones=self.stones)
        q = self.heuristics(position)
        heatmap = np.squeeze(self.heatmap(q))

        for c in range(self.N):
            for r in range(self.N):
                value = heatmap[r][c]
                x, y = GomokuTools.m2b((r, c), self.N)
                if value >= cut_off:
                    color = f"#{value:02x}0000"
                    axis.scatter([x], [y], color=color, s=self.stones_size(), zorder=10)
            
                
    def stones_size(self):
        return 120 / self.N * self.disp_width**2
        

    def save(self, filename):
        df = pd.DataFrame(self.stones)
        df.to_csv(filename, header=False, index=False)

        
    def get_value(self, compute_scores=False):
        scores = self.get_clean_scores(compute_scores)
        o = scores[1-self.current_color]
        d = scores[self.current_color]
        return np.sum(o) - np.sum(d)
        

    def get_clean_scores(self, compute_scores=False, tag=0):
        """ 
        get the scores with the occupied positions zeroed out
        """
        if compute_scores:
            self.compute_all_scores()
            
        cp = self.scores.copy()
        for pos in self.stones:
            r, c = GomokuTools.b2m(pos, self.N)
            for color in [0, 1]:
                cp[color][r][c] = tag
        return cp
        
        
    @staticmethod        
    def from_csv(filename, heuristics, size=19, disp_width=10):
        stones = pd.read_csv(filename, header=None).values.tolist()
        return GomokuBoard(heuristics, size, disp_width, stones=stones)


#%%
