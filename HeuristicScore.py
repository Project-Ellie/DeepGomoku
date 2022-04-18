import numpy as np
from GomokuTools import N_9x9

def to_bits (line):
    n = N_9x9().setline('e', line)
    line = n.as_bits()[0]    
    return line
               
class HeuristicScore:
    
    def __init__(self, kappa0=3, kappa1=5):
        self.kappa0 = kappa0
        self.kappa1 = kappa1
        
    def f_range(self, line, c=0, edges=(None, None)):
        """
        The largest adversary-free range within a given line
        
        Args:
            line: 8x2 integer array that represents the stones
            c:    0 to look at black, 1 to consider white
        """

        i=3
        while i >= 0 and line[1-c][i] == 0 and i != edges[0]:
            i-=1
        left = i + 1
        i=4
        while i <= 7 and line[1-c][i] == 0 and i != edges[1]:
            i+=1
        right = i-1
        return np.array(line[c][left:right+1])


    def cscore(self, line, c=0, edges=(None, None), cap=2):
        """
        count how many sub-lines of 5 come with the max number of stones
        Example: "oo.x*xx.." : The max num of blacks if obviously 3. And there are
                 two different adversary-free sub-lines counting three, namely '.x*xx' and 'x*xx.'.
                 Thus the cscore would be (3,2)

        Args:
            line: 8x2 integer array that represents the stones 
            c:  color: 0 to look at black, 1 to consider white
        """

        fr = self.f_range(line, c, edges)
        counts = []
        for i in range(len(fr)-3):
            counts.append(sum(fr[i:i+4]))            
        m = max(counts) if counts else 0
        c_ = sum(np.array(counts) == max(counts)) if counts else 0
        c_ = min(c_,cap)
        return (m, c_)
    

    def score(self, line, c=0, edges=(None, None)):
        """
        weighted sum of the count score
        """
        cscore = self.cscore(line, c, edges)
        mag, mul = cscore
        return mul**(1/self.kappa1) * mag
    
    
    def scores(self, lines, c=0, all_edges=None):
        if all_edges is None:
            all_edges = [(None, None), (None, None), (None, None), (None, None)]

            return [self.score(line, c, edges=edges) 
                for line, edges in zip(lines, all_edges)]
    
    
    def total_score(self, lines, c=0, all_edges=None):
        """
        total score of the given list of lines
        """
        if all_edges is None:
            all_edges = [(None, None), (None, None), (None, None), (None, None)]

        scores = [self.score(line, c, edges=edges) 
                  for line, edges in zip(lines, all_edges)]
        
        return self.euclidean_sum(scores)
    
    def euclidean_sum(self, scores):
        return sum(s**self.kappa0 for s in scores)**(1/self.kappa0) 
    
    
    def tactical_threshold(self):
        """
        The threshold that signals a MUST move
        """
        
        two_open3s = self.total_score([
            to_bits([ 0, 0, 0, 1, 1, 0, 0, 0]),
            to_bits([ 0, 0, 0, 1, 1, 0, 0, 0])])
        
        half_open4_open3 = self.total_score([
            to_bits([ 0, 2, 1, 1, 1, 0, 0, 0]),
            to_bits([ 0, 0, 0, 1, 1, 0, 0, 0])])
        
        # defended not-so-dangerous versions
        def_two_open3s = self.total_score([
            to_bits([ 0, 0, 2, 1, 1, 0, 0, 0]),
            to_bits([ 0, 0, 0, 1, 1, 0, 0, 0])])
        
        def_hopen4_open3 = self.total_score([
            to_bits([ 0, 2, 1, 1, 1, 0, 0, 0]),
            to_bits([ 0, 0, 0, 1, 1, 0, 0, 0])])
        
        threshold = (two_open3s + half_open4_open3 + def_hopen4_open3 + def_two_open3s) / 4
        
        # Threshold is in the middle of dangerous and not so dangerous
        return threshold
    
class HeuristicScore2:

    def __init__(self):
        self.c2t={
            (1,1): 1,
            (1,2): 2,
            (2,1): 3,
            (2,2): 4,
            (3,1): 5,
            (3,2): 6,
            (4,1): 8,
            (4,2): 9
        }
        
    def classify_line(self, line):
        cscore = HeuristicScore().cscore(line)
        if (cscore[0] == 0):
            return 0
        return self.c2t[cscore]
    
    def criticality(self, h, l):
        if h == 9: 
            return ('lost', 1)
        elif h == 8:
            return ('move or lose in 1', 2)
        elif h == 7: 
            return ('move or lose in 2', 3)
        elif (h, l) in [(5,5), (5,4), (6,5), (6,4), (6,6)]:
            return ('move or lose in 2', 4)
        elif (h, l) == (4,4):
            return ('move or lose in 3', 5)
        else:
            return ('defendable', 6)
    
    def classify_nh(self, nh):
        classes=[self.classify_line(nh.bits_in_line(h)) for h in ['e', 'ne', 'n', 'nw']]
        l, h = sorted(classes)[-2:]
        c = self.criticality(h, l)
        return h, l, c[1]
    
    