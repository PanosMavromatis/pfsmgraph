import math
import numpy as np
from typing import List, Callable, Tuple, Any, Dict

from ..util import unfold_tree

def ss2_alignment(seq1: List[Any], seq2: List[Any],
                  subst_cost: Callable = None,
                  gap_opening_cost: float = 1.0,
                  gap_extension_cost: float = 1.0) -> Dict:
    """
    Implements the Altschul & Erickson (1986) algorithm for finding ALL optimal
    sequence alignments with affine gap penalties.
    
    This implements the SS-2 algorithm from "Optimal Sequence Alignment Using 
    Affine Gap Costs" that correctly finds all optimal alignments, addressing 
    limitations in Gotoh's original approach.
    
    Parameters:
    -----------
    seq1, seq2 : List[Any]
        The two sequences to align (lists of musical tokens)
    subst_cost : Callable, optional
        Custom function to determine similarity between elements
    gap_opening_cost : float
        Penalty for opening a gap (v in the paper; for a gap of length k,
           total gap cost = v + k u)
    gap_extension_cost : float
        Penalty for extending a gap (u in the paper; see above)
    
    Returns:
    --------
    result : Dict
        Dictionary containing:
        - 'score': The optimal alignment score
        - 'solution_tree': A tree representation of all optimal alignments
    """
    # If no similarity function is provided, use the default scoring
    if subst_cost is None:
        def subst_cost(a, b):
            # Default cost for mismatch is 1
            return 0.0 if a == b else 1.0
    
    # Affine gap costs: w_k = v + ku as defined in the paper
    v = gap_opening_cost  # Gap opening cost
    u = gap_extension_cost  # Gap extension cost
    
    m, n = len(seq1), len(seq2)
    
    # Initialize the three rectangular number arrays as in the paper:
    # P[i,j]: Cost of path ending at node (i,j) using a vertical edge
    # Q[i,j]: Cost of path ending at node (i,j) using a horizontal edge
    # R[i,j]: Minimum cost of path ending at node (i,j)
    P = np.full((m+1, n+1), float('nan'))
    Q = np.full((m+1, n+1), float('nan'))
    R = np.full((m+1, n+1), float('nan'))
    
    # Initialize the seven rectangular bit arrays as described in the paper:
    # a[i,j]: 1 iff an optimal path to (i,j) uses vertical edge
    # b[i,j]: 1 iff an optimal path to (i,j) uses horizontal edge
    # c[i,j]: 1 iff an optimal path to (i,j) uses diagonal edge
    # d[i,j]: 1 iff among paths through (i-1,j), an optimal one uses vertical edge (i-1,j-1)->(i-1,j)
    # e[i,j]: 1 iff among paths through (i-1,j), an optimal one does not use vertical edge (i-1,j-1)->(i-1,j)
    # f[i,j]: 1 iff among paths through (i,j-1), an optimal one uses horizontal edge (i-1,j-1)->(i,j-1)
    # g[i,j]: 1 iff among paths through (i,j-1), an optimal one does not use horizontal edge (i-1,j-1)->(i,j-1)
    a = np.zeros((m+2, n+2), dtype=bool)
    b = np.zeros((m+2, n+2), dtype=bool)
    c = np.zeros((m+2, n+2), dtype=bool)
    d = np.zeros((m+2, n+2), dtype=bool)
    e = np.zeros((m+2, n+2), dtype=bool)
    f = np.zeros((m+2, n+2), dtype=bool)
    g = np.zeros((m+2, n+2), dtype=bool)

    # ===== INITIALIZATION =====
    # "For j from 0 to N, set P0,j to +∞ and R0,j to v+ju."
    for j in range(n+1):
        P[0, j] = float('inf')
        R[0, j] = v + j*u
    
    # "For i from 0 to M, set Qi,0 to +∞ and Ri,0 to v+iu."
    for i in range(m+1):
        Q[i, 0] = float('inf')
        R[i, 0] = v + i*u
    
    # "Set R0,0 to 0."
    R[0, 0] = 0
    
    # "Set c[m+1,n+1] to 1"
    c[m+1, n+1] = True
    
    # ===== COST ASSIGNMENT =====
    # "For i from 0 to M and j from 0 to N, execute steps {2}-{7}."
    for i in range(m+1):
        for j in range(n+1):
            # Step {2}: "Find the minimum cost of a path ending at node Ni,j and using edge Vi,j"
            if i > 0:
                P[i, j] = u + min(P[i-1, j], R[i-1, j] + v)
            
            # Step {3}: "Determine if cost Pi,j can be achieved using edge Vi-1,j and if it can be
            #            achieved without using edge Vi-1,j"
            if i > 0:
                if P[i, j] == P[i-1, j] + u:
                    d[i-1, j] = True
                if P[i, j] == R[i-1, j] + v + u:
                    e[i-1, j] = True
            
            # Step {4}: "Find the minimum cost of a path ending at node Ni,j and using edge Hi,j"
            if j > 0:
                Q[i, j] = u + min(Q[i, j-1], R[i, j-1] + v)
            
            # Step {5}: "Determine if cost Qi,j can be achieved using edge Hi,j-1 and if it can be
            #            achieved without using edge Hi,j-1"
            if j > 0:
                if Q[i, j] == Q[i, j-1] + u:
                    f[i, j-1] = True
                if Q[i, j] == R[i, j-1] + v + u:
                    g[i, j-1] = True
            
            # Step {6}: "Find the minimum cost of a path ending at node Ni,j"
            if i > 0 and j > 0:
                R[i, j] = min(P[i, j], Q[i, j], R[i-1, j-1] + subst_cost(seq1[i-1], seq2[j-1]))
            
            # Step {7}: "Determine if cost Ri,j can be achieved by using edge Vi,j, Hi,j or Di,j"
            if R[i, j] == P[i, j]:
                a[i, j] = True
            
            if R[i, j] == Q[i, j]:
                b[i, j] = True
            
            if i > 0 and j > 0 and R[i, j] == R[i-1, j-1] + subst_cost(seq1[i-1], seq2[j-1]):
                c[i, j] = True
    
    # # Debugging - print the arrays at any given point in the algorithm
    # print("Edit cost arrays:")
    # for i in range(m+1):
    #     print("-" * 100)
    #     for j in range(n+1):
    #         print(i, j, P[i, j], Q[i, j], R[i, j])
    # print("=" * 100)
    # print("Boolean arrays:")
    # for i in range(m+2):
    #     print("-" * 100)
    #     for j in range(n+2):
    #         print(i, j, a[i, j], b[i, j], c[i, j], d[i, j], e[i, j], f[i, j], g[i, j])
    # return ()
    
    # ===== EDGE ASSIGNMENT =====
    # "For i from M to 0 and j from N to 0, execute steps {8}-{11}."
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            # Step {8}: "If there is no optimal path passing through node Ni,j which has cost
            #            Ri,j at node Ni,j, remove edges Vi,j, Hi,j and Di,j"
            if (not (a[i+1, j] and e[i, j]) and 
                not (b[i, j+1] and g[i, j]) and 
                not c[i+1, j+1]):
                a[i, j] = b[i, j] = c[i, j] = False
            
            # Step {9}: "If no optimal path passes through node Ni,j, proceed to the next node"
            if (a[i+1, j] or b[i, j+1] or c[i+1, j+1]):
                
                # Step {10}: "If edge Vi+1,j is in an optimal path and requires edge Vi,j
                #             to be in an optimal path, determine if an optimal path that
                #             uses edge Vi+1,j must use edge Vi,j and the converse"
                if a[i+1, j] and d[i, j]:
                    d[i+1, j] = not e[i, j]
                    e[i, j] = not a[i, j]
                    a[i, j] = True
                else:
                    d[i+1, j] = e[i, j] = False
                
                # Step {11}: "If edge Hi,j+1 is in an optimal path and requires edge Hi,j
                #             to be in an optimal path, determine if an optimal path that
                #             uses edge Hi,j+1 must use edge Hi,j and the converse"
                if b[i, j+1] and f[i, j]:
                    f[i, j+1] = not g[i, j]
                    g[i, j] = not b[i, j]
                    b[i, j] = True
                else:
                    f[i, j+1] = g[i, j] = False
    
    # ===== PREPARE THE SOLUTION TREE =====
    # Create a representation of the solution tree that follows all
    # allowed paths of the affine graph
    def solution_tree( from_node ):
        from_move, i, j = from_node
        if (i,j) == (m,n):
            return [ from_node ]
        else:
            next_nodes = []
            # Check for vertical constraint
            if from_move == 'a' and e[i,j]:
                # We infer that a[i+1,j] is true
                next_nodes.append( ('a', i+1, j) )
            # Check for horizontal constraint
            elif from_move == 'b' and g[i,j]:
                # We infer that b[i,j+1] is true
                next_nodes.append( ('b', i, j+1) )
            # Unconstrained case; we need to check all possible moves
            else:
                # if from_move == 'b' and g[i,j]:        # This must be an error
                #     next_nodes.append( ('g', i, j) )
                if c[i+1,j+1]:
                    next_nodes.append( ('c', i+1, j+1) )
                if a[i+1,j]:
                    next_nodes.append( ('a', i+1, j) )
                if b[i,j+1]:
                    next_nodes.append( ('b', i, j+1) )
            branches = [ solution_tree(node) for node in next_nodes ]
            return [ from_node ] + branches
    
    def align_seqs(moves):
        seq1_align, seq2_align = [], []
        for move in moves[1:]: # Skip the starting node
            match move:
                case ('c', i, j):
                    seq1_align.append(seq1[i-1])
                    seq2_align.append(seq2[j-1])
                case ('b', i, j):
                    seq1_align.append(None)
                    seq2_align.append(seq2[j-1])
                case ('a', i, j):
                    seq1_align.append(seq1[i-1])
                    seq2_align.append(None)
        return (seq1_align, seq2_align)
    
    solution_tree = solution_tree( ('s', 0, 0) )
    solutions = unfold_tree(solution_tree)
    alignments = [ align_seqs(solution) for solution in solutions ]

    return {
        # The highest score:
        'score': R[m, n],
        # The optimal paths that attain this score:
        'solution_tree': solution_tree,
        # The optimal alignments corresponding to these paths:
        'alignments': alignments
    }
