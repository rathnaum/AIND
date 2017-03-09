# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation is the technique to apply constraints to sudoku board till a solution is found or constraints can no longer be applied. Naked Twins is a strategy to reduce the number of possibilities to different parts of sudoku. The strategy is to identify a pair of boxes belonging to the same set of peers that have the same 2 numbers as possibilities, and eliminate these two numbers from all the peers that contains the two numbers. Naked Twins can be found along Row Units, Column Units, Square Units and along 2 diagonal units in case of diagonal sudoku.

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem? 
A: For diagonal sudoku, the constraint propagation and applying various techniques remains same as normal sudoku except 2 addional diagonal units are added to list of units. Reduce Puzzle, DFS Search, Naked twins, Only Choice and elimination constraints are applied to solve the puzzle.

### Code

* `solution.py` - Complete solution for Diagonal Sudoku and Naked Twins technique.
* `solution_test.py` - Unit Test file to test Diagonal Sudoku and Naked Twins technique.
* `utils.py` - Utils file.
