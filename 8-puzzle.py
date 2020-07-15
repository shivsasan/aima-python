from utils import *
import random
import time

global remFront

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    remFront = 0
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        remFront += 1
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            print('Removed nodes from frontier: ',remFront)
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    print('Removed nodes from frontier: ',remFront)
    return None

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

def astar_search_man(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.manH, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

def astar_search_max(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.maxH, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))

    def manH(self, node):
        man = [
        [4,3,2,3,2,1,2,1,0], #0
        [0,1,2,1,2,3,2,3,4],
        [1,0,1,2,1,2,3,2,3],
        [2,1,0,3,2,1,4,3,2],
        [1,2,3,0,1,2,1,2,3],
        [2,1,2,1,0,1,2,1,2],
        [3,2,1,2,1,0,3,2,1],
        [2,3,4,1,2,3,0,1,2],
        [3,2,3,2,1,2,1,0,1], #8
        ]

        state = node.state
        i = 0
        maa = 0
        for i in range(1,9):
            maa += man[i][state.index(i)]
        return maa

    def maxH (self,node):
        return max(self.h(node), self.manH(node))

class DuckPuzzle(Problem):
    # Duck puzzle

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        return state.index(0)

    def actions(self, state):
        possible_duck_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square == 0:
            possible_duck_actions.remove('UP')
            possible_duck_actions.remove('LEFT')
        if index_blank_square == 1:
            possible_duck_actions.remove('UP')
            possible_duck_actions.remove('RIGHT')
        if index_blank_square == 2:
            possible_duck_actions.remove('DOWN')
            possible_duck_actions.remove('LEFT')
        if index_blank_square == 3:
            return possible_duck_actions
        if index_blank_square == 4:
            possible_duck_actions.remove('UP')
        if index_blank_square == 5:
            possible_duck_actions.remove('UP')
            possible_duck_actions.remove('RIGHT')
        if index_blank_square == 6:
            possible_duck_actions.remove('DOWN')
            possible_duck_actions.remove('LEFT')
        if index_blank_square == 7:
            possible_duck_actions.remove('DOWN')
        if index_blank_square == 8:
            possible_duck_actions.remove('DOWN')
            possible_duck_actions.remove('RIGHT')
        return possible_duck_actions

    def result(self, state, action):
        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        if blank < 3:
            delta = {'UP': -2, 'DOWN': 2, 'LEFT': -1, 'RIGHT': 1}
        if blank > 3:
            delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        if blank == 3:
            delta = {'UP': -2, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}

        #delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        return state == self.goal

    def check_solvability(self, state):
        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        #number of misplaced tiles
        return sum(s != g for (s, g) in zip(node.state, self.goal))

    def manH(self, node):
        man = [
        [5,4,4,3,2,1,2,1,0], #0
        [0,1,1,2,3,4,3,4,5],
        [1,0,2,1,2,3,2,3,4],
        [1,2,0,1,2,3,2,3,4],
        [2,1,1,0,1,2,1,2,3],
        [3,2,2,1,0,1,2,1,2],
        [4,3,3,2,1,0,3,2,1],
        [3,2,2,1,2,3,0,1,2],
        [4,3,3,2,1,2,1,0,1], #8
        ]

        state = node.state
        i = 0
        maa = 0
        for i in range(1,9):
            maa += man[i][state.index(i)]
        return maa

    def maxH (self,node):
        return max(self.h(node), self.manH(node))

#=====================================================================================

def make_rand_8puzzle():
    lst = random.sample(range(0,9),9)
    tup = tuple(lst)

    thePuzzle = EightPuzzle(tup)

    while(not thePuzzle.check_solvability(thePuzzle.initial)):
        tup = random.sample(range(0,9),9)
        tup = tuple(tup)
        thePuzzle = EightPuzzle(tup)

    return thePuzzle

def display(state):
    i = 0
    while i < 9:
        tempState = (state[i], state[i+1], state[i+2])

        if 0 in tempState:
            tempStateList = list(tempState)
            tempStateList[tempStateList.index(0)] = '*'
            tempState = tuple(tempStateList)
        #print(tempState[0]," ",tempState[1]," ", tempState[2])
        print(tempState[0],tempState[1], tempState[2])
        i += 3


def displayDuck(state):

    zeroIndex = state.index(0)

    if zeroIndex <=1:
        if zeroIndex == 0:
            print('*',state[1])
        else:
            print(state[0],'*')
        
        print(state[2],state[3],state[4],state[5])
        print(' ',state[6],state[7],state[8])
        return

    if zeroIndex <= 5 and zeroIndex > 1:
        print(state[0],state[1])
        if zeroIndex == 2:
            print('*',state[3],state[4],state[5])
        if zeroIndex == 3:
            print(state[2],'*',state[4],state[5])
        if zeroIndex == 4:
            print(state[2],state[3],'*',state[5])
        if zeroIndex == 5:
            print(state[2],state[3],state[4],'*')
        print(' ',state[6],state[7],state[8])
        return

    if zeroIndex <= 8 and zeroIndex > 5:
        print(state[0],state[1])
        print(state[2],state[3],state[4],state[5])
        if zeroIndex == 6:
            print(' ','*',state[7],state[8])
        if zeroIndex == 7:
            print(' ',state[6],'*',state[8])
        if zeroIndex == 8:
            print(' ',state[6],state[7],'*')
        return

def main():

    #creating 10 randomly generated 8puzzles
    #puz0 = make_rand_8puzzle()

    #test case
    #testPuz = EightPuzzle((4, 7, 1, 8, 6, 0, 5, 3, 2))
    #testPuz = DuckPuzzle((2, 3, 1, 5, 8, 7, 4, 6, 0))
    #testPuz = DuckPuzzle((3, 1, 0, 2, 7, 5, 6, 8, 4))
    testPuz = DuckPuzzle((2, 3, 1, 4, 6, 5, 0, 8, 7))
    displayDuck(testPuz.initial)

    #display(testPuz.initial)

    print("====misplaced tile====")
    testTime1 = time.time()
    testPuzSol = astar_search(testPuz).solution()
    testTime2 = time.time() - testTime1

    print(testPuzSol)
    print('Elapsed time for solution: ', testTime2)
    print('The number of tiles moved of the solution are: ',len(testPuzSol))

    #manHD = manH(testPuz)

    print("\n====MANHATTAN====")

    manTestTime1 = time.time()
    manTestPuzSol = astar_search_man(testPuz).solution()
    manTestTime2 = time.time() - manTestTime1
    print(manTestPuzSol)
    print('Elapsed time for solution: ', manTestTime2)
    print('The number of tiles moved of the solution are: ',len(manTestPuzSol))

    print("\n=====  MAX  =====")

    puz0Node = Node(testPuz.initial)

    print('.h : ',testPuz.h(puz0Node))
    print('.manH : ',testPuz.manH(puz0Node))

    maxTestTime1 = time.time()
    maxTestPuzSol = astar_search_max(testPuz).solution()
    maxTestTime2 = time.time() - maxTestTime1
    print(maxTestPuzSol)
    print('Elapsed time for solution: ', maxTestTime2)
    print('The number of tiles moved of the solution are: ',len(maxTestPuzSol))

    for i in manTestPuzSol:
        testPuz.initial = testPuz.result(testPuz.initial, i)
        displayDuck(testPuz.initial)

    print('======SOLVED=====')
    displayDuck(testPuz.initial)
    print(testPuz.goal_test(testPuz.initial))

if __name__ == "__main__":
    main()
