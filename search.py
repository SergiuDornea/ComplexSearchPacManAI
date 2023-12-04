# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # TODO !!! CHANGE THE QUEUE FROM THE BFS ALGO TO A STACK  - BFS IMPLEMENTED FIRST BECAUSE OF THE PSEUDOCODE IN BOOK :)
    # function BREADTH-FIRST-SEARCH(problem) returns a solution node or failure
    #   node ← NODE(problem.INITIAL) -
    #   TODO defined the inital node
    initialNode = problem.getStartState()
    #   if problem.IS-GOAL(node.STATE) then return node
    #  TODO check if goal has been reached
    if problem.isGoalState(initialNode):
        return []  # no actions needed if we start at the goal
    #   frontier ← a FIFO stack
    #   TODO create stack
    stack = util.Stack()  # create the stack using the provided utils
    #   with node as an element
    #   TODO add the node to the list  with an empty list for directions as an object
    stack.push((initialNode, []))
    #   reached ← {problem.INITIAL}
    #   TODO create a list to hold the values of reached nodes
    reachedNodes = []
    #   while not IS-EMPTY(frontier ) do
    while not stack.isEmpty():
        #       node ← POP(frontier )
        # TODO dequeue a nod (and list of directions) form the stack
        node, directionList = stack.pop()
        #       TODO check if node has been reached , if not add it to reachedNodes
        if node not in reachedNodes:
            reachedNodes.append(node)
            #           TODO check for exit condition - is goal state true?
            if problem.isGoalState(node):
                return directionList
            #
            #       for each child in EXPAND(problem, node) do
            #           s ← child.STATE
            #           if problem.IS-GOAL(s) then return child
            #           if s is not in reached then
            #               add s to reached
            #               add child to frontier
            #             TODO iterate over the succesors of current node
            succesorsList = problem.getSuccessors(node)
            for succesor, direction, stepCost in succesorsList:
                #               TODO append the curr direction to directionList and
                #                create newDirectionList
                #                (holds the seauence of directions from initial state to current succesor)
                newDirection = directionList + [direction]
                #                 TODO enqueue the succesor node with newDirectionList to stack
                stack.push((succesor, newDirection))
    #  return failure
    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # function BREADTH-FIRST-SEARCH(problem) returns a solution node or failure
    #   node ← NODE(problem.INITIAL) -
    #   TODO defined the inital node
    initialNode = problem.getStartState()
    #   if problem.IS-GOAL(node.STATE) then return node
    #  TODO check if goal has been reached
    if problem.isGoalState(initialNode):
        return []  # no actions needed if we start at the goal
    #   frontier ← a FIFO queue
    #   TODO create queue
    queue = util.Queue()  # create the queue using the provided utils
    #   with node as an element
    #   TODO add the node to the list  with an empty list for directions as an object
    queue.push((initialNode, []))
    #   reached ← {problem.INITIAL}
    #   TODO create a list to hold the values of reached nodes
    reachedNodes = []
    #   while not IS-EMPTY(frontier ) do
    while not queue.isEmpty():
        #       node ← POP(frontier )
        # TODO dequeue a nod (and list of directions) form the queue
        node, directionList = queue.pop()
        #       TODO check if node has been reached , if not add it to reachedNodes
        if node not in reachedNodes:
            reachedNodes.append(node)
            #           TODO check for exit condition - is goal state true?
            if problem.isGoalState(node):
                return directionList
            #
            #       for each child in EXPAND(problem, node) do
            #           s ← child.STATE
            #           if problem.IS-GOAL(s) then return child
            #           if s is not in reached then
            #               add s to reached
            #               add child to frontier
            #             TODO iterate over the succesors of current node
            succesorsList = problem.getSuccessors(node)
            for succesor, direction, stepCost in succesorsList:
                #               TODO append the curr direction to directionList and
                #                create newDirectionList
                #                (holds the seauence of directions from initial state to current succesor)
                newDirection = directionList + [direction]
                #                 TODO enqueue the succesor node with newDirectionList to queue
                queue.push((succesor, newDirection))
    #  return failure
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Strategia: expandeaza nodul cu
    # costul cel mai mic
    # Implementare: coada de
    # prioritati (cost cumulativ) are
    # rol de granita
    #     TODO set the start
    start = problem.getStartState()
    #     TODO set the cost
    totalCost = 0
    #     TODO set the goal
    goal = None
    #     TODO set the priority queue and add start to it
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((start, [], 0), 0)
    #     TODO set the reached node list
    reachedNodes = []

    #     TODO while loop
    #     while q not equal to 0
    while not priorityQueue.isEmpty():
        #     TODO get the node, cost and directions form list - u = exrtract MIN(q)
        node, directions, currCost = priorityQueue.pop()
        #
        #     TODO check if exit condition is valid - goal found
        if problem.isGoalState(node):
            return directions

        #       for fiecare nod v apartine G.Adj[u]
        if node not in reachedNodes:
            successors = problem.getSuccessors(node)
            for successor, action, stepCost in successors:
                #       RELAX(U,V, W)
                if successor[0] not in reachedNodes:
                    newCost = totalCost + stepCost  # increase the cost with current step cost
                    newDirection = directions + [action]  # add the direction to directions
                    #             add to the pryority queue item with new priority (newCost)
                    priorityQueue.push((successor, newDirection, newCost), newCost)
            reachedNodes.append(node)

    #  return failure
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """function GRAPH-SEARCH(problema, frontiera) :returneaza o solutie sau esec
        closed ← o multime goala
        frontiera ← INSERT(MAKE-NODE(INITIAL-STATE(problema)), frontiera)
        loop
            if frontiera este goala then
                return esec
            end if
            node ← REMOVE-FRONT(frontiera)
            if GOAL-TEST(problema, STATE(node)) then
                return node
            end if
            if STATE(node) nu este in closed then
                for child-node in EXPAND(STATE(node), problem) do
                frontiera ← INSERT(child-node, frontiera)
                end for
            end if
            add STATE(node) la closed
        end loop
    end function"""
    # TODO start node
    start = problem.getStartState()
    # TODO  set reached node list
    reachedNodes = []
    # TODO set queue
    priorityQueue = util.PriorityQueue()
    # todo add start node to queue
    priorityQueue.push((start, [], 0), 0)
    # TODO check if start node is not the goal
    if problem.isGoalState(start):
        return []

    while not priorityQueue.isEmpty():
        #     TODO get the node, cost and directions form queue
        node, directions, currCost = priorityQueue.pop()
        if node not in reachedNodes:
            #     TODO check if exit condition is valid - goal found
            if problem.isGoalState(node):
                return directions
            successors = problem.getSuccessors(node)
            for successor, action, stepCost in successors:
                newCost = currCost + stepCost  # increase the cost with current step cost
                heuristicCost = heuristic(successor,
                                          problem) + newCost  # calculate the f(n) by adding new cost to the result of heuristic
                newDirection = directions + [action]  # add the direction to directions
                #             add to the pryority queue item with new priority (newCost)
                priorityQueue.push((successor, newDirection, newCost),
                                   heuristicCost)  # instead of setting the priority as the cost (like in UCS) set the priority according to the heuristic cost
        #     todo add the node to the reached list
        reachedNodes.append(node)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch