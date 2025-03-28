import random
import copy

import networkx as nx
import matplotlib.pyplot as plt

UNVISITED = -1

class Automaton:
    def __init__(self, num_states=16):
        # Initialize states
        self.states = set(n for n in range(num_states))
        for n in range(random.randrange(0, 48)):  # Add extra states randomly
            self.states.add(n + num_states)

        # Define alphabet (transition symbols)
        self.alphabet = {'a', 'b'}

        # Define transitions using adjacency list
        self.transitions = {state: {symbol: random.choice(list(self.states)) for symbol in self.alphabet} for state in self.states}

        # Define final states
        self.final_states = set(random.sample(list(self.states), random.randrange(0, len(self.states) + 1)))

        # Choose a start state randomly
        self.start_state = random.choice(list(self.states))
    
    def get_children(self, state):        
        """ Returns all states reachable from `state` through any transition. """
        return {next_state for next_state in self.transitions[state].values()}
    
    def get_child(self, state, label):
        return self.transitions.get(state).get(label)        
            
    def remove_unreachable(self, distances):
        unreachable_states = self.states - set(distances.keys())
        
        for state in unreachable_states:
            self.transitions.pop(state)
            
        for state, transitions in self.transitions.items():
            self.transitions[state] = {
                symbol: next_state for symbol, next_state in transitions.items()
                if next_state not in unreachable_states
            }
    
        self.states.difference_update(unreachable_states)
        self.final_states.difference_update(unreachable_states)
    
    def __reverse_transitions(self):        
        reverse_transitions = {letter: {} for letter in self.alphabet}

        for state, transitions in self.transitions.items():
            for letter, next_state in transitions.items():
                if next_state not in reverse_transitions[letter]:
                    reverse_transitions[letter][next_state] = set()
                reverse_transitions[letter][next_state].add(state)

        return reverse_transitions        
            
    def __find_partition(self, state, waiting):
        for partition in waiting:
            if state in partition:
                return frozenset(partition)
        return None
    
    def __update_automaton(self, partitions):
        # Update with new Partitions
        state_map = {state: self.__find_partition(state, partitions) for state in self.states}
        state_names = {frozenset(partition): i for i, partition in enumerate(partitions)}

        new_transitions = {}
        for state, transition in self.transitions.items():
            mapped_state = state_map[state]
            if state_names[mapped_state] not in new_transitions:
                new_transitions[state_names[mapped_state]] = {}
            
            for symbol, next_state in transition.items():
                mapped_next_state = state_map[next_state]
                new_transitions[state_names[mapped_state]][symbol] = state_names[mapped_next_state]

        self.transitions = new_transitions
        self.states = {state_names[partition] for partition in state_map.values()}
        self.final_states = {state_names[state_map[s]] for s in self.final_states}
        self.start_state = state_names[state_map[self.start_state]]
        
        return
      
    def hopcroft_optimization(self):
        partitions = [self.states - self.final_states, self.final_states]
        waiting = copy.deepcopy(partitions) # O(n). 
        
        reverse_transitions = self.__reverse_transitions()
        
        while waiting: # O(nlogn).
            A = waiting.pop()
            A_parents = set()
            
            for letter in self.alphabet:
                # Define the split. 
                for state in A:
                    if state in reverse_transitions[letter]:
                        A_parents.update(reverse_transitions[letter][state])
                
                for partition in partitions:
                    # Compute the split.
                    intersect = partition.intersection(A_parents)   # P'  - Leads to A
                    difference = partition - A_parents              # P'' - ~Leads to A

                    # If a split has occurec
                    if intersect and difference:
                        # Update partition P with P' and P''
                        partitions.remove(partition)
                        partitions.extend([intersect, difference])
                    
                        # Update waiting set
                        if partition in waiting:
                            waiting.remove(partition)
                            waiting.extend([intersect, difference])
                        else:
                            waiting.append(intersect if len(intersect) <= len(difference) else difference)
                            
        self.__update_automaton(partitions)
        return  
      
    def print_automaton(self):
        """ Pretty-prints the automaton. """
        print("\n--- Automaton Information ---")
        print(f"Number of states: {len(self.states)}")
        print(f"States: {self.states}")
        print(f"Alphabet: {self.alphabet}")
        print(f"Start state: {self.start_state}")
        print(f"Final states: {self.final_states}")
        print("\nTransitions:")
        
        for state, transition in self.transitions.items():
            print(f"{state}, {transition}")
        print("\n-----------------------------")

    def copy(self):
        new_automaton = Automaton()
        new_automaton.states = copy.deepcopy(self.states)
        new_automaton.alphabet = copy.deepcopy(self.alphabet)
        new_automaton.transitions = copy.deepcopy(self.transitions)
        new_automaton.final_states = copy.deepcopy(self.final_states)
        new_automaton.start_state = copy.deepcopy(self.start_state)
        return new_automaton
    
    def visualize(self):
        G = nx.DiGraph()

        # Add states as nodes
        for state in self.states:
            G.add_node(state, color='green' if state in self.final_states else 'blue')

        # Add transitions as directed edges
        for state, transitions in self.transitions.items():
            for symbol, next_state in transitions.items():
                G.add_edge(state, next_state, label=symbol)

        pos = nx.spring_layout(G)

        # Get node colors
        node_colors = [G.nodes[node]['color'] for node in G.nodes]

        # Draw nodes and edges
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=12, font_color='white')

        # Draw edge labels
        edge_labels = {(s, ns): a for s, transitions in self.transitions.items() for a, ns in transitions.items()}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Automaton Visualization")
        plt.show()
    
def bfs(automaton):
    visited = set()
    edge = [automaton.start_state]
    distances = {automaton.start_state: 0}
    
    while edge:
        current = edge.pop(0)
        visited.add(current)
                
        for neighbour in automaton.get_children(current):
            if neighbour in visited or neighbour in edge:
                continue
            
            distances[neighbour] = distances[current] + 1
            edge.append(neighbour)
            
    return distances

UNVISITED = -1
id = 0
sccCount = 0
sccHigh = 0
sccLow = 0
sccSize = 0

stack = []
ids = []
low = []
visited = []
sccs = []

def dfs(automaton, at):
    global id, sccCount, sccHigh, sccLow, sccSize
    ids[at] = low[at] = id
    id += 1
    stack.append(at)
    visited[at] = True
    
    for to in automaton.get_children(at):        
        if ids[to] == UNVISITED:
            dfs(automaton, to)
        if visited[to] and to in stack:
            low[at] = min(low[at], low[to])
    
    if ids[at] == low[at]:
        while True:
            node = stack.pop()
            # visited[node] = False
            sccs[node] = sccCount
            sccSize += 1
            if node == at:
                if sccSize > sccHigh:
                    sccHigh = sccSize
                if sccSize < sccLow or sccLow == 0:
                    sccLow = sccSize
                sccSize = 0
                break
        
        sccCount += 1
    
def find_SCCs(automaton):
    global ids, low, visited, sccs, sccCount, sccHigh, sccLow, id
    
    n = len(automaton.states)
    ids = [UNVISITED] * n
    low = [0] * n
    visited = [False] * n
    sccs = [-1] * n
    sccCount = 0
    sccHigh = 0
    sccLow = 0
    id = 0

    for state in automaton.states:
        if ids[state] == UNVISITED:
            dfs(automaton, state)
    
    return sccCount, sccHigh, sccLow

if __name__ == "__main__":
    # Q1
    A = Automaton()
       
    # Q2
    A_distances = bfs(A)
    print("\n--- Automaton Information ---\n")
    print(f"Number of states of A: {len(A.states)}")
    print(f"Depth of A: {max(A_distances.values())}")
    
    # Q3 
    M = A.copy()
    M2 = A.copy()

    M.remove_unreachable(A_distances)
    M.print_automaton()
    M2.remove_unreachable(A_distances)
    
    M.hopcroft_optimization()
    M2.hopcroft_optimizationChildren()
   
    # Q4
    M_distances = bfs(M)
    print(f"\nNumber of states of M: {len(M.states)}")
    print(f"Depth of M: {max(M_distances.values())}")
    print("\n-----------------------------")
    
    M2_distances = bfs(M2)
    print(f"\nNumber of states of M2: {len(M2.states)}")
    print(f"Depth of M2: {max(M2_distances.values())}")
    print("\n-----------------------------")
    
    # Q5
    A_SCCs = find_SCCs(A)
    M_SCCs = find_SCCs(M)
    
    print("\n------ SCC Information ------\n")
    print(f"Number of strongly connected components in A: {A_SCCs[0]}")
    print(f"Size of the largest SCC in A: {A_SCCs[1]}")
    print(f"Size of the smallest SCC in A: {A_SCCs[2]}")
    
    print(f"\nNumber of strongly connected components in M: {M_SCCs[0]}")
    print(f"Size of the largest SCC in M: {M_SCCs[1]}")
    print(f"Size of the smallest SCC in M: {M_SCCs[2]}")
    print("\n-----------------------------")
    
    
    # ------------- Testing -------------- 
    # A.print_automaton()
    
    # hard code automaton
    # A.states = set(n for n in range(8))
    # A.transitions = {
    #     0: {'a': 1, 'b': 2},
    #     1: {'a': 0, 'b': 3},
    #     2: {'a': 4, 'b': 5},
    #     3: {'a': 2},
    #     4: {'a': 6, 'b': 3},
    #     5: {'a': 4, 'b': 7},
    #     6: {'a': 7},
    #     7: {'a': 4}
    #     }
    # A.final_states = {4, 5, 7}
    # A.start_state = 0
    # A.print_automaton()
    
    # A.states = set(n for n in range(5))
    # A.transitions = {
    #     0: {'a': 1, 'b': 3},
    #     1: {'a': 2},
    #     2: {'a': 0},
    #     3: {'a': 4},
    #     4: {'a': 3}
    #     }
    # A.final_states = {2}
    # A.start_state = 0
    # A.print_automaton()
    
    # A.states = {2,0,1}
    # A.transitions = {
    #     2: {'a': 2, 'b': 0},
    #     0: {'a': 1},
    #     1: {'a': 0},
    #     # 3: {'a': 3},
    #     # 4: {'a': 4}
    #     }
    # A.final_states = {2}
    # A.start_state = 0
    # A.print_automaton()
    
    # A.visualize()