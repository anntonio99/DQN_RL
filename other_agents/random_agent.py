import numpy as np

class random_Agent:
    # Selects the path among the k paths with uniform probability
    def __init__(self, environment):
        self.k = 4
        self.environment = environment

    def act(self, state, demand, source, destination, _ignore_):

        first_k_shortest_paths = self.environment.first_k_shortest_paths[str(source) + ':' + str(destination)]
        new_state = np.copy(state)

        allocable_paths = 0
        id_last_free = -1 # Indicates the id of the last free path where we will allocate the demand

        path = 0
        # Check if there are at least 2 paths
        while allocable_paths < 2 and path < len(first_k_shortest_paths) and path < self.k:
            current_path = first_k_shortest_paths[path]
            can_allocate = True  

            for edge in current_path:
                if new_state[edge][0] - demand < 0:
                    can_allocate = False
                    break

            if can_allocate == True:
                allocable_paths += 1
                id_last_free = path

            path += 1

        # Case A: no allocable paths -> we return the first one
        if allocable_paths == 0:
            return 0, None

        # Case B: just 1 allocable path -> we (are forced to) pick it
        elif allocable_paths == 1:
            return id_last_free, None
        
        # Case C: at least 2 allocable paths -> we can actually choose randomly
        else:
            while True:
                # choose at random
                action = np.random.randint(0, len(first_k_shortest_paths))  # lui qui mette len() - 1, non capisco perchè
 
                current_path = first_k_shortest_paths[action]
                can_allocate = True 

                for edge in current_path:
                    if new_state[edge][0] - demand < 0:
                        can_allocate = False
                        break

                if can_allocate == True:
                    return action, None