import numpy as np

class shortest_path_Agent:
    '''
    Select the shortest available path among the k paths
    The "None" in the return is to make the class behave in the same way as Agent.py
    '''
    
    def __init__(self, environment):
        self.k = 4
        self.environment = environment

    def act(self, state, demand, source, destination, _ignore_):
        first_k_shortest_paths = self.environment.first_k_shortest_paths[str(source) + ':' + str(destination)]
        allocated = False # Indicates 1 if we allocated the demand, 0 otherwise
        new_state = np.copy(state)
        
        # iterate over the k paths, as soon as we find an available path we pick it
        path = 0
        while allocated == False and path < len(first_k_shortest_paths) and path < self.k:
            current_path = first_k_shortest_paths[path]
            can_allocate = True 
            
            # check if the current path can be allocated
            for edge in current_path:
                if new_state[edge][0] - demand < 0:
                    can_allocate = False
                    break
            

            if can_allocate == True:
                return path, None
            
            path += 1

        # If we get here it means none of the k paths can be allocated, so we allocate the first one
        return 0, None