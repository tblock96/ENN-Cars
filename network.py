## Neural Network class

import numpy as np

def randomize(nodes):
    '''Returns a random series of matrices that will act as the weights of a nn
    nodes: A list of the sizes of each layer. The weights matrices will allow matrix-vector
    multiplication to go from size to size to size as listed in nodes.'''
    ret = []
    for i in range(len(nodes)-1):
        mat = np.random.rand(nodes[i], nodes[i+1])*2-1
        ret.append(mat)
    return ret

class Network():
    
    def __init__(self, nodes = [1]):
        self.weights = []       # Will hold ndarrays
        self.set_nodes(nodes)   # A list of nodes in each layer,
                                # starting with input layer
        
    def set_nodes(self, nodes):
        self.nodes = nodes
        self.weights = randomize(nodes)
    
    def set_weights(self, weights):
        self.weights = weights
    
    def run(self, input):
        '''Return an array of output size: self.nodes[-1]
        input: an array of size self.nodes[0]
        '''
        current_layer = input
        for i in range(len(self.nodes)-1):
            # multiplies the previous layer by the next weight matrix
            try:
                current_layer = np.dot(current_layer, self.weights[i])
            except Exception:
                print "Nodes: ", self.nodes
                print i
                print self.weights
                return
            current_layer = self.activation(current_layer)
        # Flatten the last layer to boolean
        for i in range(len(current_layer)):
            if current_layer[i] > 0:
                current_layer[i] = 1
            else:
                current_layer[i] = 0
        return current_layer
    
    def activation(self, input):
        '''Returns the input modified by the activation function (identity)'''
        return input

## Now to Evolve them

class Species():
    '''This is where the magic happens'''
    
    def __init__(self, input_size = 1, output_size = 1, depth_mut_rate = .05,
        width_mut_rate = .05, weight_mut_rate = .1):
        self.depth_mut_rate = depth_mut_rate
        self.width_mut_rate = width_mut_rate
        self.weight_mut_rate = weight_mut_rate
        self.input_size = input_size
        self.output_size = output_size
        self.parameters = []
        self.individuals = []
        self.generation = 0
        self.max_fitness = 0
        self.med_fitness = 0
    
    def set_input_size(self, s):
        self.input_size = s
    
    def set_output_size(self, s):
        self.output_size = s
    
    def add_network(self, n):
        self.individuals.append(n)
    
    def init_population(self):
        '''Sets self.parameters to be random for each individual in the species'''
        self.parameters = []
        for i in range(len(self.individuals)):
            self.parameters.append(self.random_parameters())
    
    def update_population(self):
        '''Sets each individual to follow the parameters given'''
        for i in range(len(self.individuals)):
            num_hidden, layer_size, weights = self.parameters[i]
            self.individuals[i].set_nodes([self.input_size] +
                [layer_size]*num_hidden + [self.output_size])
            self.individuals[i].set_weights(weights)
            self.individuals[i].fitness = 0
    
    def random_parameters(self):
        '''Parameters are number of hidden layers, size of hidden layers,
        and network weights.
        Returns a list in that order.'''
        num_hidden = np.random.randint(3)
        layer_size = np.random.randint(5)+1
        nodes = [self.input_size] + [layer_size]*num_hidden + [self.output_size]
        weights = randomize(nodes)
        return [num_hidden, layer_size, weights]
    
    def evolve(self):
        '''Evolves the parameters of the species'''
        self.generation += 1
        pop = len(self.individuals)
        fit_survivors = int(pop/4)
        # sort the individuals and parameters by their fitness
        sorted, sort_params = self.sort_by_fitness()
        # take the most fit individuals (and their parameters)
        survive = sorted[:fit_survivors]
        surv_params = sort_params[:fit_survivors]
        # some individuals pass on their genes because they're lucky
        luck_survivors = int(pop/8)
        i = -1
        # we pick which are the lucky ones
        while luck_survivors > 0:
            i = (i+1)%fit_survivors
            if sorted[fit_survivors+i] in survive: continue
            if np.random.random() < .25:
                survive.append(sorted[fit_survivors+i])
                surv_params.append(sort_params[fit_survivors+i])
                luck_survivors -= 1
        # Now the ones that survive get to breed.
        params = self.breed(survive, surv_params,
                            pop-fit_survivors-int(pop/8))
        # Now they mutate (maybe)
        params = self.mutate(params)
        # Update the parameters
        self.parameters = params
        print "After evolution"
        self.print_params()
        print "Max: %f\nMed: %f" %(self.max_fitness, self.med_fitness)
        # Update the individuals
        self.update_population()
    
    def sort_by_fitness(self):
        '''Self-explanatory'''
        sorted = self.individuals[:1]
        sort_params = self.parameters[:1]
        num_sorted = 1
        while num_sorted < len(self.individuals):
            inserted = False
            for i in range(num_sorted):
                if self.individuals[num_sorted].fitness > sorted[i].fitness:
                    sorted.insert(i, self.individuals[num_sorted])
                    sort_params.insert(i, self.parameters[num_sorted])
                    inserted = True
                    break
            if not inserted:
                sorted.append(self.individuals[num_sorted])
                sort_params.append(self.parameters[num_sorted])
            num_sorted += 1
        self.max_fitness = sorted[0].fitness
        self.med_fitness = sorted[len(self.individuals)/2].fitness
        return sorted, sort_params
    
    def breed(self, parents, parameters, num_children):
        '''Mixes up the values of the parameters of parets, returning num_children
        new parameters'''
        total_fitness = 0
        for i in parents:
            total_fitness += i.fitness 
        for i in range(num_children):
            p = []
            num_hidden = []
            layer_size = []
            weights = []
            j = 0
            # pick two parents (maybe the same parent -- asexual reproduction?)
            while len(p) < 2:
                if np.random.random() < (parents[j].fitness+1e-3)/total_fitness:
                    p.append(j)
                j = (j+1)%len(parents)
            # take the parents' parameters
            for index in p:
                num_hidden.append(parameters[index][0])
                layer_size.append(parameters[index][1])
                weights.append(parameters[index][2])
            
            same_nodes = []
            # get a random value between that of the parents
            c_hidden = np.random.randint(min(num_hidden), max(num_hidden)+1)
            # same
            c_layer = np.random.randint(min(layer_size),max(layer_size)+1)
            
            for j in range(2):
                # if the child has the same shape of nn as a parent, save that
                # parent's weights
                if c_hidden == num_hidden[j] and c_layer == layer_size[j]:
                    same_nodes.append(weights[j])
            nodes = [self.input_size] + [c_layer]*c_hidden + [self.output_size]
            # start with random weights
            c_weights = randomize(nodes)
            # if only one parent has the same nn shape, copy their weights
            if len(same_nodes) == 1:
                same_nodes = same_nodes*2
            # if two parents have the same nn shape, pick each weight value
            # so that it's between the parents' values
            if len(same_nodes) == 2:
                for i in range(len(c_weights)):
                    for j in range(np.shape(c_weights[i])[0]):
                        for k in range(np.shape(c_weights[i])[1]):
                            c_weights[i][j,k] = c_weights[i][j,k] * \
                                (same_nodes[0][i][j,k] - same_nodes[1][i][j,k]) + \
                                same_nodes[1][i][j,k]
            # add the child's parameters to the species
            parameters.append([c_hidden, c_layer, c_weights])
        return parameters
    
    def mutate(self, params):
        '''Randomly inserts changes into the parameters of the species'''
        for i in range(len(params)):
            # Doesn't mutate the 2 most fit individuals (this is cheating)
            if i < 2: continue
            change_nodes = False
            r = np.random.random()
            if r < self.depth_mut_rate:
                # Mutate the depth of the nn
                if r < self.depth_mut_rate/2.:
                    # Make it smaller
                    params[i][0] = max(0, params[i][0]-1)
                else:
                    # Make it larger
                    params[i][0] += 1
                # We'll need new weights
                change_nodes = True
            else:
                r -= self.depth_mut_rate
                if r < self.width_mut_rate:
                    # Mutate the number of nodes per layer
                    if r < self.width_mut_rate/2.:
                        # Make it smaller
                        params[i][1] = max(1, params[i][1]-1)
                    else:
                        # Make it larger
                        params[i][1] += 1
                    # We'll need new weights
                    change_nodes = True
                else:
                    r -= self.width_mut_rate
                    weight_muts = int(self.weight_mut_rate/r)
                    # We might mutate lots of weights
                    while weight_muts > 0:
                        # pick a matrix
                        r0 = np.random.randint(len(params[i][2]))
                        # pick a row
                        r1 = np.random.randint(np.shape(params[i][2][r0])[0])
                        # pick a column
                        r2 = np.random.randint(np.shape(params[i][2][r0])[1])
                        # random!
                        params[i][2][r0][r1,r2] = np.random.random()*2-1
                        weight_muts -= 1
            if change_nodes:
                # We need new matrices
                nodes = [self.input_size]+[params[i][1]]*params[i][0]+[self.output_size]
                params[i][2] = randomize(nodes)
        return params
    
    def print_params(self, params = None):
        '''Print the depth and width of the NNs in the species'''
        if params == None: params = self.parameters
        for i in range(len(params)):
            print i, params[i][0], params[i][1]