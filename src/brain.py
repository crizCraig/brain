class Brain(object):
  def __init__(self, num_layers, neurons_in_leaf_layer):
    """
    Build an empty brain

    Arguments:
    num_layers -- Synonymous with layers of cortex. Hierarchical layers of brain. Should be less than six.
    neurons_in_leaf_layer -- Number of neurons in bottom layer of hierarchy.
    neurons_per_region -- Number of neurons that can form connections with each other within a layer.
        Since regions don't connect with other regions on the same layer, we can parallelize region calculations.

    """
    self.num_layers = num_layers
    self.neurons_in_leaf_layer = neurons_in_leaf_layer

    self.leaf_layer_width = self.leaf_layer_height = self.neurons_in_leaf_layer ** 0.5 # Layers are perfect squares
    self.layers = []
    from layer import Layer
    child_layer = None
    num_neurons = self.neurons_in_leaf_layer
    for i in xrange(num_layers):
      is_top = True if i == num_layers - 1 else False
      new_layer = Layer(num_neurons=num_neurons, layer_num=i, child=child_layer, brain=self, is_top=is_top)
      if child_layer:
        child_layer.parent = new_layer
      self.layers.append(new_layer)
      child_layer = new_layer
      num_neurons /= 4 # Halving each side reduces total by four.
    for layer in self.layers:
      for neuron in layer.neurons.flat: # TODO, use nditer for speed (order doesn't matter).
        neuron.initConnections()

  def perceive(self, arr, learn):
    """Take a 2D array and feed it to the leaf layer. Then iterate it up the tree."""
    #  TODO: Try to feed input to more than just leaf layer to simulate visual cortex.
    for layer in self.layers:
      if layer.layer_num == 0:
        layer.set(arr)
      else:
        layer.observe()
    if learn:
      for layer in self.layers:
        layer.perceive()

  def predict(self):
    """Returns 2D numpy array of bottom (leaf) layer prediction."""
    prediction = None
    for i in reversed(xrange(len(self.layers))):
      # Predict top down, as in brain, because the context flows down.
      prediction = self.layers[i].predict()
    return prediction
