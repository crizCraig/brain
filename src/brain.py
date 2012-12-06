import math
class Brain(object):
  def __init__(self, num_layers, neurons_in_leaf_layer):
    """
    Build an empty brain

    Arguments:
    num_layers -- Synonymous with layers of cortex.
      Hierarchical layers of brain. Should be less than six.
    neurons_in_leaf_layer -- Number of neurons in bottom layer of hierarchy.
    neurons_per_region -- Number of neurons that can form connections
      with each other within a layer.
      Since regions don't connect with other regions on the same layer,
      we can parallelize region calculations.

    """
    self.num_layers = num_layers
    self.neurons_in_leaf_layer = neurons_in_leaf_layer

    from layer import Layer
    # Make layers squares.
    self.leaf_layer_width = self.leaf_layer_height =\
      math.sqrt(neurons_in_leaf_layer)

    self.layers = []
    num_neurons = self.neurons_in_leaf_layer
    for i in xrange(num_layers):
      self.layers.append(Layer(num_neurons=num_neurons,
                               layer_num=i,
                               brain=self,
                               is_top=(True if i == num_layers - 1 else False)))

      num_neurons /= 4 # Halve each side.

      if i > 0:
        # Set children / parents.
        self.layers[i - 1].parent = self.layers[i]
        self.layers[i].child = self.layers[i - 1]

    for layer in self.layers:
      # TODO, use nditer for speed (order doesn't matter).
      for neuron in layer.neurons.flat:
        neuron.initConnections()


  def perceive(self, signal, learn):
    """Take a 2D array and feed it to the leaf layer. Then iterate it up the tree."""
    # TODO: Try to feed input to more than just leaf layer to simulate visual cortex.

    for (layer_num, layer) in enumerate(self.layers):
      # TODO: Fire each layer half as often as its child layer.
      if layer_num != 0 and layer.expected(signal):
        # Input was predicted, don't waste energy processing it.
        # TODO: Do this at the neuron level, not the entire layer! (Some neurons (color) are more sensitive than others.)
        # This is another way of doing spatial pooling (besides the pyramidal structure).
        # We could do random boosts like Numenta,
        # but the visual system doesn't do that and it always seemed kind of kludgy to me anyway.
        break
      else:
        # Input was not predicted, learn from it.
        layer.observe(signal)
        if learn:
          layer.learn()

  def predict(self):
    """Returns 2D numpy array of bottom (leaf) layer prediction."""
    prediction = None
    for i in reversed(xrange(len(self.layers))):
      # TODO: Put this in separate thread.
      # Predict top down, as in brain, because the context flows down.
      prediction = self.layers[i].predict()
    return prediction