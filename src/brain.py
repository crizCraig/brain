import math
class Brain(object):

  # This goes along with Hawkins spatial pooling theory. i.e. Concepts are
  # abstracted by correlating patterns that occur close to eachother. I don't
  # see any evidence for accomplishing this via layer contraction within
  # neuroscience. Rather distance from input may be the only factor needed
  # to accomplish abstraction. However, I'm leaving this here as a starting-point
  # for future experimentation.
  # For example the visual cortex in humans has over 1 billion neurons
  # http://www.ncbi.nlm.nih.gov/pubmed/7244322
  # And V1 has about 14% of that.
  # http://www.klab.caltech.edu/~harel/fun/v1.html
  # Here's a nice image:
  # http://benniemols.blogspot.com/2013_03_01_archive.html
  LAYER_CONTRACTION_RATIO = 1

  # This goes along with Hawkins' temporal pooling theory. i.e. Concepts are
  # abstracted by correlating events that occur around the same time.
  # I don't see any evidence for accomplishing this via firing rate.
  # It seems that the simple (prediction / observation / learning)
  # processes within each neuron accomplishes this more elegantly and in a more
  # parallelizable way.
  # LAYER_SLOWDOWN_RATIO = 0.5

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
    self.appendLayers()
    self.initConnections()

  def appendLayers(self):
    # Make layers squares.
    self.leaf_layer_width = self.leaf_layer_height = math.sqrt(self.neurons_in_leaf_layer)
    self.layers = []
    num_neurons = self.neurons_in_leaf_layer
    for i in xrange(self.num_layers):
      self.appendLayer(i, self.num_layers, num_neurons)
      # Spatial pooling. Layers are squares so num_neurons decreases by
      # layer contraction ratio squared.
      num_neurons *= (self.LAYER_CONTRACTION_RATIO ** 2)

  def appendLayer(self, i, num_layers, num_neurons):
    from layer import Layer
    self.layers.append(Layer(num_neurons=num_neurons,
                             layer_num=i,
                             brain=self,
                             is_top=(True if i == num_layers - 1 else False)))
    if i > 0:
      # Set child and parent layers
      self.layers[i - 1].parent = self.layers[i]
      self.layers[i].child = self.layers[i - 1]

  def initConnections(self):
    for layer in self.layers:
      # TODO, use nditer for speed (order doesn't matter).
      for neuron in layer.neurons.flat:
        neuron.initConnections()

  def perceive(self, signal, learn):
    """Take a 2D array and feed it to the leaf layer. Then iterate it up the tree."""
    #TODO: Try to feed input to more than just leaf layer to simulate visual cortex.
    #TODO: Some neurons (color) are more sensitive than others allowing for increasing resolution with longer exposure.
    #TODO: Cortical magnification for attention: http://en.wikipedia.org/wiki/Cortical_magnification

    for (layer_num, layer) in enumerate(self.layers):
        layer.predict()
        layer.observe(signal)
        if learn:
          layer.learn()

  # No longer needed since prediction happens at neuron level.
  # def predict(self):
  #   """Returns 2D numpy array of bottom (leaf) layer prediction."""
  #   prediction = None
  #   for i in reversed(xrange(len(self.layers))):
  #     # TODO: Put this in separate thread.
  #     # Predict top down, as in brain, because the context flows down.
  #     prediction = self.layers[i].predict()
  #   return prediction