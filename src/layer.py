import numpy
from neuron import Neuron

class Layer(object):
  """
  This represents a layer of the neo-cortex, of which there are around six in the human brain.
  Each layer halves as we move up the hierarchy, forming a three dimensional binary tree.
  """
  def __init__(self, num_neurons, layer_num=-1, parent=None, child=None, brain=None, is_top=False):
    """Initializes Layer with num_neurons x num_neurons array of Neurons.

    Args:
      num_neurons: Total number of neurons in layer
      layer_num: Level within the hierarchy, layer zero is the bottom a.k.a. leaf a.k.a. input layer.
      parent: Layer object above current layer, i.e. has higher layer_num.
      child: Layer object below current layer, i.e. has lower layer_num.
      brain: Brain object in which layer exists.
      is_top: Whether or not this layer is the top layer of the hierarchy.
    """
    self.layer_num = layer_num
    self.parent = parent
    self.child = child
    self.width = self.height = int(num_neurons ** 0.5) # Region should be a perfect square.
    self.brain = brain
    self.is_top = is_top
    neurons = []
    for y in xrange(self.height):
      row = []
      for x in xrange(self.width):
        row.append(Neuron(layer=self, x=x, y=y))
      neurons.append(row)
    self.neurons = numpy.array(neurons) # Two dimensional array of neurons.

  def set(self, arr):
    """Set each neuron in the layer, for feeding input into bottom layer.

    Args:
      arr: 2D numpy array of 1's and 0's
    """
    for y in xrange(self.height):
      for x in xrange(self.width):
        self.neurons[y, x].set(arr[y, x] == 1)

  def perceive(self):
    """Have each neuron interpret its current state relative to its connections."""
    for y in xrange(self.height):
      for x in xrange(self.width):
        self.neurons[y, x].perceive()

  def predict(self):
    """Returns predicted state for next time cycle."""
    ret = numpy.empty((self.height, self.width), dtype=numpy.int32)
    for y in xrange(self.height):
      for x in xrange(self.width):
        ret[y, x] = 1 if self.neurons[y, x].predict() else 0
    return ret