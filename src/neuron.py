import sys
import numpy
import random
import math

class Neuron(object):
  # How far back each neuron can remember.
  MAX_HISTORY = 5

  # Range of history values, to avoid recalculation.
  HISTORY_RANGE = range(1, MAX_HISTORY + 1)

  # Maximum distance neurons can be from each other to connect within same layer.
  LOCALITY_DISTANCE = 8 # 16 x 16 connection square

  # The fraction of a neuron's total connections that are siblings connections.
  # (Based off biological brains.)
  SIBLING_CONNECTION_RATIO = 0.9

  # The maximum value for a connection between two neurons.
  MAX_CONNECTION_STRENGTH = sys.maxint

  # The number of times a neuron needs to see a pattern to predict it.
  CONNECTION_THRESHOLD = 0 #17

  STDP_INCR = 10
  STDP_DECR = 1

  def __init__(self, x, y, layer):
    """Construct a neuron and initialize its connections.

    Args:
      layer: Layer in which this neuron resides.
      x: Horizontal position within layer.
      y: Vertical position within layer.
    """
    self.layer = layer
    self.brain = layer.brain
    self.x = x
    self.y = y

    # Init dicts of connections with neurons as keys and integer connection strengths for values.
    self.parents  = {} # A neuron can have more than two parents
    self.children = {}
    self.siblings = {}

    # Dict of siblings that predict this neuron.
    self.predictor_siblings = [set() for i in xrange(self.MAX_HISTORY)]

    # TODO?: Make siblings a 3D array (x, y, t) of connections to allow for more numpy speediness.

    # Current state of neuron.
    self.is_on = False

    # The minimum number of frames ago that this neuron was on.
    # This is base 1 so a value of 1 means just on whereas
    # value of zero means the neuron has not been on recently.
    self.last_on = 0

  def initConnections(self):
    """Initialize the connections of this neuron to other neurons.

    We call this after __init__ so that parent neurons exist.
    Sibling connections are initialized one-way.
    The reverse direction will get initialized when the sibling calls initConnections().
    Child/parent connections are initialized two-way so we only call this once per child/parent connection.
    """

    self.initSiblingConnections()

    # Square with self at center.
    self.total_siblings = min(2 * self.LOCALITY_DISTANCE ** 2, self.layer.width * self.layer.height)

    # ~10k per neuron in our brain.
    self.total_connections = \
      self.total_siblings / self.SIBLING_CONNECTION_RATIO

    self.total_children = (self.total_connections - self.total_siblings) / 2

    if self.layer.layer_num > 0:
      self.initChildConnections()

  def _minMaxXY(self, x, y, width, height, distance):
    """Returns bounds for 2D area of potential connections."""
    min_x = max(0, x - distance)
    max_x = min(width - 1, x + distance)
    min_y = max(0, y - distance)
    max_y = min(height - 1, y + distance)
    return min_x, max_x, min_y, max_y

  def initSiblingConnections(self):
    """Create connections with neurons on same layer within
    self.LOCALITY_DISTANCE."""

    min_x, max_x, min_y, max_y = self._minMaxXY(
      self.x,
      self.y,
      self.layer.width,
      self.layer.height,
      self.LOCALITY_DISTANCE)

    for y in xrange(min_y, max_y + 1):
      for x in xrange(min_x, max_x + 1):
        if not(x == self.x and y == self.y):
          self.siblings[self.layer.neurons[y, x]] = 0

  def initChildConnections(self):
    """Initialize child connections.

    To the extent there is more than one child connection, lower level data
    is 'pooled' or siphoned into higher level neurons, thus abstracting
    the low level information.
    Similarly, the layers in the human visual cortex sample larger visual fields further up the hierarchy.
    See layer V3 here: http://en.wikipedia.org/wiki/Visual_cortex or http://hilinkit.appspot.com/y9qdtf
    """
    child_layer = self.layer.child
    rel_x = float(self.x + 1) / self.layer.width
    rel_y = float(self.y + 1) / self.layer.height
    center_x = int(round(rel_x * child_layer.width)) - 1
    center_y = int(round(rel_y * child_layer.height)) - 1
    distance = int(math.ceil(self.total_children / 2)) # Relies on layers being squares.

    # TODO: Shift towards center if hits edge.
    min_x, max_x, min_y, max_y = self._minMaxXY(
      center_x,
      center_y,
      child_layer.width,
      child_layer.height,
      distance)

    for x in xrange(min_x, max_x):
      for y in xrange(min_y, max_y):
        child = child_layer.neurons[y, x]
        self.children[child] = 0
        child.parents[self] = 0

  def set(self, state):
    """Set boolean state of neuron. Only called on leaf neurons."""
    if self.is_on:
      self.last_on = 1
    elif self.last_on:
      self.last_on += 1
      if self.last_on > self.MAX_HISTORY:
        self.last_on = 0

    self.is_on = state

  def get_z(self):
    """Property representing vertical level within hierarchy."""
    return self.layer.layer_num
  z = property(get_z)

  def __cmp__(self, other):
    """Compare by coordinates. Top left of leaf layer is less than all others."""
    return cmp(self.coordinates(), other.coordinates())

  def coordinates(self):
    """Return location within brain as z, y, x."""
    return self.z, self.y, self.x

  def observe(self):
    """Funnel in input from below."""
    if self.layer.layer_num > 0:
      for child in self.children:
        if child.is_on:
          self.set(True)
          return

  def learn(self):
    """Adjust connection strengths with other neurons.
    Strengthen sibling connections from previous time cycle per STDP.
    http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity

    Basically, go connections to this neuron on the same layer
    and see if they predict if this neuron is going to turn on.
    """

#  if not predicted (by siblings or parents)
#    for sibling in self.siblings:
#      adjust_connection()
#        for sibling.history
#          if sibling was predicted
#            strengthen connection

    # If reality not predicted by the past, learn.

    # Do not take state directly from parents though. Parents only predict.
    # For example, if I play piano and miss a note, the prediction of the correct
    # note from above was incorrect. Learn the input that caused the incorrect
    # prediction and perhaps even learn that it leads to the negative consequence.
    # i.e. by strengthening an inhibitory connection. (Some connections in the
    # brain are known to do this as well. http://en.wikipedia.org/wiki/Neurotransmitter#Excitatory_and_inhibitory)
    # If input was predicted, we should either ignore the input (saving cycles) or strengthen the connections that
    # predicted it based on some randomness function.
    # Also, we should detect when novelty of input is too high at the layer level and ignore it if it is.

    if self.is_on:
      if self.predicted:
        # TODO: For some percentage of the time, strengthen siblings that predicted me.
        pass
      else:
        # Learn from siblings that predicted me
        siblings = self.siblings # Speed up lookup by making siblings a local variable.
        for delay in self.HISTORY_RANGE:

          # Get siblings from previous frame so we can see if they fired and predicted this neuron.
          predictor_siblings = self.predictor_siblings[delay - 1]

          for sibling, strength in siblings.items():
            if sibling.last_on == delay:
              # Sibling predicted me, strengthen my connection with this sibling.
              strength = min(strength + self.STDP_INCR, self.MAX_CONNECTION_STRENGTH)
              if strength >= self.CONNECTION_THRESHOLD:
                predictor_siblings.add(sibling)
              siblings[sibling] = strength
            else:
              # Sibling did not predict me, weaken connection with this sibling slightly.
              strength = max(strength - self.STDP_DECR, 0)

              if strength < self.CONNECTION_THRESHOLD:
                try:
                  predictor_siblings.remove(sibling)
                except KeyError, e:
                  # Sibling was not a predictor.
                  pass
              siblings[sibling] = strength

      pass
      # TODO: Update children connections half the time.
      # If didn't predict children, learn from them.
      #

#      for child, connection_strength in self.children.items():
#        if child.wasOn():
#          self.increaseCxnStrength(self.children, child, amount=10)
#        else:
#          self.decreaseCxnStrength(self.children, child, amount=5)
      # TODO: If neuron off, should we decrease connections with neurons that predicted it?
      #self.print_connections(self.siblings)

  def predict(self):
    """Return a bool representing whether this neuron is predicted to fire during the next time cycle."""
    potential = 0
    POTENTIAL_THRESHOLD = 1
    # TODO: Look at parents.
    for delay in self.HISTORY_RANGE:
      for sib in self.predictor_siblings[delay - 1]:
        if sib.last_on == delay:
          potential += 1

  #        else:
  #          # The predictive sibling is off, so decrease the likelihood of firing.
           # I disabled this because it wasn't working well, and upon reflection is not a good idea.
  #          potential -= 1
    return potential > POTENTIAL_THRESHOLD

  def print_connections(self, connections):
    """Pretty prints connections within same layer as 2D array.

    Args:
      connections: List of connections, i.e. siblings or children.
    """
    sample_cxn = connections.keys()[0]
    to_print = numpy.ones((sample_cxn.layer.height, sample_cxn.layer.width), dtype=numpy.int8) * -1
    for neuron, connection_strength in connections.items():
      to_print[neuron.y, neuron.x] = connection_strength
    print to_print
