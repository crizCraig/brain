import sys
import numpy
import random

class Neuron(object):
  # Max size of sliding history window.
  MAX_HISTORY = 5

  # To avoid massive recalculation.
  HISTORY_RANGE = range(1, MAX_HISTORY + 1)

  # Maximum distance neurons can be to connect with each other in the same layer.
  LOCALITY_DISTANCE = 8 # 16 x 16 connection square

  # The fraction of a neuron's total connections that are siblings connections. (Based off biological brains.)
  SIBLING_CONNECTION_RATIO = 0.9

  # The maximum value for a connection between two neurons.
  MAX_CONNECTION_STRENGTH = sys.maxint

  # The number of times over which a neuron needs to see a pattern in order to predict it.
  CONNECTION_THRESHOLD = 0 #17

  def __init__(self, x, y, layer):
    """Construct a neuron and initialize its connections.

    Args:
      layer: Layer in which this neuron resides.
      x: Horizontal position within layer.
      y: Vertical position within layer.
    """
    self.layer = layer
    self.x = x
    self.y = y

    # Init dicts of connections with neurons as keys and integer connection strengths for values.
    self.parents = {} # A neuron can have more than two parents
    self.children = {}
    self.siblings = {}

    # Dict of siblings that predict this neuron.
    self.predictor_siblings = [set() for i in xrange(self.MAX_HISTORY)]

    # TODO: Make siblings a 3D array (x, y, t) of connections to allow
    # for more numpy speediness.

    # Current state of neuron.
    self.is_on = False

    # The minimum number of frames ago that this neuron was on.
    # This is base 1 so value of 1 means just on whereas
    # value of zero means the neuron has not been on recently.
    self.last_on = 0

  def initConnections(self):
    """Initialize the connections of this neuron to other neurons.

    We call this after __init__ so that parent neurons exist.
    Sibling connections are initialized one-way.
    The reverse direction will get initialized when the sibling calls initConnections().
    Child/parent connections are initialized two-way so we only call this once per child/parent connection.
    """

    # Square with self at center.
    self.total_siblings = min(
      2 * self.LOCALITY_DISTANCE ** 2,
      self.layer.width * self.layer.height) # Use entire layer if smaller.

    # ~10k per neuron in our brain.
    self.total_connections = \
      self.total_siblings / self.SIBLING_CONNECTION_RATIO

    self.total_children = (self.total_connections - self.total_siblings) / 2

    self.initSiblingConnections()
#    if self.layer.layer_num > 0:
#      self.initChildConnections()

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
    """Initialize child connections within twice the radius of the
    locality distance on the layer below.

    This is done to siphon larger areas of input into the increasingly smaller layers as we move up the hierarchy.
    Similarly, the layers in the human visual cortex sample larger visual fields further up the hierarchy.
    See layer V3 here: http://en.wikipedia.org/wiki/Visual_cortex
    """
    child_layer = self.layer.child
    rel_x = float((self.x + 1)) / self.layer.width
    rel_y = float((self.y + 1)) / self.layer.height
    center_x = int(round(rel_x * child_layer.width))
    center_y = int(round(rel_y * child_layer.height))
    distance = self.LOCALITY_DISTANCE * 2
    min_x, max_x, min_y, max_y = self._minMaxXY(
      center_x,
      center_y,
      child_layer.width,
      child_layer.height,
      distance)
    coordinates_added = set()
    child_count = 0

    # Pick a random child relatively 'close by' that hasn't already been added.
    # Collisions are limited because there are X times more potential
    # connections than children.
    # where X =
    # (SIBLING_CONNECTION_RATIO / (1 - SIBLING_CONNECTION_RATIO)) ** 2 = 81
    while child_count < self.total_children:
      x = random.randint(min_x, max_x)
      y = random.randint(min_y, max_y)
      if (x, y) not in coordinates_added:
        coordinates_added.add((x, y))
        child = child_layer.neurons[y, x]
        self.children[child] = 0
        child.parents[self] = 0
        child_count += 1

  def set(self, state):
    """Set boolean state of neuron. Only called on leaf neurons."""
    if self.is_on:
      self.last_on = 1
    elif self.last_on:
      self.last_on += 1
      if self.last_on > self.MAX_HISTORY:
        self.last_on = 0

    self.is_on = state

  def getZ(self):
    """Property representing vertical level within hierarchy."""
    return self.layer.layer_num
  z = property(getZ)

  def __cmp__(self, other):
    """Compare by coordinates. Top left of leaf layer is less than all others."""
    return cmp(self.coordinates(), other.coordinates())

  def coordinates(self):
    """Return location within brain as z, y, x."""
    return self.z, self.y, self.x

  def observe(self):
    """Funnel in input below from below."""
    if self.layer.layer_num > 0:
      for child in self.children:
        if child.is_on:
          self.set(True)
          return

  def perceive(self):
    """Adjust connection strengths with other neurons.
    Strengthen sibling connections from previous time cycle per STDP.
    http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
    """
    if self.is_on:
      sibs = self.siblings
      for delay in self.HISTORY_RANGE:
        predictor_sibs = self.predictor_siblings[delay - 1]
        for sib, strength in sibs.items():
          if sib.last_on == delay:
            # Learn
            # TODO: Do we need a max connection strength?
            strength = min(strength + 2, self.MAX_CONNECTION_STRENGTH)
            if strength >= self.CONNECTION_THRESHOLD:
              predictor_sibs.add(sib)
            sibs[sib] = strength
#          else:
#            # Forget
#            strength = max(strength - 1, 0)
#            if strength < self.CONNECTION_THRESHOLD:
#              try:
#                predictor_sibs.remove(sib)
#              except KeyError, e:
#                pass
#            sibs[sib] = strength


      # TODO: Update children connections (increments of 10?).
#      for child, connection_strength in self.children.items():
#        if child.wasOn():
#          self.increaseCxnStrength(self.children, child, amount=10)
#        else:
#          self.decreaseCxnStrength(self.children, child, amount=5)
      # TODO: If neuron off, should we decrease connections with neurons that predicted it?
      #self.printConnections(self.siblings)

  def predict(self):
    """Return a bool representing whether this neuron is predicted to fire during the next time cycle."""
    potential = 0
    POTENTIAL_THRESHOLD = 1
    for delay in self.HISTORY_RANGE:
      for sib in self.predictor_siblings[delay - 1]:
        if sib.last_on == delay:
          potential += 1

  #        else:
  #          # The predictive sibling is off, so decrease the likelihood of firing.
           # I disabled this because it wasn't working well, and upon reflection is not a good idea.
  #          potential -= 1
    return potential > POTENTIAL_THRESHOLD

  def printConnections(self, connections):
    """Pretty prints connections within same layer as 2D array.

    Args:
      connections: List of connections, i.e. siblings or children.
    """
    sample_cxn = connections.keys()[0]
    to_print = numpy.ones((sample_cxn.layer.height, sample_cxn.layer.width), dtype=numpy.int8) * -1
    for neuron, connection_strength in connections.items():
      to_print[neuron.y, neuron.x] = connection_strength
    print to_print

def create_neuron(x, y, layer):
  return Neuron(x, y, layer)
