import sys
import numpy
import random

class Neuron(object):
  # Max size of sliding history window.
  MAX_HISTORY = 5

  # Maximum distance neurons can be to connect with each other in the same layer.
  LOCALITY_DISTANCE = 8 # 16 x 16 connection square

  # The fraction of a neuron's total connections that are siblings connections. (Based off biological brains.)
  SIBLING_CONNECTION_RATIO = 0.9

  # The maximum value for a connection between two neurons.
  MAX_CONNECTION_STRENGTH = sys.maxint

  # The number of times a neuron sees a pattern in order to predict it.
  CONNECTION_THRESHOLD = 17

  def __init__(self, layer, x, y):
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
    self.parents = {}
    self.children = {}
    self.siblings = {}

    # On/off neuron state history, most recent at end.
    self.history = [False]

  def initConnections(self):
    """Initialize the connections of this neuron to other neurons.

    Sibling connections are initialized one-way.
    The reverse direction will get initialized when the sibling calls initConnections().
    Child/parent connections are initialized two-way so we only call this once per child/parent connection.
    """
    self.total_siblings = (2 * self.LOCALITY_DISTANCE) ** 2 # Square with self at center.
    self.total_connections = self.total_siblings / self.SIBLING_CONNECTION_RATIO # ~10k in our brain.
    self.total_children = (self.total_connections - self.total_siblings) / 2

    self.initSiblingConnections()
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
    """Create connections with all neurons on same layer within self.LOCALITY_DISTANCE."""
    min_x, max_x, min_y, max_y = self._minMaxXY(self.x, self.y, self.layer.width, self.layer.height,
                                               self.LOCALITY_DISTANCE)
    for y in xrange(min_y, max_y + 1):
      for x in xrange(min_x, max_x + 1):
        if not(x == self.x and y == self.y):
          self.siblings[self.layer.neurons[y, x]] = 0

  def initChildConnections(self):
    """Initialize child connections within twice the area of the locality distance.

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
    min_x, max_x, min_y, max_y = self._minMaxXY(center_x, center_y, child_layer.width, child_layer.height, distance)

    coordinates_added = set()
    child_count = 0
    while child_count < total_children:
      # Pick a random child that's relatively 'close by' and hasn't already been added.
      # Collisions are limited because there are x times more potential connections than children,
      # where x = (SIBLING_CONNECTION_RATIO / (1 - SIBLING_CONNECTION_RATIO)) ** 2 = 81
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
    if len(self.history) >= self.MAX_HISTORY:
      self.history.pop(0)
    self.history.append(state)

  def isOn(self):
    return self.history[-1]

  def wasOn(self):
    try:
      return self.history[-2]
    except IndexError:
      # Just born.
      return False

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

  def increaseConnectionStrength(self, connections, key):
    """Increase connection strength by two if less than maximum connection strength."""
    if connections[key] < self.MAX_CONNECTION_STRENGTH - 1:
      connections[key] += 2

  def decreaseConnectionStrength(self, connections, key):
    """Decrease connection strength by one if greater than zero."""
    if connections[key] > 0:
      connections[key] -= 1

  def perceive(self):
    """Adjust connection strengths with other neurons."""
    if self.isOn():
      for neuron, connection_strength in self.siblings.items():
        # Strengthen sibling connections from previous time cycle per STDP.
        # http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
        if neuron.wasOn():
          self.increaseConnectionStrength(self.siblings, neuron)
        else:
          self.decreaseConnectionStrength(self.siblings, neuron)
      # TODO: Update children connections (increments of 10?).
      # TODO: Update parent connections.
      # TODO: If neuron off, should we decrease connections with neurons that predicted it?
    #self.printConnections(self.siblings)

  def predict(self):
    """Return a bool representing whether this neuron is predicted to fire during the next time cycle."""
    direction = 0
    for neuron, connection_strength in self.siblings.items():
      if connection_strength >= self.CONNECTION_THRESHOLD:
        # TODO: Maintain separate list of 'connected' neurons if this gets slow.
        if neuron.isOn():
          direction += 1
        else:
          direction -= 1
    return direction > 0

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
