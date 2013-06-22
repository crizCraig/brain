import numpy
import random
import math

from connection import Connection


class Neuron(object):
  # Simulates propagation delay from neuron to neuron by storing history
  # and checking for casual relationships throughout a range of time.
  MAX_HISTORY = 5

  # Range of history values, to avoid recalculation.
  HISTORY_RANGE = range(1, MAX_HISTORY + 1)

  # Maximum distance neurons can be from each other to connect within same layer.
  SIBLING_LOCALITY_DISTANCE = 8 # 16 x 16 connection square

  # Maximum in-plane distance a child neuron can be from self.
  CHILD_LOCALITY_DISTANCE = 2

  # Maximum in-plane distance a parent neuron can be from self.
  PARENT_LOCALITY_DISTANCE = 2

  # The fraction of a neuron's total connections that are siblings connections.
  # (Based off biological brains.)
  SIBLING_CONNECTION_RATIO = 0.9

  # The fraction of a neuron's total connections that are child connections.
  CHILD_CONNECTION_RATIO = .05

  # The fraction of a neuron's total connections that are parent connections.
  PARENT_CONNECTION_RATIO = .05

  # Equivalent to the sum of excitatory postsynaptic potentials and
  # inhibitory postsynaptic potentials necessary for a predictive sibling to
  # trigger the firing of a child.
  # See: http://en.wikipedia.org/wiki/Axon_hillock
  SIBLING_TRIGGERING_THRESHOLD = 1

  # Equivalent to the sum of excitatory postsynaptic potentials and
  # inhibitory postsynaptic potentials necessary for a predictive parent to
  # trigger the firing of a child.
  # See: http://en.wikipedia.org/wiki/Axon_hillock
  CHILD_TRIGGERING_THRESHOLD = 10

  def __init__(self, x, y, layer):
    """Construct a neuron and initialize its connections.

    Args:
      layer: Layer in which this neuron resides.
      x: Horizontal position within layer.
      y: Vertical position within layer.
    """
    self.layer = layer
    self.brain = self.layer.brain

    if self.layer.layer_num == 0:
      # This neuron is in the leaf layer closest to sensory data.
      self.child_layer = None
    else:
      self.child_layer = self.brain.layers[self.layer.layer_num - 1]

    if self.layer.is_top:
      # This neuron is in the root layer farthest from sensory data.
      self.parent_layer = None
    else:
      self.parent_layer = self.brain.layers[self.brain.num_layers - 1]

    self.x = x
    self.y = y

    self.parent_connections  = []
    self.child_connections   = []
    self.sibling_connections = []

    # Connections to siblings that predict self.
    self.predictive_sibling_connections = [set() for _ in xrange(self.MAX_HISTORY)]
    self.predictive_child_connections   = [set() for _ in xrange(self.MAX_HISTORY)]
    self.predictive_parent_connections  = [set() for _ in xrange(self.MAX_HISTORY)]

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
    The reverse direction will get initialized when the sibling calls
    initConnections(). Child/parent connections are initialized two-way so we
    only call this once per child/parent connection.
    """

    # Siblings
    self.initSiblingConnections()

    # Children
    if self.child_layer:
      self.total_children = min(
        2 * self.CHILD_LOCALITY_DISTANCE ** 2,
        self.child_layer.width * self.child_layer.height
      )
      self.initChildConnections()

    # Parents
    if self.parent_layer:
      self.total_parents = min(
        2 * self.PARENT_LOCALITY_DISTANCE ** 2,
        self.layer.width * self.layer.height
      )
      self.initParentConnections()

  def _minMaxXY(self, x, y, width, height, distance):
    """Returns bounds for 2D area of potential connections."""
    min_x = max(0, x - distance)
    max_x = min(width - 1, x + distance)
    min_y = max(0, y - distance)
    max_y = min(height - 1, y + distance)
    return min_x, max_x, min_y, max_y

  def _neuronsWithinSquare(self, distance, layer):
    """ Return count of neurons within square with self at center.
    *--------* <-*
    |        |   | <- x
    |   []   | <-*
    |        |
    *--------*
    ^---2x---^

     AREA = 2x^2

    If this square doesn't fit in layer, make it as big as possible. #TODO: Wrap layer.
    """
    min(2 * distance ** 2, layer.width * layer.height)

  def initSiblingConnections(self):
    """Create connections with neurons on same layer within
    self.LOCALITY_DISTANCE."""

    self.total_siblings = self._neuronsWithinSquare(
      self.SIBLING_LOCALITY_DISTANCE, self.layer)

    min_x, max_x, min_y, max_y = self._minMaxXY(
      self.x,
      self.y,
      self.layer.width,
      self.layer.height,
      self.SIBLING_LOCALITY_DISTANCE)

    for y in xrange(min_y, max_y + 1):
      for x in xrange(min_x, max_x + 1):
        if not (x == self.x and y == self.y):
          self.sibling_connections.append(
            Connection(to=self.layer.neurons[y, x]))

  def initChildConnections(self):
    """Initialize child connections.

    To the extent there is more than one child connection, lower level data
    is 'pooled' or siphoned into higher level neurons, thus abstracting
    the lower level information.
    Similarly, the layers in the human visual cortex sample larger visual fields further up the hierarchy.
    See info on layer V3 here: http://en.wikipedia.org/wiki/Visual_cortex or
    http://hilinkit.appspot.com/y9qdtf
    """
    child_layer = self.layer.child
    rel_x = float(self.x + 1) / self.layer.width
    rel_y = float(self.y + 1) / self.layer.height
    center_x = int(round(rel_x * child_layer.width)) - 1
    center_y = int(round(rel_y * child_layer.height)) - 1

    min_x, max_x, min_y, max_y = self._minMaxXY(
      center_x,
      center_y,
      child_layer.width,
      child_layer.height,
      self.CHILD_LOCALITY_DISTANCE)

    for x in xrange(min_x, max_x):
      for y in xrange(min_y, max_y):
        child = child_layer.neurons[y, x]
        self.child_connections.append(Connection(to=child))

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
      for child in self.child_connections:
        # TODO: Sum all predicting connections and test against threshold.
        # TODO: Subsample less neurons when predictions are being met to conserve resources.

        # Neurons which connect to output (motor control) will
        # fire sporadically at first since feedback connections will be
        # weak and unvaried.
        # Similarly feedback connections will have a lower correspodence
        # with firing in early training as compared to feedforward connections
        # that are grabbing input from a more structured world.
        # Once structure is introduced into the brain, however, this will
        # change and feedback connections will exert more control over the
        # neuron.
        if child.is_on:
          self.set(True)
          return

  def learn(self):
    """Adjust connection strengths with other neurons.
    Strengthen sibling connections from previous time cycle per STDP.
    :param self:
    http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity

    TODO: Learn subset of connections online and adjust all connections offline
          over longer periods of time.
    """
    if self.is_on and (

      # If reality not predicted by the past, learn.
      not self.predicted or

      # Take  time to reinforce what we already know.
      # However, don't spend too much energy on it.
      random(1.0) < 0.15
    ):

      self.learn_from_children()
      self.learn_from_parents()
      self.learn_from_siblings()


  def learn_from_children(self):
    self.learn_from(self.child_connections, self.predictive_child_connections)

  def learn_from_parents(self):
    self.learn_from(self.parent_connections, self.predictive_parent_connections)

  def learn_from_siblings(self):
    self.learn_from(self.child_connections, self.predictive_sibling_connections)

  def learn_from(self, connections, predictive_connections):
    # Different connections have different propagation times.
    # See: https://www.dropbox.com/s/jt3vmmqyv5ldg3r/STDP%20timing.pdf
    for delay in self.HISTORY_RANGE:
      # Update connections by learning from neighbors that predicted me.
      # Get siblings from previous frame so we can see if they fired and
      # predicted this neuron.
      predictive_connections_for_frame = predictive_connections[delay - 1]
      for connection in connections:
        connection.adjust_strength(delay, predictive_connections_for_frame)

  def predict(self):
    """Return a bool representing whether this neuron is predicted to fire."""
    potential = 0

    for delay in self.HISTORY_RANGE:

      # Siblings
      for sibling_connection in self.predictive_sibling_connections[delay - 1]:
        if sibling_connection.neighbor.last_on == delay:
          potential += sibling_connection.strength

      # Parents
      for parent_connection in self.predictive_parent_connections[delay - 1]:
        if parent_connection.neighbor.last_on == delay:
          potential += parent_connection.strength

      # Children
      for child_connection in self.predictive_child_connections[delay - 1]:
        if child_connection.neighbor.last_on == delay:
          potential += child_connection.strength


    self.predicted = potential > self.SIBLING_TRIGGERING_THRESHOLD
    return self.predicted

  # def setPredicted(self):
  #   return 'psuedocode below'
  #
  #   sum = 0
  #   for parent in self.parent_connections:
  #     if self.isPredictedBy(parent):
  #       sum += 1
  #
  #   if sum > self.CHILD_TRIGGERING_THRESHOLD:
  #     return True
  #   else:
  #     return False

  def print_connections(self, connections):
    """Pretty prints connections within same layer as 2D array.

    Args:
      connections: List of connections, i.e. siblings or children.
    """
    sample_cxn = connections.keys()[0]
    to_print = numpy.ones((sample_cxn.layer.height, sample_cxn.layer.width),
                          dtype=numpy.int8) * -1
    for neuron, connection_strength in connections.items():
      to_print[neuron.y, neuron.x] = connection_strength
    print to_print
