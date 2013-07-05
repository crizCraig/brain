import numpy
import random

from connection import Connection


class Neuron(object):
  # Neurons follow a PREDICT | OBSERVE | LEARN process.
  #
  # PREDICT
  # Check parent and sibling connections for clues as to what will happen
  # next based off of what happened in the past. This input alone may be
  # enough to cause the neuron to fire, depending on what happens in the OBSERVE
  # phase.
  #
  # OBSERVE
  # Take input from child neurons flowing from senses (aka feedforward input)
  # and compare it with the potential from PREDICT. If firing was not predicted
  # by current connections then LEARN connections to neurons that did predict
  # our firing. If we were correctly predicted, still LEARN, but don't spend as
  # much time and energy as if the pattern were novel.
  #
  # LEARN
  # Adjust all of our connections to neurons around us if we just fired.
  # If a neuron around us fired recently as well, then strengthen the
  # connection. If it did not, then weaken the connection. Very weak connections
  # will become inhibitory.


  # TODO: Perhaps this should be renamed to column.
  # Why? Because this has feedforward connections (children),
  # feedback connections(parents), and intra-layer connections (siblings),
  # as well as a non-overlapping receptive field with other neurons.

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
  # This is much lower than sibling triggering threshold according to Hawkins
  # (see Berkeley talk), because feed-forward connections, i.e. proximal
  # child dendrites, represent a higher significance signal per connection,
  # albeit through less connections.
  PARENT_TRIGGERING_THRESHOLD = 0.001

  # Child connections (and parent ones) are thick within columns despite being
  # less in number. Therefore
  CHILD_TRIGGERING_THRESHOLD = 0.001

  # Amount above triggering threshold that we care about.
  # Stimulation beyond this amount can be ignored for efficiency purposes.
  # For example, a novel input triggers high excitation of a higher layer
  # neuron due to several child connections sending a high signal.
  # If our triggering threshold is 10 and 100 children are sending us a 1x
  # signal, then we can safely fire after looking at 20 children with our
  # highest excitation rate.
  THRESHOLD_SIZE = 2.0

  # This is a way of signaling to higher layers that input was novel and
  # is therefore important. I'm leaving this at 1.0 for now because novelty
  # is already detected by prioritizing neuron learning after missed predictions.
  # However, real neurons do seem to fire faster for novel input.
  # There's also some variability here between new neurons and old ones.
  # http://well.blogs.nytimes.com/2013/07/03/how-exercise-can-calm-anxiety/
  # This level should also probably fluctuate at the brain level with
  # along with some sort of energy value that fluctuates according to
  # how much total novelty is being observed. For example, a movie with a random
  # picture flashed 30 times per second should be ignored.
  NOVELTY_POTENTIAL_BOOST = 1.0

  # How to weight the potential of a firing neighbor neuron. Some theories say
  # only the connection strength matters, not the firing rate of the neighbor.
  # This parameter exists to test that theory with a genetic algorithm or
  # similar later on.
  IMPORTANCE_OF_NEIGHBOR_POTENTIAL = 0

  def __init__(self, x, y, layer):
    """Construct a neuron and initialize its connections.

    Args:
      layer: Layer in which this neuron resides.
      x: Horizontal position within layer.
      y: Vertical position within layer.
    """
    self.layer = layer
    self.brain = self.layer.brain

    self.x = x
    self.y = y

    self.potential = 0

    #TODO: Create ConnectionGroup container class, a constituent of the Column/Neuron class.
    self.parent_connections  = []
    self.child_connections   = []
    self.sibling_connections = []

    # Connections to siblings that predict self.
    self.strong_sibling_connections = [set() for _ in xrange(self.MAX_HISTORY)]
    self.strong_child_connections   = [set() for _ in xrange(self.MAX_HISTORY)]
    self.strong_parent_connections  = [set() for _ in xrange(self.MAX_HISTORY)]

    # TODO?: Make siblings a 3D array (x, y, t) of connections to allow for more
    # numpy speediness.

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
    self.initSiblingConnections()
    if self.layer.child:
      self.initChildConnections()
    if self.layer.parent:
      self.initParentConnections()

  def initSiblingConnections(self):
    """Create connections with neurons on same layer."""
    self.initConnectionsForLayer(layer=self.layer,
                                 center_x=self.x,
                                 center_y=self.y,
                                 connections=self.sibling_connections,
                                 distance=self.SIBLING_LOCALITY_DISTANCE)

  def initChildConnections(self):
    """Create connections with neurons in the layer closer to input.

    To the extent there is more than one child connection, lower level data
    is 'pooled' or siphoned into higher level neurons, thus abstracting
    the lower level information.
    Similarly, the layers in the human visual cortex sample larger visual fields
    further up the hierarchy. See info on layer V3 here:
    http://en.wikipedia.org/wiki/Visual_cortex or
    http://hilinkit.appspot.com/y9qdtf
    """
    center_x, center_y = self.relativePositionWithinLayer(self.layer.child)
    self.initConnectionsForLayer(layer=self.layer.child,
                                 connections=self.child_connections,
                                 center_x=center_x,
                                 center_y=center_y,
                                 distance=self.CHILD_LOCALITY_DISTANCE)

  def initParentConnections(self):
    """Create connections with neurons in the layer farther from input.

    Parent connections start out more noisy than child connections due to lack
    of input from the outside world. Eventually, however, as the brain's model
    of the world strengthens, parent signals will come to be more and more
    predictive of this neuron's state and therefore *should* become stronger
    than child connections.
    # TODO: Test this theory.
    """
    center_x, center_y = self.relativePositionWithinLayer(self.layer.parent)
    self.initConnectionsForLayer(layer=self.layer.parent,
                                 connections=self.parent_connections,
                                 center_x=center_x,
                                 center_y=center_y,
                                 distance=self.PARENT_LOCALITY_DISTANCE)

  def relativePositionWithinLayer(self, layer):
    """ Return relative position of self within another layer."""
    rel_x = float(self.x + 1) / layer.width
    rel_y = float(self.y + 1) / layer.height
    center_x = int(round(rel_x * layer.child.width)) - 1
    center_y = int(round(rel_y * layer.child.height)) - 1
    return center_x, center_y

  def initConnectionsForLayer(self, layer, connections, center_x, center_y,
                              distance):
    """ Add connections to neurons in `layer` within specified `distance`."""
    min_x, max_x, min_y, max_y = self._minMaxXY(
      center_x,
      center_y,
      layer.width,
      layer.height,
      distance)
    for y in xrange(min_y, max_y + 1):
      for x in xrange(min_x, max_x + 1):
        neighbor = layer.neurons[y, x]
        if neighbor != self:
          connections.append(Connection(to=neighbor))

  def _minMaxXY(self, x, y, width, height, distance):
    """Returns bounds for 2D area of potential connections."""
    # TODO: Wrap.
    min_x = max(0, x - distance)
    max_x = min(width - 1, x + distance)
    min_y = max(0, y - distance)
    max_y = min(height - 1, y + distance)
    return min_x, max_x, min_y, max_y

  def set(self, state):
    """Set boolean state of neuron. Only called on leaf neurons."""
    if self.is_on:
      self.last_on = 1
    elif self.last_on:
      self.last_on += 1
      if self.last_on > self.MAX_HISTORY:
        self.last_on = 0

    self.is_on = state

  def observe(self):
    """Funnel input from below."""
    if not self.layer.is_bottom:
      # Layer zero gets set directly to input for now.

      # TODO: Sparsify input by creating an LGN type of preprocessing
      # that sends color, background, intensity, contrast, areas of uniform color
      # etc...
      #
      # From How to Create a Mind:
      # In a study published in Nature, Frank S. Werblin, professor of molecular
      # and cell biology at the University of California at Berkeley, and
      # doctoral student Boton Roska, MD, showed that the optic nerve carries
      # ten to twelve output channels, each of which carries only a small amount
      # of information about a given scene.2 One group of what are called
      # ganglion cells sends information only about edges (changes in contrast).
      # Another group detects only large areas of uniform color, whereas a third
      # group is sensitive only to the backgrounds behind figures of interest.


      self.is_on = self.testConnections(self.strong_child_connections,
                                        self.CHILD_TRIGGERING_THRESHOLD)

      if not self.predicted:
        # Unexpected firings get a stronger activation akin to higher firing
        # rate in real neurons.
        self.potential *= self.NOVELTY_POTENTIAL_BOOST

        # PERF: Subsample less neurons when predictions are being met to
        # conserve resources.

  def learn(self):
    """Adjust connection strengths with other neurons.
    Strengthen sibling connections from previous time cycle per STDP.
    http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity

    TODO: Learn subset of connections online and adjust all connections offline
          over longer periods of time. (Like hippocampus / sleep patterns)
    """
    if self.is_on and (

      # If reality not predicted by the past, learn.
      not self.predicted or

      # Take time to reinforce what we already know, but don't spend too much
      # energy on it.
      random.random() < 0.15
    ):

      self.learn_from_children()
      self.learn_from_parents()
      self.learn_from_siblings()

    self.resetPotential()

  def learn_from_children(self):
    self.learn_from(self.child_connections, self.strong_child_connections)

  def learn_from_parents(self):
    self.learn_from(self.parent_connections, self.strong_parent_connections)

  def learn_from_siblings(self):
    self.learn_from(self.child_connections, self.strong_sibling_connections)

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

  def resetPotential(self):
    """Reset potential to quiet state."""
    self.potential = 0

  def potentialFromConnections(self, connections, delay, max_potential,
                               current_potential):
    for connection in connections:
      if connection.neighbor.last_on == delay:
        current_potential += self.potentialFromConnection(connection)
        if current_potential > max_potential:
          return current_potential
    return current_potential

  def potentialFromConnection(self, connection):
    return connection.strength * self.intensityBoost(connection)

  def intensityBoost(self, connection):
    """Figure in strength of signal to increase importance of novel patterns."""
    return self.IMPORTANCE_OF_NEIGHBOR_POTENTIAL * connection.neighbor.potential

  def predict(self):
    """Return a bool representing whether this neuron is predicted to fire."""

    # Predict from parents first since there are less connections and
    # we can avoid extra work if they predict us.

    self.predicted = (self.testConnections(self.strong_parent_connections,
                              self.PARENT_TRIGGERING_THRESHOLD) or
                      self.testConnections(self.strong_sibling_connections,
                              self.SIBLING_TRIGGERING_THRESHOLD))
    return self.predicted

  def testConnections(self, connections, threshold):
    max_potential = threshold * self.THRESHOLD_SIZE
    for delay in self.HISTORY_RANGE:
      self.potential += self.potentialFromConnections(connections[delay - 1],
                                                 delay, max_potential)
      if self.potential > max_potential:
        # set of connections is different than normal connections.
        break

    fire = self.potential > threshold
    return fire

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


# Kurzweil notes:
# Only higher level primates have spindle neurons with connections that span entire brain.
# We'll need a recursive layer that has lots of connections to itself.
# The thalamus appears to talk a lot with layer six of the cortex.
# Scale should be adjusted by focusing preprocessors that are specific to input (vision)

