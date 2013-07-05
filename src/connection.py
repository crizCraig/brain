import sys
class Connection(object):
  """
  There a lot of these, so maybe we will probably need to implement in
  C/assembley/CUDA eventually.
  """

  # The maximum value for a connection between two neurons.
  MAX_CONNECTION_STRENGTH = sys.maxint

  # The minimum value for a connection between two neurons.
  # Negative connections are inhibitory.
  MIN_CONNECTION_STRENGTH = -sys.maxint - 1

  # Minimum strength of a strong predictive connection.
  PREDICTIVE_CONNECTION_THRESHOLD = 0 #17

  # Maximum strength of a strong inhibitory connection.
  INHIBITORY_CONNECTION_THRESHOLD = -10

  # Amount to increase connection strength when neighboring neuron fires before
  # self is on. STDP = Spike Timing Dependent Plasticity
  # http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
  STDP_INCREMENT = 10

  # Amount to decrease connection strength when neighboring neuron fires before
  # self is off. STDP = Spike Timing Dependent Plasticity
  # http://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
  # Single pyramidal cell receives about 30,000 excitatory inputs and 1700
  # inhibitory (IPSPs) inputs.
  # http://hilinkit.appspot.com/yx45sw
  # http://en.wikipedia.org/wiki/Pyramidal_cell
  STDP_DECREMENT = 1

  def __init__(self, to=None):
    self.neighbor = to
    self.strength = 0

  def adjust_strength(self, delay, strong_connections_for_frame):
    """ Set connection strength to newly learned value. """
    if self.neighbor.last_on == delay:
      self.boost_strength(strong_connections_for_frame)
    else:
      self.decrease_strength(strong_connections_for_frame)

  def boost_strength(self, strong_connections_for_frame):
    self.strength = min(self.strength + self.STDP_INCREMENT,
                        self.MAX_CONNECTION_STRENGTH)

    if self.strength >= self.PREDICTIVE_CONNECTION_THRESHOLD:
      self.add_to(strong_connections_for_frame)
    elif self.strength > self.INHIBITORY_CONNECTION_THRESHOLD:
      self.remove_from(strong_connections_for_frame)

  def decrease_strength(self, strong_connections_for_frame):
    self.strength = max(self.strength - self.STDP_DECREMENT,
                        self.MIN_CONNECTION_STRENGTH)

    if self.strength < self.PREDICTIVE_CONNECTION_THRESHOLD:
      self.remove_from(strong_connections_for_frame)
    elif self.strength <= self.INHIBITORY_CONNECTION_THRESHOLD:
      self.add_to(strong_connections_for_frame)

  def add_to(self, strong_connections_for_frame):
    strong_connections_for_frame.add(self)

  def remove_from(self, strong_connections_for_frame):
    try:
      strong_connections_for_frame.remove(self)
    except KeyError, _:
      # Sibling was not a predictor.
      pass