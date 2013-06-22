import sys
class Connection(object):
  """
  There a lot of these, so maybe we will probably need to implement in
  C/assembley/CUDA eventually.
  """

  # The maximum value for a connection between two neurons.
  MAX_CONNECTION_STRENGTH = sys.maxint

  MIN_CONNECTION_STRENGTH = -sys.maxint - 1

  # The number of times a neuron needs to see a pattern to predict it.
  PREDICTIVE_CONNECTION_THRESHOLD = 0 #17

  # Amount to increase connection strength when neighboring neuron fires before
  # self is on.
  STDP_INCREMENT = 10

  # Amount to decrease connection strength when neighboring neuron fires before
  # self is off.
  STDP_DECREMENT = 1

  def __init__(self, to=None):
    self.neighbor = to
    self.strength = 0

  def adjust_strength(self, delay, predictive_connections_for_frame):
    """ Set connection strenght to newly learned value. """
    if self.neighbor.last_on == delay:
      self.boost_strength(predictive_connections_for_frame)
    else:
      self.decrease_strength(predictive_connections_for_frame)

  def boost_strength(self, predictive_connections_for_frame):
    self.strength = min(self.strength + self.STDP_INCREMENT,
                        self.MAX_CONNECTION_STRENGTH)

    if self.strength >= self.PREDICTIVE_CONNECTION_THRESHOLD:
      predictive_connections_for_frame.add(self)

    # TODO: Check if we should remove from inhibitory connections.

  def decrease_strength(self, predictive_connections_for_frame):
    self.strength = max(self.strength - self.STDP_DECREMENT,
                        self.MIN_CONNECTION_STRENGTH)

    if self.strength < self.PREDICTIVE_CONNECTION_THRESHOLD:
      try:
        predictive_connections_for_frame.remove(neighbor)
        # TODO: add to inibitory connections.
      except KeyError, e:
        # Sibling was not a predictor.
        pass