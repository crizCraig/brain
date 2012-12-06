import unittest
import numpy
import json
import os
from src.brain import Brain
from src.neuron import Neuron
import src.util

class TestSimple(unittest.TestCase):
  # Unless true, will pass all tests and write predicted frames to HTML viewer.
  AUTOMATED_TEST = False

  def setUp(self):
    """Set some parameters to speed up testing and some class level parameters."""

    self.ORIGINAL_NEURON_CONNECTION_THRESHOLD = Neuron.CONNECTION_THRESHOLD
    # These are just simple test cases.
    # Let the brain learn the pattern in one cycle to speed things up.
    Neuron.CONNECTION_THRESHOLD = 1

    self.ORIGINAL_NEURON_LOCALITY_DISTANCE = Neuron.LOCALITY_DISTANCE
#    # Set the locality distance to the whole area
#    # so all positions are predictable for testing.
#    Neuron.LOCALITY_DISTANCE = 16

  def testLineMoveRight(self):
    """Test predicting vertical line moving left to right."""
    self.learnAndTestPredictions(self.getFrames('lines'))
#
#  def testLineJumping(self):
#    """Test predicting vertical line jumping around to a repeating pattern of random columns."""
#    input_frames = self.getFrames('lines')
#    random.shuffle(input_frames)
#    self.learnAndTestPredictions(input_frames)

  def testBouncingPixel(self):
    """Test predicting pixel that bounces around."""
    self.learnAndTestPredictions(self.getFrames('bouncing_pixel'))

  def testBounceThenLine(self):
    """Test predicting pixel that bounces around."""
    self.learnAndTestPredictions(self.getFrames('bounce_then_line'))

#  def testInitMultiLayer(self):
#    b = Brain(num_layers=2, neurons_in_leaf_layer=256)
#    self.assertEqual(b.layers[1].neurons.size, 64)

  def testSetNeuron(self):
    n = Neuron(0, 0, None)
    assert(n.last_on == 0) # No history.
    n.set(True)
    assert(n.last_on == 0) # Is currently on, but not was on.
    n.set(False)
    assert(n.last_on == 1) # Was just on.
    n.set(False)
    assert(n.last_on == 2) # On two frames ago.
    n.set(True)
    assert(n.last_on == 3) # On now and three frames ago.
    n.set(False)
    assert(n.last_on == 1) # Was just on again.
    for i in xrange(Neuron.MAX_HISTORY):
      n.set(False)
    assert(n.last_on == 0) # Too long ago to remember.

  def learnAndTestPredictions(self, input_frames):
    """Teach brain with input_frames and make sure predicted frames match."""
    b = Brain(num_layers=2, neurons_in_leaf_layer=256)

    # Empty 2D array is always first predicted frame because we have nothing to learnAndTestPredictions on.
    empty_frame = numpy.zeros(
      (b.layers[0].height, b.layers[0].width), dtype=numpy.int32)
    predicted_frames = []
    input_frames = map(lambda flat_arr: numpy.array(flat_arr), input_frames)
    actual_layer_frames = [[] for _ in range(b.num_layers)]
    predicted_layer_frames = [[] for _ in range(b.num_layers)]

    def record():
      if self.AUTOMATED_TEST:
        return

      for layer_index, layer in enumerate(b.layers):
        actual_layer_frames[layer_index].append(
          b.layers[layer_index].state().tolist())

        predicted_layer_frames[layer_index].append(
          b.layers[layer_index].predict().tolist())

    # Learn the sequence.
    for frame in input_frames:
      b.perceive(frame, learn=True)
      record()

    # Let some time pass so brain doesn't predict replay.
    for i in range(Neuron.MAX_HISTORY):
      b.perceive(empty_frame, learn=False)
      record()

    for index, frame in enumerate(input_frames):
      # Now that we've learned, test predictions.
      b.perceive(frame, learn=False)
      prediction = b.predict().tolist() # Bottom layer prediction.
      record()
#      print index
#      print frame
#      print numpy.array(prediction)
      predicted_frames.append(prediction)
      if self.AUTOMATED_TEST and prediction != input_frames[index + 1]:
        expected_prediction = numpy.array(input_frames[index + 1])
        actual_prediction = numpy.array(prediction)
        self.fail('Frame: ' + str(index + 1) + ' doesn\'t match prediction.\n\n' +
                  'Predicted:\n' + str(actual_prediction) + '\n\n' +
                  'Got:\n' + str(expected_prediction))
    src.util.writeFrames(actual_layer_frames, 'actual', self.curr_test_name)
    src.util.writeFrames(predicted_frames, 'predicted', self.curr_test_name)

  def getFrames(self, name):
    self.curr_test_name = name
    js = open(os.path.join('data', 'json', name, 'actual.js')).read()
    # Remove JavaScript variable declaration and convert JSON.
    return json.loads(js[js.index('=') + 1:].strip())

  def tearDown(self):
    # Restore class level parameters we changed.
    Neuron.CONNECTION_THRESHOLD = self.ORIGINAL_NEURON_CONNECTION_THRESHOLD
    Neuron.LOCALITY_DISTANCE = self.ORIGINAL_NEURON_LOCALITY_DISTANCE

if __name__ == '__main__':
  import cProfile
  cProfile.run("unittest.main()")