import unittest
import numpy
import json
import os
import random
from src.brain import Brain
from src.neuron import Neuron
import src.util

class TestSimple(unittest.TestCase):
  # Will pass all tests and write predicted frames to HTML viewer if True.
  PASS_TO_HTML_VIEWER = True

  def setUp(self):
    """Set some parameters to speed up testing and w/e class level parameters."""

    # These are just simple test cases. Let the brain learn the pattern in one cycle to speed things up.
    self.ORIGINAL_NEURON_CONNECTION_THRESHOLD = Neuron.CONNECTION_THRESHOLD
    Neuron.CONNECTION_THRESHOLD = 1

    # Set the locality distance to the whole area so all positions are predictable for testing.
    self.ORIGINAL_NEURON_LOCALITY_DISTANCE = Neuron.LOCALITY_DISTANCE
    Neuron.LOCALITY_DISTANCE = 16

  def testLineMoveRight(self):
    """Test predicting vertical line moving left to right."""
    self.learnAndTestPredictions(self.getFrames('lines'))

  def testLineJumping(self):
    """Test predicting vertical line jumping around to a repeating pattern of random columns."""
    input_frames = self.getFrames('lines')
    random.shuffle(input_frames)
    self.learnAndTestPredictions(input_frames)

  def testBouncingPixel(self):
    """Test predicting pixel that bounces around."""
    self.learnAndTestPredictions(self.getFrames('bouncing_pixel'))

  def learnAndTestPredictions(self, input_frames):
    """Teach brain with input_frames and test predicted frames match."""
    b = Brain(num_layers=1, neurons_in_leaf_layer=256)
    LEARN_CYCLES = 2
    # Empty 2D array is always first predicted frame because we have nothing to learnAndTestPredictions on.
    predicted_frames = [numpy.zeros((b.layers[0].height, b.layers[0].width), dtype=numpy.int32).tolist()]

    for cycle_index in xrange(LEARN_CYCLES):
      for frame_index in xrange(len(input_frames)):
        pixels = numpy.array(input_frames[frame_index])
        b.perceive(pixels)
        prediction_frame_index = frame_index + 1
        if cycle_index == LEARN_CYCLES - 1 and prediction_frame_index < len(input_frames):
          # Test on last cycle, after learning.
          prediction = map(lambda x: x.tolist(), b.predict())
          predicted_frames.append(prediction)
          if not self.PASS_TO_HTML_VIEWER and prediction != input_frames[prediction_frame_index]:
            expected = numpy.array(input_frames[prediction_frame_index])
            actual = numpy.array(prediction)
            self.fail("Frame: " + str(prediction_frame_index) + " doesn't match prediction.\n\n" +
                      "Predicted:\n" + str(actual) + "\n\n" +
                      "Got:\n" + str(expected))
    src.util.writeFrames(predicted_frames, self.curr_test_name, prediction=True)

  def getFrames(self, name):
    self.curr_test_name = name
    js = open(os.path.join('data', 'json', name, 'in.js')).read()
    # Remove JavaScript variable declaration and convert JSON.
    return json.loads(js[js.index('=') + 1:].strip())


  def tearDown(self):
    # Restore class level parameters we changed.
    Neuron.CONNECTION_THRESHOLD = self.ORIGINAL_NEURON_CONNECTION_THRESHOLD
    Neuron.LOCALITY_DISTANCE = self.ORIGINAL_NEURON_LOCALITY_DISTANCE

if __name__ == '__main__':
  unittest.main()