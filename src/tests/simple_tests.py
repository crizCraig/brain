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
    """Set some parameters to speed up testing and some class level parameters."""

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

  def testInitMultiLayer(self):
    b = Brain(num_layers=2, neurons_in_leaf_layer=256)
    self.assertEqual(b.layers[1].neurons.size, 64)

  def learnAndTestPredictions(self, input_frames):
    """Teach brain with input_frames and make sure predicted frames match."""
    b = Brain(num_layers=2, neurons_in_leaf_layer=256)

    # Empty 2D array is always first predicted frame because we have nothing to learnAndTestPredictions on.
    empty_frame = numpy.zeros((b.layers[0].height, b.layers[0].width), dtype=numpy.int32)
    predicted_frames = [empty_frame.tolist()]
    input_frames = map(lambda flat_arr: numpy.array(flat_arr), input_frames)
    layer_frames = [[] for x in range(b.num_layers - 1)]

    for frame in input_frames:
      b.perceive(frame, learn=True)
      for layer_index, layer in enumerate(b.layers[1:], start=1):
        # Record all non-leaf layers, so we can se what's hapenning inside the brain.
        # Leaf layer is just straight-up input.
        layer_frames[layer_index - 1].append(b.layers[layer_index].state().tolist())

    for i in range(Neuron.MAX_HISTORY):
      b.perceive(empty_frame, learn=False)
      # Let some time pass so brain doesn't predict replay

    for index, frame in enumerate(input_frames[:-1]): # Predict what happens after frames 1 through n - 1
      b.perceive(frame, learn=False)
      prediction = b.predict().tolist()
      print index
      print frame
      print numpy.array(prediction)
      predicted_frames.append(prediction)
      if not self.PASS_TO_HTML_VIEWER and prediction != input_frames[index + 1]:
        expected = numpy.array(input_frames[index + 1])
        actual = numpy.array(prediction)
        self.fail('Frame: ' + str(index + 1) + ' doesn\'t match prediction.\n\n' +
                  'Predicted:\n' + str(actual) + '\n\n' +
                  'Got:\n' + str(expected))
    src.util.writeFrames(layer_frames, 'layers', self.curr_test_name)
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
  unittest.main()