(function() {
  // Populate test data from test names in test_list.js.
  $.each(tests.names, function(i, testName) {
    // Load the test JSON.
    $.each(['actual', 'predicted'], function(i, name) {
      var script = document.createElement('script');
      script.type = 'text/javascript';
      script.src = 'tests/data/json/' + testName + '/' + name + '.js';
      document.body.appendChild(script);
    });
  });

  // Populate the test drop down.
  var testSelect = $('#test_select');
  testSelect.append(Mustache.to_html($('#test_select_template').html(), tests));

  // Load the test when it's selected from the drop down.
  testSelect.change(function() {
    var test_name = testSelect.val();
    viewTest({
      'test_name': test_name,
      'actual': window[test_name + '_actual'],
      'predicted': window[test_name + '_predicted']
    });
  });

  function viewTest(data) {
    var actual = data.actual;
    var predicted = data.predicted;
    var UNIT_SIZE = 16;
    var SIDE_LENGTH = 16;
    var PIXEL_SIDE_LENGTH = UNIT_SIZE * SIDE_LENGTH;
    var NUM_FRAMES = Math.min(actual.length, predicted.length);

    var actualCanvas = getCanvas('actual');
    var predictedCanvas = getCanvas('predicted');
    var frameNum = $('#frame_num');

    // Avoid playing two tests simultaneously.
    $('.controls button').unbind('click');

    var getNextFrameIndex = (function() {
      var stepIndex = 0;
      return function(args) {
        if(args.back === true) {
          if(stepIndex === 0) {
            stepIndex = NUM_FRAMES;
          }
          stepIndex--;
        }
        else {
          stepIndex++;
          if(stepIndex === NUM_FRAMES) {
            stepIndex = 0;
          }
        }
        return stepIndex;
      }
    })();

    var stepButton = $('#step');
    stepButton.click(function() {
      draw({'play': false}); // Advance frame. Wrap at end.
    });

    var backButton = $('#back');
    backButton.click(function() {
      draw({'play': false, 'back': true}); // Advance frame. Wrap at end.
    });

    $('#play').click(function() {
      draw({'play': true});
    });

    var timer;
    function draw(args) {
      var index = getNextFrameIndex(args);
      $.map([actualCanvas, predictedCanvas], clearCanvas);
      var frame = actual[index];
      for(var y = 0, yy = frame.length; y < yy; y++) {
        var row = frame[y];
        for(var x = 0, xx = row.length; x < xx; x++) {
          if(actual[index][y][x] > 0) {
            drawBigPixel(actualCanvas, x, y);
          }
          if(predicted[index][y][x] > 0) {
            drawBigPixel(predictedCanvas, x, y);
          }
          frameNum.html(index);
        }
      }
      if(args.play && index + 1 < actual.length) {
        timer = setTimeout(function() {draw(args);}, 100);
      }
      else {
        clearTimeout(timer);
      }
    }

    function drawBigPixel(canvas, x, y) {
      canvas.fillRect(x * UNIT_SIZE, y * UNIT_SIZE, UNIT_SIZE, UNIT_SIZE);
    }

    function clearCanvas(canvas) {
      canvas.clearRect(0, 0, PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH);
    }

    function getCanvas(id) {
      var canvas = document.getElementById(id);
      canvas.width = PIXEL_SIDE_LENGTH;
      canvas.height = PIXEL_SIDE_LENGTH;
      var context = canvas.getContext('2d');
      context.fillStyle = '#000';
      return context;
    }
  }
})();