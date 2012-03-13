(function() {
  $.each(tests.names, function(i, testName) {
    // Load the test data.
    $.each(['in', 'out'], function(i, name) {
      var script = document.createElement('script');
      script.type = 'text/javascript';
      script.src = 'tests/data/json/' + testName + '/' + name + '.js';
      document.body.appendChild(script);
    });
  });

  var testSelect = $('#test_select');
  testSelect.append(Mustache.to_html($('#test_select_template').html(), tests)); // Populate the test drop down.
  testSelect.change(function() {
    // Load the test when it's selected from the drop down.
    var test_name = testSelect.val();
    viewTest({
      'test_name': test_name,
      'actual': window[test_name],
      'predicted': window[test_name + '_predicted']
    });
  });

  function viewTest(data) {
    var actual = data.actual;
    var predicted = data.predicted;
    var UNIT_SIZE = 16;
    var SIDE_LENGTH = 16;
    var PIXEL_SIDE_LENGTH = UNIT_SIZE * SIDE_LENGTH;

    var actualCanvas = getCanvas('actual');
    var predictedCanvas = getCanvas('predicted');

    $.map($('.controls').children(), function(i, control) {
      $(control).unbind(); // Avoids playing two tests simultaneously.
    });

    var stepButton = $('#step');
    var stepIndex = 0;
    stepButton.click(function() {
      draw((stepIndex++ % actual.length), false); // Advance frame. Wrap at end.
    });

    $('#play').click(function() {
      draw(0);
    });

    function draw(index, play) {
      if(play === undefined) {
        play = true;
      }
      $.map([actualCanvas, predictedCanvas], clearCanvas);
      var frame = actual[index];
      for(var y = 0, yy = frame.length; y < yy; y++) {
        var row = frame[y];
        for(var x = 0, xx = row.length; x < xx; x++) {
          if(row[x] === 1) {
            drawBigPixel(actualCanvas, x, y);
          }
          if(predicted[index][y][x] === 1) {
            drawBigPixel(predictedCanvas, x, y);
          }
        }
      }
      if(play && index + 1 < actual.length) {
        setTimeout(function() {draw(index + 1);}, 250);
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