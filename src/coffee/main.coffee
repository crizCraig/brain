$(window).load -> viewTest "bounce_then_line"

# Populate test data from test names in test_list.js.
$.each tests.names, (i, testName) ->
  # Load the test JSON.
  $.each [ "actual", "predicted" ], (i, name) ->
    script = document.createElement("script")
    script.type = "text/javascript"
    script.src = "tests/data/json/" + testName + "/" + name + ".js"
    document.body.appendChild script


# Populate the test drop down.
testSelect = $("#test_select")
testSelect.append Mustache.to_html($("#test_select_template").html(), tests)

# Load the test when it's selected from the drop down.
testSelect.change -> viewTest testSelect.val()

testSelect = $("#test_select")
testSelect.append Mustache.to_html($("#layer_template").html(), tests)

window.viewtest = viewTest = (name) ->
  timer = null
  draw = (args) ->
    index = getNextFrameIndex(args)
    $.map [ actualCanvas, predictedCanvas ], clearCanvas
    frame = actual[index]
    y = 0
    yy = frame.length

    while y < yy
      row = frame[y]
      x = 0
      xx = row.length

      while x < xx
        drawBigPixel actualCanvas, x, y  if actual[index][y][x] > 0
        drawBigPixel predictedCanvas, x, y  if predicted[index][y][x] > 0
        frameNum.html index
        x++
      y++
    if args.play and index + 1 < actual.length
      callback = -> draw args
      timer = setTimeout callback, 100
    else
      clearTimeout timer
    return true

  drawBigPixel = (canvas, x, y) ->
    canvas.fillRect x * UNIT_SIZE, y * UNIT_SIZE, UNIT_SIZE, UNIT_SIZE

  clearCanvas = (canvas) ->
    canvas.clearRect 0, 0, PIXEL_SIDE_LENGTH, PIXEL_SIDE_LENGTH

  getCanvas = (id) ->
    canvas = document.getElementById(id)
    canvas.width = PIXEL_SIDE_LENGTH
    canvas.height = PIXEL_SIDE_LENGTH
    context = canvas.getContext("2d")
    context.fillStyle = "#000"
    context

  actual = window[name + "_actual"]
  predicted = window[name + "_predicted"]
  UNIT_SIZE = 16
  SIDE_LENGTH = 16
  PIXEL_SIDE_LENGTH = UNIT_SIZE * SIDE_LENGTH
  NUM_FRAMES = Math.min(actual.length, predicted.length)
  actualCanvas = getCanvas("actual")
  predictedCanvas = getCanvas("predicted")
  frameNum = $("#frame_num")

  # Avoid playing two tests simultaneously.
  $(".controls button").unbind "click"

  getNextFrameIndex = (->
    stepIndex = 0 # Private closure variable for state.
    return (args) ->
      if not args.back
        stepIndex++
        stepIndex = 0  if stepIndex is NUM_FRAMES
      else
        stepIndex = NUM_FRAMES  if stepIndex is 0
        stepIndex--
      return stepIndex
  )()

  $("#play").click -> draw play: true
  $('#stop').click -> draw play: false
  $("#step").click -> draw play: false
  $("#back").click -> draw play: false, back: true


  return true
