from PIL import Image
import numpy
import os
import json

def main():
  imagesToJSON('lines')
  createBouncingPixel()
  pass

def imagesToJSON(folder_name):
  """Condense image files representing frames of sequential input into one JSON file to speed up reading."""
  path = os.path.join('tests', 'data', 'images', folder_name)
  file_names = [file_name for file_name in os.listdir(path) if not file_name.startswith('.')]
  frames = []
  for name in file_names:
    image = Image.open(os.path.join(path, name))
    arr = []
    width, height = image.size
    for y in xrange(height):
      row = []
      for x in xrange(width):
        row.append(1 if not image.getpixel((x,y)) else 0)
      arr.append(row)
    frames.append(arr)
  writeFrames(frames, folder_name)

def createBouncingPixel():
  """Create sample input that's just a bouncing pixel on a 16x16 square."""
  height = width = 16
  empty = numpy.zeros((height, width), dtype=numpy.int32)
  x = 0
  y = 0
  x_diff = 1
  y_diff = 1
  frames = []
  for i in xrange(100):
    copy = numpy.array(empty)
    if x + x_diff not in range(0, width):
      x_diff *= -1
    if y + y_diff not in range(0, height):
      y_diff *= -1
    copy[y, x] = 1
    print copy
    x += x_diff
    if i % 2:
      y += y_diff
    frames.append(copy.tolist())
  writeFrames(frames, 'bouncing_pixel')

def writeFrames(frames, folder_name, prediction=False):
  """Write a sequence of 2D frames to JSON."""
  file_name = 'out.js' if prediction else 'in.js'
  var_name = folder_name + ('_predicted' if prediction else '')
  out_file = open(os.path.join(('tests' if not prediction else ''),
                               'data', 'json', folder_name, file_name), 'w')
  out_file.write(var_name + ' = ' + json.dumps(frames))
  out_file.close()

if __name__ == '__main__':
  main()