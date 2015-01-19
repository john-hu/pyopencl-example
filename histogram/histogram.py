import argparse
import sys
import numpy
import math
import pyopencl as cl
from time import time
from PIL import Image

def cpu_histogram(img):
  return img.histogram()

def opencl_histogram(pixels):
  # format of pixels is RGBRGBRGB each of character in a byte
  # calculate buffer size
  groupSize = 1
  binSize = 1024
  pixelSize = len(pixels) / 3 
  trunkSize = int(math.ceil(math.ceil(pixelSize / groupSize) / binSize))
  globalSize = int(math.ceil(pixelSize / binSize))
  globalSize += (groupSize - globalSize % groupSize)
  # buffer size is 768(whole space) * group size * trunk size
  outputBufSize = 768 * groupSize * trunkSize

  #create context/queue
  clContext = cl.create_some_context()
  clQueue = cl.CommandQueue(clContext)
  f = open('histogram.c', 'r')
  fstr = ''.join(f.readlines())
  # create the program
  clProgram = cl.Program(clContext, fstr).build()
  # create buffers
  mf = cl.mem_flags
  bufPixels = cl.Buffer(clContext, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=pixels)
  bufPixelSize = cl.Buffer(clContext, mf.READ_ONLY | mf.USE_HOST_PTR, size=4, hostbuf=numpy.asarray([pixelSize]).astype(numpy.uint32))
  bufTempResult = cl.Buffer(clContext, mf.READ_WRITE, size=outputBufSize * 4, hostbuf=None)
  bufOutput = cl.Buffer(clContext, mf.WRITE_ONLY, size=768 * 4, hostbuf=None)
  # execute program
  clProgram.histogram(clQueue, (globalSize, ), None,
                            bufPixels, bufPixelSize, bufTempResult, bufOutput)
  # read data back
  finalResult = numpy.zeros(768, dtype=numpy.uint32)
  cl.enqueue_read_buffer(clQueue, bufOutput, finalResult)
  clQueue.finish()
  return finalResult

parser = argparse.ArgumentParser(description='Dump histogram data.')
parser.add_argument('--input', help='the input image')
parser.add_argument('--dump', help='dump the histogram')

args = parser.parse_args()

if args.input is None:
  parser.print_help()
  sys.exit(1)
print ('trying to build histogram data for {}'.format(args.input))

image = Image.open(args.input)

(width, height) = image.size

print ('-' * 20)
# the histogram format is RRRR...RRGGGG...GGGBBB...BBB.
start_time = time()
histogramC = cpu_histogram(image)
end_time = time()
print ('time elapsed with sequential CPU: {0}s'.format(end_time - start_time))

print ('-' * 20)

start_time = time()
histogramG = opencl_histogram(image.tobytes())
end_time = time()
print ('time elapsed with open cl: {0}s'.format(end_time - start_time))

histogram = histogramC;

print ('-' * 20)
print ('file.mode: {}'.format(image.mode))
print ('file.size: {0}x{1}'.format(width, height))
print ('file.format: {}'.format(image.format))
print ('-' * 20)
print ('(size: {0})'.format(len(histogram)))
if args.dump is not None:
  for i in range(256):
    print ('R: {0}, G: {0}, B: {0} => ({1}, {2}, {3})'.format(i,
                                                        histogram[i],
                                                        histogram[256 + i],
                                                        histogram[256 * 2 + i]))
print ('=' * 20)
