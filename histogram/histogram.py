import argparse
import sys
import numpy
import math
import pyopencl as cl
from time import time
from PIL import Image

def cpu_histogram(img):
  return img.histogram()

def opencl_histogram(img):
  # format of pixels is RGBRGBRGB each of character in a byte
  pixels = image.tobytes()
  # calculate buffer size
  groupSize = 4
  binSize = 1024
  pixelSize = len(pixels) / 3 
  trunkSize = int(math.ceil(math.ceil(pixelSize / groupSize) / binSize))
  globalSize = int(math.ceil(pixelSize / binSize))
  globalSize += (groupSize - globalSize % groupSize)
  # buffer size is 768(whole space) * group size * trunk size
  outputBufSize = 768 * groupSize * trunkSize
  print 'pixel count: {}, trunk count: {}, buffer size: {}, global size: {}'.format(pixelSize, trunkSize, outputBufSize, globalSize)
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
  bufOutput = cl.Buffer(clContext, mf.WRITE_ONLY, size=outputBufSize * 4, hostbuf=None)
  start_time = time()
  clProgram.histogram(clQueue, (globalSize, ), (groupSize, ), bufPixels, bufOutput)
  end_time = time()
  print ('time: {}'.format(end_time - start_time))
  semiFinal = numpy.zeros(outputBufSize, dtype=numpy.uint32)
  evt = cl.enqueue_read_buffer(clQueue, bufOutput, semiFinal)
  evt.wait()

  finalResult = [0] * 768
  for i in range(outputBufSize):
    finalResult[i % 768] += semiFinal[i]

  return finalResult

parser = argparse.ArgumentParser(description='Dump histogram data.')
parser.add_argument('--input', help='the input image')

args = parser.parse_args()

if args.input is None:
  parser.print_help()
  sys.exit(1)
print ('trying to build histogram data for {}'.format(args.input))

image = Image.open(args.input)

(width, height) = image.size

# start_time = time()
# histogram = cpu_histogram(image)
# end_time = time()

start_time = time()
histogram = opencl_histogram(image)
end_time = time()

print ('-' * 20)
print ('time elapsed: {0}s'.format(end_time - start_time))
print ('file.mode: {}'.format(image.mode))
print ('file.size: {0}x{1}'.format(width, height))
print ('file.format: {}'.format(image.format))
print ('-' * 20)
print ('(size: {0})'.format(len(histogram)))
for i in range(256):
  print ('R: {0}, G: {0}, B: {0} => ({1}, {2}, {3})'.format(i,
                                                          histogram[i],
                                                          histogram[256 + i],
                                                          histogram[256 * 2 + i]))
print ('=' * 20)
