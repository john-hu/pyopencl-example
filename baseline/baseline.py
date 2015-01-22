import argparse
import pyopencl as cl
from time import time
from array import array

class Baseline:
  def __init__(self, options):
    self.options = options
    self.result = []

  def run(self):
    # measure elapsed time for creating context and queue
    tQueue = time()
    clContext = cl.create_some_context()
    clQueue = cl.CommandQueue(clContext)
    self.result.append({'type': 'create queue', 'time': (time() - tQueue)})
    # measure elapsed time for reading kernel program from file system
    tReadFile = time()
    f = open('baseline.c', 'r')
    fstr = ''.join(f.readlines())
    self.result.append({'type': 'read file', 'time': (time() - tReadFile)})
    # measure elapsed time for building the kernel program.
    tProgram = time()
    clProgram = cl.Program(clContext, fstr).build()
    self.result.append({'type': 'build program', 'time': (time() - tProgram)})
    # measure elapsed time for creating buffer.
    mf = cl.mem_flags
    pyBuffer = array('i', [0] * self.options.buffer_size)
    tBuffer = time()
    clBuffer = cl.Buffer(clContext, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=pyBuffer)
    cl.enqueue_write_buffer(clQueue, clBuffer, pyBuffer).wait()
    self.result.append({'type': 'create buffer', 'time': (time() - tBuffer)})
    # measure elapsed time for running the program
    tRun = time()
    clProgram.baseline(clQueue, (self.options.buffer_size, ), None, clBuffer)
    self.result.append({'type': 'run program', 'time': (time() - tRun)})
    # we don't need to measure the time for reading data back because we use
    # `USE_HOST_PTR` which creates the buffer in main memory instead of GPU's.
    return self.result

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test your machine. Don\'t forget to add PYOPENCL_CTX to choose the default platform')
  parser.add_argument('--buffer-size', help='the buffer size for test', default=1024 * 1024)
  args = parser.parse_args()
  b = Baseline(args)
  result = b.run()

  print '=' * 40
  print 'buffer size (in int): {}'.format(args.buffer_size)
  total = 0
  for data in result:
    total += data['time']
    print '{}: {:.6f}s'.format(data['type'], data['time'])
  print '-' * 40
  print 'total: {:.6f}s'.format(total)
  print '=' * 40