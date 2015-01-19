// 1024 pixels per work item.
#define BIN_SIZE 1024
#define RESULT_SIZE 768

__kernel void histogram(__global unsigned char* bytes, __global unsigned int* pixelCount,
                        __global unsigned int* result)
{
  unsigned int lid = get_local_id(0);
  unsigned int gid = get_group_id(0);
  unsigned int gsize = get_local_size(0);
  unsigned int globalId = get_global_id(0);

  unsigned int i, bValue;
  unsigned int basePixelIdx = lid * BIN_SIZE + gid * gsize * BIN_SIZE;
  unsigned int baseResultIdx = globalId * RESULT_SIZE;
  unsigned int maxPixel = *pixelCount;
  unsigned int privateBuffer[RESULT_SIZE];

  for (i = 0; i < RESULT_SIZE; i++) {
    result[baseResultIdx + i] = 0;
  }

  unsigned int processIndex = 0;
  while (processIndex < BIN_SIZE && (basePixelIdx + processIndex < maxPixel)) {
    // data partition of bytes is RGBRGBRGB....
    bValue = bytes[basePixelIdx * 3 + processIndex * 3];
    // result partition is RR..RRGG..GGBB..BB.
    privateBuffer[bValue]++;
    // G
    bValue = bytes[basePixelIdx * 3 + processIndex * 3 + 1];
    privateBuffer[256 + bValue]++;
    // B
    bValue = bytes[basePixelIdx * 3 + processIndex * 3 + 2];
    privateBuffer[512 + bValue]++;
    processIndex++;
  }

  for (i = 0; i < RESULT_SIZE; i++) {
    result[baseResultIdx + i] = privateBuffer[i];
  }
}
