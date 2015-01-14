// 1024 pixels per work item.
#define BIN_SIZE 1024
#define RESULT_SIZE 768

__kernel void histogram(__global unsigned char* bytes, __global unsigned int* result)
{
  unsigned int lid = get_local_id(0);
  unsigned int gid = get_group_id(0);
  unsigned int gsize = get_local_size(0);
  unsigned int globalId = get_global_id(0);

  unsigned int i, bValue;
  unsigned int basePixelIdx = lid * BIN_SIZE + gid * gsize * BIN_SIZE;
  unsigned int baseResultIdx = globalId * RESULT_SIZE;

  for (i = 0; i < RESULT_SIZE; i++) {
    result[baseResultIdx + i] = 0;
  }

  for (i = 0; i < BIN_SIZE; i++) {
    // R byte of this pixel is (basePixelIdx + i) * 3
    bValue = bytes[basePixelIdx * 3 + i * 3];
    // the R index of baseResultIndex + bValue * 3
    // Data partition is RGBRGBRGBRGB...RGB. If the R = 1, the R index is 3.
    result[baseResultIdx + bValue]++;
    bValue = bytes[basePixelIdx * 3 + i * 3 + 1];
    result[baseResultIdx + 256 + bValue]++;
    bValue = bytes[basePixelIdx * 3 + i * 3 + 2];
    result[baseResultIdx + 512 + bValue]++;
  }
}
