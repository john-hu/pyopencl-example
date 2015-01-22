__kernel void baseline(__global unsigned int* data)
{
  atom_inc(&data[get_global_id(0)]);
}
