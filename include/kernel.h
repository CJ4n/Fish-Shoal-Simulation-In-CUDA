// #include "constants.h"
// #include "boid.h"

// Boid *init_boids();
__global__ void update_boids_position(float4 *pos, unsigned int width, unsigned int height, float time);
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time);
void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);


struct Boid
{
    float *x_coords;
    float *y_coords;
};
