#include "boid.h"

__global__ void draw_boids(float3 *pos, Boid *boids, int num_boids);