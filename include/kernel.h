#pragma once

#include "boid.h"

__global__ void update_boids_position(float4 *pos, unsigned int width, unsigned int height, float time);
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time);
void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time,Boid* boids);
Boid *init_boids();