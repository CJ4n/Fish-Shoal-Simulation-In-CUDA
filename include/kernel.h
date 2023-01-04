#pragma once

#include "boid.h"

__global__ void update_boids_position(float3 *pos,  int width,  int height, float time);
__global__ void simple_vbo_kernel(float3 *pos,  int width,  int height, float time);
void launch_kernel(float3 *pos,  int mesh_width,  int mesh_height, float time,Boid* boids);
Boid *init_boids();