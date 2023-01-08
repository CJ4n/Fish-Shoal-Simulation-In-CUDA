#pragma once
#include "config.h"

struct Boid
{
    DataType *coord;
    DataType *velocity;
};

Boid *init_boids();
__global__ void draw_boids(float3 *pos, Boid *boids, int num_boids);
__global__ void update_boids_position(Boid *boids, float interaction_radius_2, float factor_separation,
                                      float factor_alignment, float factor_cohesion, float factor_intertia);