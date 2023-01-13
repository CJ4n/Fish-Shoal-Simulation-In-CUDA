#pragma once
#include "config.h"

struct Boid
{
    DataType *coord;
    DataType *velocity;
};

void free_boids(Boid*boids);
Boid *init_boids();