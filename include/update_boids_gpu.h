#include "boid.h"

__global__ void update_boids_position(Boid *boids, float interaction_radius_2, float velocity, float factor_separation,
                                      float factor_alignment, float factor_cohesion, float factor_intertia);