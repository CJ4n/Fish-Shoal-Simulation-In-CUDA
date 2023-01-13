#include "boid.h"

Boid *init_boids()
{
    srand(0);
    Boid *boids;
    cudaMallocManaged(&boids, sizeof(Boid));
    cudaMallocManaged(&(boids->coord), sizeof(*(boids->coord)) * num_boids);
    cudaMallocManaged(&(boids->velocity), sizeof(*(boids->velocity)) * num_boids);

    for (int boid = 0; boid < num_boids; ++boid)
    {
        float x = rand() % box_size;
        float y = rand() % box_size;
        float z = (is_3d != 0) ? rand() % box_size : 0;
        boids->coord[boid] = make_float3(x, y, z);

        float x_v = rand() % velocity_num_degres;
        float y_v = rand() % velocity_num_degres;
        float z_v = (is_3d != 0) ? rand() % velocity_num_degres : 0;
        float sum_v = x_v + y_v;
        if (sum_v > 0)
        {
            x_v = x_v / sum_v;
            y_v = y_v / sum_v;
        }
        if (rand() % 2 == 0)
        {
            x_v = -x_v;
        }

        if (rand() % 2 == 0)
        {
            y_v = -y_v;
        }
        if (rand() % 2 == 0)
        {
            z_v = -z_v;
        }
        boids->velocity[boid] = make_float3(x_v, y_v, z_v);
    }
    return boids;
}

void free_boids(Boid*boids){
    cudaFree(boids->coord);
    cudaFree(boids->velocity);
    cudaFree(boids);
}

