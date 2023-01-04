#include <stdlib.h>
#include <iostream>
#include "kernel.h"
#include "constants.h"

__global__ void simple_vbo_kernel(float3 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // write output vertex
    pos[y * width * 3 + x * 3] = make_float3(0, 0, 0);
    pos[y * width * 3 + x * 3 + 1] = make_float3(0, 0, 0);
    pos[y * width * 3 + x * 3 + 2] = make_float3(0, 0, 0);
}

__global__ void draw_boids(float3 *pos, Boid *boids, unsigned int width, unsigned int height, float time, int num_boids)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_boids)
    {
        return;
    }
    float x = boids->x_coords[gid];
    float y = boids->y_coords[gid];
    // write output vertex
    // if (x >= width || x < 0)
    //     return;
    // if (y >= width || y < 0)
    //     return;
    //     float u = x / (float)width;
    //     float v = y / (float)height;
    //     //  u = u * 2.0f - 1.0f;
    //     //  v = v * 2.0f - 1.0f;

    //     // calculate simple sine wave pattern
    //     // float freq = 4.0f;
    //     // float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    //     pos[gid *3 ] = make_float3(u, v, 0);
    //     pos[gid *3+1 ]  = make_float3((x + 1) / (float)width, v, 0);
    //    pos[gid *3 +2]  = make_float3(u, (y + 1) / (float)height, 0);

    pos[gid * 3] = make_float3(x/(float)width, y/(float)height, 0);
    pos[gid * 3 + 1] = make_float3((x + 1)/(float)width,y/(float)height, 0);
    pos[gid * 3 + 2] = make_float3(x/(float)width,(y+1)/(float)height, 0);
}

__global__ void update_boids_position(Boid *boid, unsigned int width, unsigned int height)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (boid->count <= gid)
    {
        return;
    }
    // boid->x_coords[gid] += boid->x_velocity[gid];
    // boid->y_coords[gid] += boid->y_velocity[gid];
}

Boid *init_boids()
{
    srand(0);
    Boid *boids;
    cudaMallocManaged(&boids, sizeof(Boid));
    boids->count = num_boids;

    cudaMallocManaged(&(boids->x_coords), sizeof(*(boids->x_coords)) * num_boids);
    cudaMallocManaged(&(boids->y_coords), sizeof(*(boids->y_coords)) * num_boids);
    cudaMallocManaged(&(boids->x_velocity), sizeof(*(boids->x_velocity)) * num_boids);
    cudaMallocManaged(&(boids->y_velocity), sizeof(*(boids->y_velocity)) * num_boids);

    for (int boid = 0; boid < num_boids; ++boid)
    {
        int x = rand() % mesh_width;
        int y = rand() % mesh_height;
        boids->x_coords[boid] = x;
        boids->y_coords[boid] = y;

        int x_v = rand() % (max_velocity - min_velocity) + min_velocity;
        int y_v = rand() % (max_velocity - min_velocity) + min_velocity;
        if (rand() % 2 == 0)
        {
            boids->x_velocity[boid] = x_v;
        }
        else
        {
            boids->x_velocity[boid] = x_v;
        }

        if (rand() % 2 == 0)
        {
            boids->y_velocity[boid] = y_v;
        }
        else
        {
            boids->y_velocity[boid] = y_v;
        }

        std::cout << "x:" << x << ", y: " << y << ", x_v: " << boids->x_velocity[boid] << ", y_v: " << boids->y_velocity[boid] << std::endl;
    }
    return boids;
}

void launch_kernel(float3 *pos, unsigned int mesh_width, unsigned int mesh_height, float time, Boid *boids)
{

    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    // Boid *boids = init_boids();
    const int num_threads = std::min(1024, num_boids);
    const int num_blocks = std::ceil(num_boids / num_threads);
    // update_boids_position<<<num_blocks, num_threads>>>(boids, mesh_width, mesh_height);
    // simple_vbo_kernel<<<grid, block>>>(pos, mesh_width, mesh_height, time);
    draw_boids<<<num_blocks, num_threads>>>(pos, boids, mesh_width, mesh_height, time, num_boids);

}