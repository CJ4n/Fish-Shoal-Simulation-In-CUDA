#include <stdlib.h>
#include <iostream>
#include "kernel.h"
#include "constants.h"

__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

   

    // write output vertex
    pos[y * width + x] = make_float4(0, 0, 0, 1.0f);
}

__global__ void draw_boids(float4 *pos, Boid *boids, unsigned int width, unsigned int height, float time, int num_boids)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_boids)
    {
        return;
    }
    float x = boids->x_coords[gid];
    float y = boids->y_coords[gid];
    // write output vertex
    if (x >= width)
        return;
    if (y >= width)
        return;
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    // float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    pos[(int)y * width + (int)x] = make_float4(u, 0, v, 1.0f);
}

__global__ void update_boids_position(Boid *boid, unsigned int width, unsigned int height)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    boid->x_coords[gid] += 1;
    boid->y_coords[gid] += 1;
}

Boid *init_boids()
{
    srand(0);
    Boid *boids;
    cudaMallocManaged(&boids, sizeof(Boid));

    cudaMallocManaged(&(boids->x_coords), sizeof(*(boids->x_coords)) * num_boids);
    cudaMallocManaged(&(boids->y_coords), sizeof(*(boids->y_coords)) * num_boids);

    for (int boid = 0; boid < num_boids; ++boid)
    {
        int x = rand() % mesh_width;
        int y = rand() % mesh_height;
        boids->x_coords[boid] = x;
        boids->y_coords[boid] = y;

        std::cout << "x:" << x << ", y: " << y << std::endl;
    }
    return boids;
}

void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time, Boid *boids)
{

    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    // Boid *boids = init_boids();
    simple_vbo_kernel<<<grid, block>>>(pos, mesh_width, mesh_height, time);
    const int num_threads = 1024;
    const int num_blocks = std::ceil(num_boids / num_threads);
    update_boids_position<<<num_blocks, num_threads>>>(boids, mesh_width, mesh_height);
    draw_boids<<<num_blocks, num_threads>>>(pos, boids, mesh_width, mesh_height, time, num_boids);
}