#include <stdlib.h>
#include <iostream>
#include "kernel.h"
#include "constants.h"
#include <stdio.h>

// __global__ void simple_vbo_kernel(float3 *pos, int width, int height, float time)
// {
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

//     // write output vertex
//     pos[y * width * 3 + x * 3] = make_float3(0, 0, 0);
//     pos[y * width * 3 + x * 3 + 1] = make_float3(0, 0, 0);
//     pos[y * width * 3 + x * 3 + 2] = make_float3(0, 0, 0);
// }

__global__ void draw_boids(float3 *pos, Boid *boids, int num_boids)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_boids)
    {
        return;
    }
    MyType x = boids->x_coords[gid];
    MyType y = boids->y_coords[gid];

    MyType x_v = boids->x_velocity[gid] * velocity;
    MyType y_v = boids->y_velocity[gid] * velocity;

    MyType x_p1 = (x + x_v * size_boid) / (MyType)mesh_width;
    MyType y_p1 = (y + y_v * size_boid) / (MyType)mesh_height;

    MyType x_p2 = (x - y_v * size_boid / 2) / (MyType)mesh_width;
    MyType y_p2 = (y + x_v * size_boid / 2) / (MyType)mesh_height;

    MyType x_p3 = (x + y_v * size_boid / 2) / (MyType)mesh_width;
    MyType y_p3 = (y - x_v * size_boid / 2) / (MyType)mesh_height;

    pos[gid * 3] = make_float3((float)x_p1, (float)y_p1, 1.2);
    pos[gid * 3 + 1] = make_float3((float)x_p2, (float)y_p2, 1.2);
    pos[gid * 3 + 2] = make_float3((float)x_p3, (float)y_p3, 1.2);
}

__device__ MyType dist(MyType x_1, MyType y_1, MyType x_2, MyType y_2)
{
    return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void update_boids_position(Boid *boids)
{
    // problem jak bois>4300 nan jest wtedy z ajkiegos powdu??
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("-------------\n");

    if (num_boids <= gid)
    {
        return;
    }
    MyType x = boids->x_coords[gid];
    MyType y = boids->y_coords[gid];

    MyType x_v = boids->x_velocity[gid];
    MyType y_v = boids->y_velocity[gid];

    MyType x_v_acumulate = 0;
    MyType y_v_acumulate = 0;

    MyType x_acumulate = 0;
    MyType y_acumulate = 0;

    int count = 0;

    for (int idx = 0; idx < num_boids; ++idx)
    {
        MyType x_neighbour = boids->x_coords[idx];
        MyType y_neighbour = boids->y_coords[idx];
        // z jakiegos powodu call to tej funkcji freezuje program xD
        // if (dist(x, y, x_neighbour, y_neighbour) > interaction_radius_2)
        if ((x - x_neighbour) * (x - x_neighbour) + (y - y_neighbour) * (y - y_neighbour) > interaction_radius_2)
        {
            continue;
        }
        if (idx == gid)
        {
            continue;
        }
        x_acumulate += x_neighbour;
        y_acumulate += y_neighbour;
        x_v_acumulate += boids->x_velocity[idx];
        y_v_acumulate += boids->y_velocity[idx];
        count++;
        // overflow typu??
    }
    if (count > 0)
    {

        x_v_acumulate /=(MyType) count;
        y_v_acumulate /= (MyType) count;
        x_acumulate /= (MyType) count;
        y_acumulate /= (MyType) count;

        x_acumulate -= x;
        y_acumulate -= y;

        MyType sum_v_acumualte = std::abs(x_v_acumulate) + std::abs(y_v_acumulate);
        MyType sum_acumulate = std::abs(x_acumulate) + std::abs(y_acumulate);
        if (sum_v_acumualte > 0.1)
        {
            x_v_acumulate /= sum_v_acumualte;
            y_v_acumulate /= sum_v_acumualte;
        }
        if (sum_acumulate > 0.1)
        {
            x_acumulate /= sum_acumulate;
            y_acumulate /= sum_acumulate;
        }
        // printf("%d,%d\n", sum_v_acumualte, sum_acumulate);
        x_v = x_v ;//+ x_v_acumulate * factor_alignment + x_acumulate * factor_cohesion;
        y_v = y_v ;//+ y_v_acumulate * factor_alignment + y_acumulate * factor_cohesion;
        MyType sum = std::abs(x_v) + std::abs(y_v);
        if (sum > 0.1)
        {
            x_v /= sum;
            y_v /= sum;
        }
    }
    MyType new_x = x + x_v;
    MyType new_y = y + y_v;

    if (new_x >= box_size || new_x <= 0)
    {
        x_v = -x_v;
    }

    if (new_y >= box_size || new_y <= 0)
    {
        y_v = -y_v;
    }
    boids->x_velocity[gid] = x_v;
    boids->y_velocity[gid] = y_v;

    boids->x_coords[gid] += x_v * velocity;
    boids->y_coords[gid] += y_v * velocity;
}

Boid *init_boids()
{
    srand(0);
    Boid *boids;
    cudaMallocManaged(&boids, sizeof(Boid));

    cudaMallocManaged(&(boids->x_coords), sizeof(*(boids->x_coords)) * num_boids);
    cudaMallocManaged(&(boids->y_coords), sizeof(*(boids->y_coords)) * num_boids);
    cudaMallocManaged(&(boids->x_velocity), sizeof(*(boids->x_velocity)) * num_boids);
    cudaMallocManaged(&(boids->y_velocity), sizeof(*(boids->y_velocity)) * num_boids);

    for (int boid = 0; boid < num_boids; ++boid)
    {
        MyType x = rand() % box_size;
        MyType y = rand() % box_size;
        boids->x_coords[boid] = x;
        boids->y_coords[boid] = y;

        MyType x_v = rand() % velocity_num_degres;
        MyType y_v = rand() % velocity_num_degres;
        MyType sum_v = x_v + y_v;

        x_v = x_v / sum_v;
        y_v = y_v / sum_v;

        if (rand() % 2 == 0)
        {
            boids->x_velocity[boid] = x_v;
        }
        else
        {
            boids->x_velocity[boid] = -x_v;
        }

        if (rand() % 2 == 0)
        {
            boids->y_velocity[boid] = y_v;
        }
        else
        {
            boids->y_velocity[boid] = -y_v;
        }

        // std::cout << "x:" << x << ", y: " << y << ", x_v: " << boids->x_velocity[boid] << ", y_v: " << boids->y_velocity[boid] << std::endl;
    }
    return boids;
}

void launch_kernel(float3 *pos, Boid *boids)
{

    // execute the kernel
    // dim3 block(8, 8, 1);
    // dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    // simple_vbo_kernel<<<grid, block>>>(pos, mesh_width, mesh_height, time);
    const int num_threads = std::min(1024, num_boids);
    const int num_blocks = std::ceil((float)num_boids / (float)num_threads);
    update_boids_position<<<num_blocks, num_threads>>>(boids);
    // for (int boid = 0; boid < 1; ++boid)
    // {

    //     std::cout << "x:" << boids->x_coords[boid] << ", y: " << boids->y_coords[boid] << ", x_v: " << boids->x_velocity[boid] << ", y_v: " << boids->y_velocity[boid] << std::endl;
    // }
    draw_boids<<<num_blocks, num_threads>>>(pos, boids, num_boids);
}