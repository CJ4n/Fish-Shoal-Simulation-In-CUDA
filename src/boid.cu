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
        boids->coord[boid] = make_float2(x, y);

        float x_v = rand() % velocity_num_degres;
        float y_v = rand() % velocity_num_degres;
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
        boids->velocity[boid] = make_float2(x_v, y_v);
        // std::cout << "x:" << x << ", y: " << y << ", x_v: " << boids->velocity[boid].x << ", y_v: " << boids->velocity[boid].y << std::endl;
    }
    return boids;
}

__global__ void draw_boids(float3 *pos, Boid *boids, int num_boids)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_boids)
    {
        return;
    }
    float x = boids->coord[gid].x;
    float y = boids->coord[gid].y;

    float x_v = boids->velocity[gid].x;
    float y_v = boids->velocity[gid].y;

    float x_p1 = (x + x_v * size_boid) / (float)mesh_width;
    float y_p1 = (y + y_v * size_boid) / (float)mesh_height;

    float x_p2 = (x - y_v * size_boid / 2) / (float)mesh_width;
    float y_p2 = (y + x_v * size_boid / 2) / (float)mesh_height;

    float x_p3 = (x + y_v * size_boid / 2) / (float)mesh_width;
    float y_p3 = (y - x_v * size_boid / 2) / (float)mesh_height;

    pos[gid * 3] = make_float3((float)x_p1, (float)y_p1, 1.2);
    pos[gid * 3 + 1] = make_float3((float)x_p2, (float)y_p2, 1.2);
    pos[gid * 3 + 2] = make_float3((float)x_p3, (float)y_p3, 1.2);
}

__device__ float dist(float x_1, float y_1, float x_2, float y_2)
{
    return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void update_boids_position(Boid *boids, float interaction_radius_2, float factor_separation,
                                      float factor_alignment, float factor_cohesion, float factor_intertia)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (num_boids <= gid)
    {
        return;
    }

    DataType pos = boids->coord[gid];
    DataType vel = boids->velocity[gid];

    DataType pos_average = make_float2(0, 0);
    DataType vel_average = make_float2(0, 0);
    int count = 0;
    DataType new_velocity = vel;

    for (int idx = 0; idx < num_boids; ++idx)
    {
        DataType neighbour = boids->coord[idx];
        // if (dist(x, y, x_neighbour, y_neighbour) > interaction_radius_2)
        float d = (pos.x - neighbour.x) * (pos.x - neighbour.x) + (pos.y - neighbour.y) * (pos.y - neighbour.y);
        if (d > interaction_radius_2)
        {
            continue;
        }
        if (idx == gid)
        {
            continue;
        }
        pos_average.x += neighbour.x;
        pos_average.y += neighbour.y;
        vel_average.x += boids->velocity[idx].x;
        vel_average.y += boids->velocity[idx].y;
        count++;
    }

    if (count > 0)
    {
        vel_average.x /= (float)count;
        vel_average.y /= (float)count;
        pos_average.x /= (float)count;
        pos_average.y /= (float)count;

        float sum_v_acumualte = std::abs(vel_average.x) + std::abs(vel_average.y);
        float sum_acumulate = std::abs(pos_average.x) + std::abs(pos_average.y);
        if (sum_v_acumualte > 0)
        {
            vel_average.x /= sum_v_acumualte;
            vel_average.y /= sum_v_acumualte;
        }
        DataType sep, coh, ali;

        // calculate separation force
        sep = make_float2(pos.x - pos_average.x, pos.y - pos_average.y);
        float sum_sep = std::abs(sep.x) + std::abs(sep.y);
        if (sum_sep > 0)
        {
            sep.x /= sum_sep;
            sep.y /= sum_sep;
        }
        DataType separation_force = make_float2(factor_separation * sep.x, factor_separation * sep.y);
        // calculate separation force

        // calculate alignment force
        ali = make_float2(vel_average.x - vel.x, vel_average.y - vel.y);
        float sum_ali = std::abs(ali.x) + std::abs(ali.y);
        if (sum_ali > 0)
        {
            ali.x /= sum_ali;
            ali.y /= sum_ali;
        }
        DataType alignment_force = make_float2(factor_alignment * ali.x, factor_alignment * ali.y);
        // calculate alignment force

        // calculate cohision force
        coh = make_float2(pos_average.x - pos.x, pos_average.y - pos.y);
        float sum_coh = std::abs(coh.x) + std::abs(coh.y);
        if (sum_coh > 0)
        {
            coh.x /= sum_coh;
            coh.y /= sum_coh;
        }
        DataType cohesion_force = make_float2(factor_cohesion * coh.x, factor_cohesion * coh.y);
        // calculate cohision force
        new_velocity = make_float2(factor_intertia * vel.x + separation_force.x + cohesion_force.x + alignment_force.x,
                                   factor_intertia * vel.y + separation_force.y + cohesion_force.y + alignment_force.y);

        float sum_vel = std::abs(new_velocity.x) + std::abs(new_velocity.y);
        if (sum_vel > 0)
        {
            new_velocity.x /= sum_vel;
            new_velocity.y /= sum_vel;
        }
    }

    DataType new_pos = make_float2(pos.x + new_velocity.x * velocity, pos.y + new_velocity.y * velocity);

    if (new_pos.x >= box_size || new_pos.x <= 0)
    {
        new_velocity.x = -new_velocity.x;
    }

    if (new_pos.y >= box_size || new_pos.y <= 0)
    {
        new_velocity.y = -new_velocity.y;
    }
    boids->velocity[gid] = new_velocity;
    boids->coord[gid] = new_pos;
}
