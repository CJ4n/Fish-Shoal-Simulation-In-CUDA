#include "update_boids_gpu.h"
#include <math.h>

__device__ float interaction_scope_gpu(float3 p1_pos, float3 p2_pos, float3 p1_v)
{
    float x1 = p1_pos.x + p1_v.x, y1 = p1_pos.y + p1_v.y;
    float x2 = p2_pos.x - p1_pos.x, y2 = p2_pos.y - p1_pos.y;

    float dot = x1 * x2 + y1 * y2;
    float det = x1 * y2 - y1 * x2;
    float angle = std::atan2(det, dot);
    float threshhold = 45;
    if (angle < threshhold / 180.0 * M_PI && advanced_interaction_scope)
    {
        return __FLT_MAX__;
    }
    else
    {
        return (p1_pos.x - p2_pos.x) * (p1_pos.x - p2_pos.x) + (p1_pos.y - p2_pos.y) * (p1_pos.y - p2_pos.y);
    }
}

__global__ void update_boids_position(Boid *boids, float interaction_radius_2, float velocity, float factor_separation,
                                      float factor_alignment, float factor_cohesion, float factor_intertia)
{
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (num_boids <= gid)
    {
        return;
    }

    DataType pos = boids->coord[gid];
    DataType vel = boids->velocity[gid];

    DataType pos_average = make_float3(0, 0, 0);
    DataType vel_average = make_float3(0, 0, 0);
    int count = 0;
    DataType new_velocity = vel;

    for (int idx = 0; idx < num_boids; ++idx)
    {
        DataType neighbour = boids->coord[idx];
        float d = (pos.x - neighbour.x) * (pos.x - neighbour.x) + (pos.y - neighbour.y) * (pos.y - neighbour.y) + (pos.z - neighbour.z) * (pos.z - neighbour.z);
        // if (d > interaction_radius_2)
        if (interaction_scope_gpu(pos,neighbour,vel) > interaction_radius_2)
        {
            continue;
        }
        if (idx == gid)
        {
            continue;
        }
        pos_average.x += neighbour.x;
        pos_average.y += neighbour.y;
        if (is_3d)
        {
            pos_average.z += neighbour.z;
        }
        vel_average.x += boids->velocity[idx].x;
        vel_average.y += boids->velocity[idx].y;
        if (is_3d)
        {
            vel_average.z += boids->velocity[idx].z;
        }
        count++;
    }

    if (count > 0)
    {
        vel_average.x /= (float)count;
        vel_average.y /= (float)count;
        // if (is_3d)
        vel_average.z /= (float)count;
        pos_average.x /= (float)count;
        pos_average.y /= (float)count;
        // if (is_3d)
        pos_average.z /= (float)count;

        float sum_v_acumualte = std::abs(vel_average.x) + std::abs(vel_average.y) + std::abs(vel_average.z);
        float sum_acumulate = std::abs(pos_average.x) + std::abs(pos_average.y) + std::abs(pos_average.z);
        if (sum_v_acumualte > 0)
        {
            vel_average.x /= sum_v_acumualte;
            vel_average.y /= sum_v_acumualte;
            vel_average.z /= sum_v_acumualte;
        }
        DataType sep, coh, ali;

        // calculate separation force
        sep = make_float3(pos.x - pos_average.x, pos.y - pos_average.y, pos.z - pos_average.z);
        float sum_sep = std::abs(sep.x) + std::abs(sep.y) + std::abs(sep.z);
        if (sum_sep > 0)
        {
            sep.x /= sum_sep;
            sep.y /= sum_sep;
            sep.z /= sum_sep;
        }
        DataType separation_force = make_float3(factor_separation * sep.x, factor_separation * sep.y, factor_separation * sep.z);
        // calculate separation force

        // calculate alignment force
        ali = make_float3(vel_average.x - vel.x, vel_average.y - vel.y, vel_average.z - vel.z);
        float sum_ali = std::abs(ali.x) + std::abs(ali.y) + std::abs(ali.z);
        if (sum_ali > 0)
        {
            ali.x /= sum_ali;
            ali.y /= sum_ali;
            ali.z /= sum_ali;
        }
        DataType alignment_force = make_float3(factor_alignment * ali.x, factor_alignment * ali.y, factor_alignment * ali.z);
        // calculate alignment force

        // calculate cohision force
        coh = make_float3(pos_average.x - pos.x, pos_average.y - pos.y, pos_average.z - pos.z);
        float sum_coh = std::abs(coh.x) + std::abs(coh.y) + +std::abs(coh.z);
        if (sum_coh > 0)
        {
            coh.x /= sum_coh;
            coh.y /= sum_coh;
            coh.z /= sum_coh;
        }
        DataType cohesion_force = make_float3(factor_cohesion * coh.x, factor_cohesion * coh.y, factor_cohesion * coh.z);
        // calculate cohision force
        new_velocity = make_float3(factor_intertia * vel.x + separation_force.x + cohesion_force.x + alignment_force.x,
                                   factor_intertia * vel.y + separation_force.y + cohesion_force.y + alignment_force.y,
                                   factor_intertia * vel.z + separation_force.z + cohesion_force.z + alignment_force.z);

        float sum_vel = std::abs(new_velocity.x) + std::abs(new_velocity.y) + std::abs(new_velocity.z);
        if (sum_vel > 0)
        {
            new_velocity.x /= sum_vel;
            new_velocity.y /= sum_vel;
            new_velocity.z /= sum_vel;
        }
    }

    DataType new_pos = make_float3(pos.x + new_velocity.x * velocity, pos.y + new_velocity.y * velocity, pos.z + new_velocity.z * velocity);

    if (new_pos.x >= box_size || new_pos.x <= 0)
    {
        if (new_pos.x >= box_size)
        {
            new_pos.x = box_size;
        }
        else
        {
            new_pos.x = 0;
        }
        new_velocity.x = -new_velocity.x;
    }

    if (new_pos.y >= box_size || new_pos.y <= 0)
    {
        if (new_pos.y >= box_size)
        {
            new_pos.y = box_size;
        }
        else
        {
            new_pos.y = 0;
        }
        new_velocity.y = -new_velocity.y;
    }

    if (new_pos.z >= box_size || new_pos.z <= 0)
    {
        if (new_pos.z >= box_size)
        {
            new_pos.z = box_size;
        }
        else
        {
            new_pos.z = 0;
        }
        new_velocity.z = -new_velocity.z;
    }
    boids->velocity[gid] = new_velocity;
    boids->coord[gid] = new_pos;
}
