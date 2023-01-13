#include "draw_boids_cpu.h"

void draw_boids_cpu(float3 *pos, Boid *boids)
{
    for (int gid = 0; gid < num_boids; ++gid)
    {
        if (gid >= num_boids)
        {
            return;
        }
        float x = boids->coord[gid].x;
        float y = boids->coord[gid].y;
        float z = boids->coord[gid].z;

        float x_v = boids->velocity[gid].x;
        float y_v = boids->velocity[gid].y;
        float z_v = boids->velocity[gid].z;

        float x_p1 = (x + x_v * size_boid) / (float)mesh_width * 2;
        float y_p1 = (y + y_v * size_boid) / (float)mesh_height * 2;
        float z_p1 = (z + z_v * size_boid) / (float)mesh_height * 2;

        float x_p2 = (x - y_v * size_boid / 2) / (float)mesh_width * 2;
        float y_p2 = (y + x_v * size_boid / 2) / (float)mesh_height * 2;
        float z_p2 = (z + z_v * size_boid / 2) / (float)mesh_height * 2;

        float x_p3 = (x + y_v * size_boid / 2) / (float)mesh_width * 2;
        float y_p3 = (y - x_v * size_boid / 2) / (float)mesh_height * 2;
        float z_p3 = (z + z_v * size_boid / 2) / (float)mesh_height * 2;

        pos[gid * 3] = make_float3((float)x_p1 - 1, (float)y_p1 - 1, (is_3d != 0) ? (float)z_p1 - 1 : 1.2);
        pos[gid * 3 + 1] = make_float3((float)x_p2 - 1, (float)y_p2 - 1, (is_3d != 0) ? (float)z_p2 - 1 : 1.2);
        pos[gid * 3 + 2] = make_float3((float)x_p3 - 1, (float)y_p3 - 1, (is_3d != 0) ? (float)z_p3 - 1 : 1.2);
    }
}