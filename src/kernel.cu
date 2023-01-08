// // #include <stdlib.h>
// #include <algorithm>
// #include "kernel.h"
// // #include <stdio.h>





// void launch_kernel(float3 *pos, Boid *boids)
// {
//     const int num_threads = std::min(1024, num_boids);
//     const int num_blocks = std::ceil((float)num_boids / (float)num_threads);
//     update_boids_position<<<num_blocks, num_threads>>>(boids, interaction_radius_2, factor_separation, factor_alignment, factor_cohesion, factor_intertia);
//     draw_boids<<<num_blocks, num_threads>>>(pos, boids, num_boids);
//     // for (int boid = 0; boid <num_boids; ++boid)
//     // {
//     //     std::cout << "x:" << boids->coord[boid].x << ", y: " << boids->coord[boid].y << ", x_v: " << boids->velocity[boid].x << ", y_v: " << boids->velocity[boid].y << std::endl;
//     // }
// }