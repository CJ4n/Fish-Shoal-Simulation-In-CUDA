#pragma once

typedef float3 DataType;

constexpr int num_boids = 50000;

constexpr int window_width = 512 * 3;
constexpr int window_height = 512 * 3;

constexpr int velocity_num_degres = 50;
constexpr int change_velocity_step = 1;

constexpr int base_dim = 1024 * 32;
constexpr int box_size = base_dim;
constexpr int mesh_width = base_dim;
constexpr int mesh_height = base_dim;

constexpr float size_boid = 20 * 5;

constexpr float change_force_factor_step =0.05;
constexpr float chanage_radious_step=10;

extern float velocity;
extern float radius;
extern float interaction_radius_2;
extern float factor_separation;
extern float factor_alignment;
extern float factor_cohesion;
extern float factor_intertia;
extern bool animate;