#pragma once

typedef float3 DataType;
static constexpr int is_3d = 0;

static constexpr int num_boids = 10000;

static constexpr int window_width = 512 * 3;
static constexpr int window_height = 512 * 3;

static constexpr int velocity_num_degres = 50;
static constexpr int change_velocity_step = 1;

static constexpr int base_dim = 1024 * 32;
static constexpr int box_size = base_dim;
static constexpr int mesh_width = base_dim;
static constexpr int mesh_height = base_dim;

static constexpr float size_boid = 20 * 5*2;

static constexpr float change_force_factor_step = 0.05;
static constexpr float chanage_radious_step = 10;

constexpr bool advanced_interaction_scope=false;

extern float velocity;
extern float radius;
extern float interaction_radius_2;
extern float factor_separation;
extern float factor_alignment;
extern float factor_cohesion;
extern float factor_intertia;
extern bool animate;

extern bool gpu_render;

