#pragma once

typedef float MyType;

typedef float2 DataType;

constexpr int num_boids = 20000;

const int window_width = 512 * 3;
const int window_height = 512 * 3;

const int velocity_num_degres = 50;
const MyType velocity = 10;

const int base_dim = 1024 * 16;
const int box_size = base_dim;
const int mesh_width = base_dim;
const int mesh_height = base_dim;

const MyType size_boid = 20 * 5;

const MyType radius = 200;
const MyType interaction_radius_2 = radius * radius;

const MyType factor_separation = 0.51;
const MyType factor_alignment = 0.4;
const MyType factor_cohesion = 0.401;
const MyType factor_intertia = 1;
