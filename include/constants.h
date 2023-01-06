#pragma once


typedef double MyType;

constexpr int num_boids = 4300;

const int window_width = 512 * 3;
const int window_height = 512 * 3;

const int velocity_num_degres= 100;
const MyType velocity = 30.0;

const int base_dim=1024*8;
const int box_size = base_dim;
const int mesh_width =  base_dim;
const int mesh_height =  base_dim;

const MyType size_boid=20*5/velocity;

const MyType interaction_radius_2= 60*60;

const MyType factor_separation = 1.0;
const MyType factor_alignment = 1.0;
const MyType factor_cohesion = 1.0;
