% main.m
clear all
clc;

% Load necessary packages
pkg load control
pkg load signal

% Run the simulation modules
radar_specs;
target_definition;
waveform_generation;
signal_generation;
range_measurement;
range_doppler_response;
cfar_implementation;