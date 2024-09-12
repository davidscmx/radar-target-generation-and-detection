% waveform_generation.m
% FMCW Waveform Generation
B = c / (2*range_res);
Tchirp = 5.5 * 2 * (max_range/c);
slope = B/Tchirp;

Nd = 128;  % Number of doppler cells
Nr = 1024;  % Number of range cells

% Time vector
t = linspace(0, Nd*Tchirp, Nr*Nd);

% Initialize signal vectors
Tx = zeros(1, length(t));
Rx = zeros(1, length(t));
Mix = zeros(1, length(t));

r_t = zeros(1, length(t));
td = zeros(1, length(t));