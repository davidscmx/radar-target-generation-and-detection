% range_measurement.m
% RANGE MEASUREMENT
Mix = reshape(Mix, [Nr, Nd]);
signal_fft = fft(Mix, Nr);
signal_fft = abs(signal_fft);
signal_fft = signal_fft ./ max(signal_fft);
signal_fft = signal_fft(1 : Nr/2-1);

% Plotting
figure('Name', 'Range from First FFT');
plot(signal_fft);
axis([0 200 0 1]);
title('Range from First FFT');
ylabel('Amplitude (Normalized)');
xlabel('Range [m]');


% Save the figure
saveas(gcf, './images/range_1st_fft.jpg');