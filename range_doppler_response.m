% range_doppler_response.m
% RANGE DOPPLER RESPONSE
Mix = reshape(Mix, [Nr, Nd]);
signal_fft2 = fft2(Mix, Nr, Nd);
signal_fft2 = signal_fft2(1:Nr/2, 1:Nd);
signal_fft2 = fftshift(signal_fft2);

RDM = abs(signal_fft2);
RDM = 10*log10(RDM);

% Plotting
doppler_axis = linspace(-100, 100, Nd);
range_axis = linspace(-200, 200, Nr/2)*((Nr/2)/400);

figure;
surf(doppler_axis, range_axis, RDM);
title('Amplitude and Range From FFT2');
xlabel('Speed');
ylabel('Range');
zlabel('Amplitude');

saveas(gcf, './images/range_doppler_map.jpg');