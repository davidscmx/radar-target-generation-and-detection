% cfar_implementation.m
% CFAR implementation
n_train_cells = 10;
n_train_bands = 8;
n_guard_cells = 4;
n_guard_bands = 4;
offset = 1.4;

RDM = RDM / max(RDM(:));

% CFAR processing loop
for row0 = n_train_cells + n_guard_cells + 1 : (Nr/2) - (n_train_cells + n_guard_cells)
    for col0 = n_train_bands + n_guard_bands + 1 : (Nd) - (n_train_bands + n_guard_bands)
        % ... (CFAR implementation details)
    end
end

RDM(RDM~=0 & RDM~=1) = 0;

% Plotting CFAR output
figure('Name', 'CA-CFAR Filtered RDM');
surf(doppler_axis, range_axis, RDM);
title('CA-CFAR Filtered RDM surface plot');
xlabel('Speed');
ylabel('Range');
zlabel('Normalized Amplitude');
view(315, 45);

saveas(gcf, './images/cfar_filtered_rdm.jpg');
