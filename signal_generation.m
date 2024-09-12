% signal_generation.m
% Signal generation and Moving Target simulation
for i = 1:length(t)
    r_t(i) = range + (vel*t(i));
    td(i) = (2 * r_t(i)) / c;

    Tx(i) = cos(2*pi*(fc*t(i) + (slope*t(i)^2)/2));
    Rx(i) = cos(2*pi*(fc*(t(i) -td(i)) + (slope * (t(i)-td(i))^2)/2));

    Mix(i) = Tx(i) .* Rx(i);
end