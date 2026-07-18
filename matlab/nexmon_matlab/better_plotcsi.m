function [] = better_plotcsi(csi, nfft, normalize)

csi_buff = fftshift(csi,2);

amp = abs(csi_buff);
phase = rad2deg(angle(csi_buff));

if normalize
    amp = amp ./ max(amp,[],2);
end

x = -(nfft/2):(nfft/2-1);

figure('Position',[100 100 1200 800])

subplot(2,2,1)
plot(x,mean(amp,1),'LineWidth',2)
grid on
xlabel('Subcarrier')
ylabel('Mean Magnitude')
title('Mean CSI Magnitude')

subplot(2,2,2)
plot(x,var(amp,0,1),'LineWidth',2)
grid on
xlabel('Subcarrier')
ylabel('Variance')
title('CSI Variance')

subplot(2,2,3)
imagesc(x,1:size(amp,1),amp)
set(gca,'YDir','normal')
xlabel('Subcarrier')
ylabel('Packet')
title('CSI Amplitude')
colorbar

subplot(2,2,4)
imagesc(x,1:size(phase,1),phase)
set(gca,'YDir','normal')
xlabel('Subcarrier')
ylabel('Packet')
title('CSI Phase')
colorbar

end