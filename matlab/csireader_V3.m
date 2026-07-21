clear all
%% csireader.m
%
% read and plot CSI from UDPs created using the nexmon CSI extractor (nexmon.org/csi)
% modify the configuration section to your needs
% make sure you run >mex unpack_float.c before reading values from bcm4358 or bcm4366c0 for the first time
%
% the example.pcap file contains 4(core 0-1, nss 0-1) packets captured on a bcm4358
%

%% configuration
CHIP = '43455c0';          % wifi chip (possible values 4339, 4358, 43455c0, 4366c0)
BW = 80;                % bandwidth
FILE = 'captures\csi_capture.pcap';% capture file
NPKTS_MAX = 10000;       % max number of UDPs to process

%% read file
HOFFSET = 16;           % header offset
NFFT = BW*3.2;          % fft size
p = readpcap();
p.open(FILE);
n = min(length(p.all()),NPKTS_MAX);
p.from_start();
csi_buff = complex(zeros(n,NFFT),0);
k = 1;
while (k <= n)
    f = p.next();
    if isempty(f)
        disp('no more frames');
        break;
    end
    if f.header.orig_len-(HOFFSET-1)*4 ~= NFFT*4
        disp('skipped frame with incorrect size');
        continue;
    end
    payload = f.payload;
    H = payload(HOFFSET:HOFFSET+NFFT-1);
    if (strcmp(CHIP,'4339') || strcmp(CHIP,'43455c0'))
        Hout = typecast(H, 'int16');
    elseif (strcmp(CHIP,'4358'))
        Hout = unpack_float(int32(0), int32(NFFT), H);
    elseif (strcmp(CHIP,'4366c0'))
        Hout = unpack_float(int32(1), int32(NFFT), H);
    else
        disp('invalid CHIP');
        break;
    end

    if k == 1

        fprintf('Min int16 value = %d\n', min(Hout));
        fprintf('Max int16 value = %d\n', max(Hout));

        fprintf('Number of zeros = %d\n', sum(Hout == 0));
        fprintf('Total samples   = %d\n', length(Hout));

        figure;
        histogram(Hout,100);
        title('Distribution of raw int16 CSI samples');

    end

    Hout = reshape(Hout,2,[]).';
    cmplx = double(Hout(1:NFFT,1))+1j*double(Hout(1:NFFT,2));
    csi_buff(k,:) = cmplx.';
    k = k + 1;
end

%% Connection information
payload8 = typecast(payload,'uint8');

seq      = typecast(payload8(53:54),'uint16');
core_ss  = typecast(payload8(55:56),'uint16');
chanspec = typecast(payload8(57:58),'uint16');
chipver  = typecast(payload8(59:60),'uint16');

core = bitand(core_ss,7);
nss  = bitand(bitshift(core_ss,-3),7);

fprintf('Sequence = %u\n', seq);
fprintf('Core     = %u\n', core);
fprintf('NSS      = %u\n', nss);
fprintf('Chanspec = 0x%04X\n', chanspec);
fprintf('ChipVer  = 0x%04X\n', chipver);


%% Find strongest
pkt = abs(csi_buff(100,:));
[vals,idx] = sort(pkt,'descend');
for i=1:15
    fprintf('%3d  %.1f\n',idx(i),vals(i));
end



%{
%% plot
csi = fftshift(csi_buff,2);

figure
imagesc(abs(csi))
xlabel('Subcarrier')
ylabel('Packet')
title('CSI Amplitude')
colorbar

amp = abs(csi);
variance = var(amp,0,1);

figure
plot(variance,'LineWidth',2)
grid on
xlabel('Subcarrier')
ylabel('Variance')
title('CSI Variance')
%}

%% Pre graph
better_plotcsi(csi_buff,NFFT,false);

%% Masked center and begin
csi2 = csi_buff;
csi2(:,1:2) = NaN;
csi2(:,126:132) = NaN;

better_plotcsi(csi2,NFFT,false);