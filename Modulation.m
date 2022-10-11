%% HAMED AJORLOU 97101167
%% Generating arbitrary b[n]
b=[1 0 0 1 0 1 1 0 0 0 1 1];
% using our written function to devide the code
[b1,b2]=Devide(b)
%% sinosuidal shaping shaping
%building shape of bits,given the coresponding shape of 1&0
yu= linspace(0,0.1,250);
shape1 = sin(500*yu) ;
shape2 = (-1)*sin(500*yu);
x1=Pulseshaping(shape1,shape2,b1)
%% sinosuidal shaping
yu= linspace(0,0.1,250);
shape1 = sin(500*yu) ;
shape2 = (-1)*sin(500*yu);
x2=Pulseshaping(shape1,shape2,b2)
%% rectangular pulse shaping
shape1 = linspace(1,1,250);
shape2 = linspace(-1,-1,250);
x1=Pulseshaping(shape1,shape2,b1)
%% rectangular pulse shaping
shape1 = linspace(1,1,250);
shape2 = linspace(-1,-1,250);
x2=Pulseshaping(shape1,shape2,b2)
%% Anolog Modulation
x_c = AnalogMod1(x1,x2,10000,1000000);
L=length(x_c);
Fs=1000000
u=fft(x_c);
P2 = abs(u/L);
P1 = P2(1:L/2+1);
f = Fs*(0:(L/2))/L;
figure()
plot(f,P1)
title('spectrum of Xc(t)')
%% Channel effect
out = Channel(x_c,1000000,10000,15000)
fout=fft(out)
Q2 = abs(fout/L);
Q1 = Q2(1:L/2+1);
f = Fs*(0:(L/2))/L;
plot(f,Q1)
title('output of channel')
%% Analog Demodulation
[i,j]=AnalogDemod(out,1000000,10000,1000)
fs=1000000;
Ti = (0:(1500-1))*(1/fs);
ip=2000000*i
plot(Ti,ip)
title("demodulated signal")
%%
for i=1:10000
if abs(ip(1,i))>0.5
    ip(1,i)=1;
else
    ip(1,i)=0;
end
end
ip
    

%% Match filter
w1 = linspace(1,1,10000)
w0 = linspace(-1,-1,10000)
[estimate,L0,L1]=Matchedfilter(w1,w0,i)
%% Error Probability

sigma = 0:0.1:10;
sig = 10.^(sigma/10);
BER = 1/2.*erfc(sqrt(sig));
semilogy(sigma,BER)
grid on
ylabel('BER')
xlabel('E_b/N_0 (dB)')
title('Bit Error Rate for Binary Phase-Shift Keying')
%%
sigma = 0:0.1:20;
sig = 10.^(sigma/10);
BER = 1/2.*erfc(sqrt(1./(2*sig)));
semilogy(sigma,BER)
grid on
ylabel('BER')
xlabel('sigma')
title('Bit Error Rate for Binary Phase-Shift Keying')
%%
Ba=[];
Bb=[];
Ja=[];
Jb=[];
for i=1:6
    SNR = [0.00001 0.000005 0.0000001 1  0.002 0.012323]
    rxSig = awgn(out,SNR(i),'measured')
    [ii,jj]=AnalogDemod(rxSig,1000000,10000,1000)
    w1 = linspace(1,1,10000);
    w0 = linspace(-1,-1,10000);
    [estimate,L00,L11]=Matchedfilter(w1,w0,ii)
    Ba = [Ba L00];
    Bb = [Bb L11];
    [estimate,L000,L111]=Matchedfilter(w1,w0,jj)
    Ja = [Ja L00];
    Jb = [Jb L11];
end
%%
plus = [Ba Ja]
minus = [Bb Jb]
scatter(plus,minus)
grid on

%%
B1=[];
B2=[];
ooo = [60 70 110 200]
n = sourceGenerator(ooo)
for j = 1 :length(ooo)
    [b1,b2]=Devide(n(j,:))
    B1 = [B1;b1]
    B2 = [B2;b2]
end
%% 
X1=[];
X2=[];
shape1 = linspace(1,1,10000);
shape2 = linspace(-1,-1,10000);
for j=1:length(ooo)
    x1=Pulseshaping(shape1,shape2,B1(j,:))
    X1 = [X1;x1]
end
%%
figure();
t = linspace(0,0.01*length(b1),10000*length(b1))
subplot(2,2,1)
plot(t,X1(1,:))
subplot(2,2,2)
plot(t,X1(2,:))
subplot(2,2,3)
plot(t,X1(3,:))
subplot(2,2,4)
plot(t,X1(4,:))
%% 
shape1 = linspace(1,1,10000);
shape2 = linspace(-1,-1,10000);
for j=1:length(ooo)
    x2=Pulseshaping(shape1,shape2,B2(j,:))
    X2 = [X2;x2]
end
%%
figure();
t = linspace(0,0.01*length(b1),10000*length(b1))
subplot(2,2,1)
plot(t,X2(1,:))
subplot(2,2,2)
plot(t,X2(2,:))
subplot(2,2,3)
plot(t,X2(3,:))
subplot(2,2,4)
plot(t,X2(4,:))
%% 
X_c=[]
for k = 1:length(b1)
   X_c(k,:) = AnalogMod1(X1(k,:),X2(k,:),10000,1000000)
end
%%
fs=1000000;
T = (0:(40000-1))*(1/fs);
figure()
subplot(2,2,1)
plot(T,X_c(1,:))
subplot(2,2,2)
plot(T,X_c(2,:))
subplot(2,2,3)
plot(T,X_c(3,:))
subplot(2,2,4)
plot(T,X_c(4,:))

%%
L=length(X_c);
u=fft(X_c(1,:));
P22 = abs(u/L);
P12 = P22(1:L/2+1);
f = Fs*(0:(L/2))/L;
figure()
plot(f,P12)
%%
E=[];
for k=1:length(b1)
L=length(X_c);
Fs=1000000;
ui(k,:)=fft(X_c(k,:));
P2 = abs(ui(k,:)/L);
P1 = P2(1:L/2+1);
E = [E;P1];
end
%%
f = Fs*(0:(L/2))/L;
figure()
subplot(2,2,1)
plot(f,E(1,:))
subplot(2,2,2)
plot(f,E(2,:))
subplot(2,2,3)
plot(f,E(3,:))
subplot(2,2,4)
plot(f,E(4,:))

%% 
for k = 1:length(b1)
   OUt(k,:) = Channel(X_c(k,:),1000000,10000,1000)
end
%%
F=[];
for k=1:length(b1)
L=length(OUt);
Fs=1000000;
Fout(k,:)=fft(OUt(k,:));
Q2 = abs(Fout(k,:)/L);
Q1 = Q2(1:L/2+1);
F = [F;Q1];
end
%%
f = Fs*(0:(L/2))/L;
figure();
subplot(2,2,1)
plot(f,F(1,:))
subplot(2,2,2)
plot(f,F(2,:))
subplot(2,2,3)
plot(f,F(3,:))
subplot(2,2,4)
plot(f,F(4,:))
%%
for rr=1:length(b1)
    [iii(rr,:),jjj(rr,:)]=AnalogDemod(OUt(rr,:),1000000,10000,1000);
end
%%
figure()
fs=1000000;
Ti = (0:(40000-1))*(1/fs);
subplot(2,2,1)
plot(Ti,2*10^6*iii(1,:))
subplot(2,2,2)
plot(Ti,2*10^6*iii(2,:))
subplot(2,2,3)
plot(Ti,2*10^6*iii(3,:))
subplot(2,2,4)
plot(Ti,2*10^6*iii(4,:))
%%

sd=outputGenerator(iii(1))
%%
k=1;
rty=zeros(1,80);
iopo = 255*rand(1,10);
ipoo=floor(iopo);
uio=sourceGenerator(ipoo)
for i=1:8:72
    rty(1,i:i+7)=uio(k,:);
    k=k+1;
end


%%
M = 16; % Alphabet size, 16-QAM
x = randi([0 M-1],5000,1);

cpts = qammod(0:M-1,M);
constDiag = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-2 2],'YLimits',[-2 2]);
y = qammod(x,M);
ynoisy = awgn(y,5,'measured');
z = qamdemod(ynoisy,M);
[num,rt] = symerr(x,z)
constDiag(ynoisy)







%% Functions

function kl = sourceGenerator(inp)
    kl=de2bi(inp,8)
end
function jk = outputGenerator(ju)
    jk=bi2de(ju)
end

function [outpuT,P0,P1] = Matchedfilter(w1,w0,out)
     k1 = conv(out,w1);
     k0 = conv(out,w0);
     t1 = linspace(0,0.02,19999);
     figure();
     plot(t1,k0)
     title(" output of 0 matched filter");
     figure();
     plot(t1,k1)
     title(" output of 1 matched filter"); 
     if (k1(10000)>k0(10000))
         outpuT = 1 ;
     else 
         outpuT = 0 ;
     end    
     P0=k0(10000);
     P1=k1(10000);
end
function Output = Channel(Signal, fs, fc, BW)
    FFT = fft(Signal) / fs ;
    N = length(FFT) ;
    fc_N = floor(fc/fs*N) 
    BW = floor((BW/2) / fs * N)  
    fil = zeros(1, N) ;
    fil(fc_N-BW:1:fc_N+BW)=ones(1,length(fc_N-BW:1:fc_N+BW)) ;
    fil( N - fc_N - BW:1:N-fc_N+BW)=ones(1,length(fc_N-BW:1:fc_N+BW)) ;
    FFT_C = FFT .* fil ;
    Output = ifft(FFT_C) ;
end

function [b1,b2]=Devide(a)
    b1=[];
    b2=b1;
    for i=1:length(a)
        b1=[b1 a(1)];
        b2=[b2 a(2)];
        if (length(a)>2)
            a(1)=[];
            a(1)=[];
        else
            break
        end
    end
end
function a = Combine(b1,b2)
    a=[]
    for i=1:length(b1)
        a = [a b1(1)];
        a = [a b2(1)];
        if (length(b1)>1)
            b1(1)=[];
            b2(1)=[];
        else
            break
        end
    end
end
function B=Pulseshaping(a1,a2,A)
    B=[];
    t = linspace(0,0.01*length(A),250*length(A))
    for i=1:length(A)
        if A(1) == 1
            B = [B a1];
        else
            B = [B a2];
        end
        if(~isempty(A))
            A(1)=[];
        else
            break
        end
    
    end
plot(t,B,'b')
grid on 
xlabel('t')
ylabel('v')
legend('output wave')
end
function x_c = AnalogMod(a1,a2,Fc,fs)
    T = (0:(10000-1))*(1/fs);
    A1 = a1.*cos(2*pi*Fc*T);
    A2 = a2.*sin(2*pi*Fc*T);
    x_c=A1+A2;
    plot(T,x_c)
end

function [y1,y2] = AnalogDemod(Signal_C, fs, fc, BW)
    N = length(Signal_C) ;
    BW = floor((BW/2) / fs * N) ;
    Mu1 = cos(2 * pi*fc*[0:1/fs:(length(Signal_C) - 1)/fs]) ;
    Mu2 = sin(2 * pi*fc*[0:1/fs:(length(Signal_C) - 1)/fs]) ;
    Multo1 = Signal_C .* Mu1 ;
    Multo2 = Signal_C .* Mu2 ;
    Multfo1 = fft(Multo1) ;
    Multfo2 = fft(Multo2) ;
    fil = zeros(1, N) ;
    fil(1:1:BW )=ones(1,length(1:1:BW));
    fil(N-BW:1:N)=ones(1,length(N-BW:1:N)) ;
    Y1_FFT = Multfo1.* fil ;
    Y2_FFT = Multfo2 .* fil ;
    y1 = ifft(Y1_FFT) ;
    y2 = ifft(Y2_FFT) ;
end
function x_c = AnalogMod1(a1,a2,Fc,fs)
    T = (0:(1500)-1)*(1/fs);
    A1 = a1.*cos(2*pi*Fc*T);
    A2 = a2.*sin(2*pi*Fc*T);
    x_c=A1+A2;
    plot(T,x_c)
    title('Xc(t)')
end