% 定义信号信息
fs=2^6;    %采样频率
dt=1/fs;    %时间精度
timestart=-8;
timeend=8;
t=(0:(timeend-timestart)/dt-1)*dt+timestart;
L=length(t);

z=4*sin(2*pi*linspace(6,12,L).*t);
%旧版本
wavename='cmor1-3'; %可变参数，分别为cmor的
%举一个频率转尺度的例子
fmin=2;
fmax=20;
df=0.1;

f=fmin:df:fmax-df;%预期的频率
wcf=centfrq(wavename); %小波的中心频率
scal=fs*wcf./f;%利用频率转换尺度
coefs = cwt(z,scal,wavename);
figure(2)
pcolor(t,f,abs(coefs));shading interp
