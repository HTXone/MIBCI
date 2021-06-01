import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.font_manager import FontProperties

#chinese_font = FontProperties(fname='A.ttc')
sampling_rate = 64
t = np.arange(0, 1.0, 1.0 / sampling_rate)
f1 = 100
f2 = 200
f3 = 300
data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                    [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
                     lambda t: np.sin(2 * np.pi * f3 * t)])
wavename = 'cgau8'
totalscal = 90
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
print(scales)
scales_ = range(1,65)
[cwtmatr, frequencies] = pywt.cwt(data, scales_, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel("时间(秒)")
plt.title("300Hz和200Hz和100Hz的分段波形和时频谱", fontsize=20)
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel("频率(Hz)")
plt.xlabel("时间(秒)")
plt.subplots_adjust(hspace=0.4)
print(frequencies.shape,cwtmatr.shape,data.shape)

plt.show()