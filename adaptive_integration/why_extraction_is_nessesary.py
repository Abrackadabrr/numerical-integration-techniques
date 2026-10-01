import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from base_routines import SegmentRule, AdaptiveIntegration, GQ1, GQ2, GQ3, GQ5, GQ9


def one_on_r(x):
	return 1. / x
	
def ln(x):
	return np.log(x)
	
def one_on_r2(x):
	return 1. / x**2

def minus_one_on_r(x):
	return -one_on_r(x)
	
# Задаем функцию и её первообразоную
f = one_on_r2
int_f = minus_one_on_r
# Точности интегрирования
rtol = 1e-5
atol = 1e-14
# Максимальная глубина
max_level = 30

# Расчет интегралов и ошибок 
min_a = 1e-8
a_s = np.logspace(0, 0.2, 30, base=1.05)[::-1] - 1 + min_a
errors = []
true_values = []
levels = []

integrator = AdaptiveIntegration(GQ5(), rtol, atol)

for a in a_s:
	b = a + 1
	res, level, log = integrator.integrate(a, b, f, max_level)
	an_res = int_f(b) - int_f(a)
	true_values.append(an_res)
	errors.append(np.abs(an_res - res))
	levels.append(level)
	
fig, ax = plt.subplots(nrows=2)
ax[0].plot(np.log10(a_s), levels, label="глубина разбиения")
ax[0].grid()
ax[0].legend()
ax[1].plot(np.log10(a_s), np.log10(errors), label="относительная ошибка с аналитикой")
ax[1].grid()
ax[1].legend()
fig.savefig("one_over_r.png", dpi=300)
plt.show()
print(errors)
print(true_values)

