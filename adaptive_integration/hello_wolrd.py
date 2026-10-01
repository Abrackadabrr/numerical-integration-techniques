import numpy as np
import matplotlib.pyplot as plt
from base_routines import SegmentRule, AdaptiveIntegration

def f(x):
	return 1/np.sqrt(x)

def int_f(x):
	return 2 * np.sqrt(x)
	
rtol = 1e-6
atol = 1e-14
max_level = 100
a = 1
b = 2
middle_rect_rule = SegmentRule(np.array([0.]), np.array([2]))
integrator = AdaptiveIntegration(middle_rect_rule, rtol, atol)
res, level, log = integrator.integrate(a, b, f, max_level)
an_res = int_f(b) - int_f(a)

print("Relative error = ", np.abs(res - an_res))
print(level)

# Построение сходимости по уровням подразбиения
plt.plot(np.log(np.abs((log - log[-1])[0:-1])))
plt.plot(np.log(np.abs((log - an_res))))
plt.grid()
plt.show()


