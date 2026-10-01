import numpy as np

class SegmentRule:
	def __init__(self, points, weights):
		self.p = points
		self.w = weights  # sum equals 2
	
	def integrate(self, a, b, f):
		l = (b - a)
		center = (b + a) / 2
		p = center + self.p * l/2
		w = self.w * l/2
		values = f(p)
		return np.sum(values * w)


class AdaptiveIntegration:
	def __init__(self, rule : SegmentRule, rtol, atol):
		self.r = rule
		self.atol = atol
		self.rtol = rtol
	
	def conv_predicate(self, v1, v2):
		target = self.rtol * np.abs(v2) + self.atol
		return np.abs(v1 - v2) < target
	
	def integrate(self, a, b, f, max_level):
		cur_value = self.r.integrate(a, b, f)
		log = [cur_value]
		prev_value = cur_value
		leng = b - a
		for k in range(max_level):
			depth = 2**(k+1)
			cur_len = leng / depth
			cur_value = 0
			# split segment by smaller ones
			for i in range(depth):
				a_k = a + cur_len * i
				b_k = a_k + cur_len
				cur_value += self.r.integrate(a_k, b_k, f)
			log.append(cur_value)
			is_converged = self.conv_predicate(prev_value, cur_value)
			if is_converged:
				return cur_value, k, log
			prev_value = cur_value
		return cur_value, k + 1, log

# General Segment gaussian quadratures

def GQ1():
    return SegmentRule(
        np.array([0.]),
        np.array([2.])
    )


def GQ2():
    return SegmentRule(
        np.array([
            -0.5773502691896257,
             0.5773502691896257
        ]),
        np.array([
            1.,
            1.
        ])
    )


def GQ3():
    return SegmentRule(
        np.array([
            -0.7745966692414834,
             0.,
             0.7745966692414834
        ]),
        np.array([
            0.5555555555555556,
            0.8888888888888888,
            0.5555555555555556
        ])
    )


def GQ4():
    return SegmentRule(
        np.array([
            -0.8611363115940526,
            -0.3399810435848563,
             0.3399810435848563,
             0.8611363115940526
        ]),
        np.array([
            0.3478548451374539,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374539
        ])
    )


def GQ5():
    return SegmentRule(
        np.array([
            -0.9061798459386640,
            -0.5384693101056831,
             0.,
             0.5384693101056831,
             0.9061798459386640
        ]),
        np.array([
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891
        ])
    )


def GQ6():
    return SegmentRule(
        np.array([
            -0.9324695142031521,
            -0.6612093864662645,
            -0.2386191860831969,
             0.2386191860831969,
             0.6612093864662645,
             0.9324695142031521
        ]),
        np.array([
            0.1713244923791704,
            0.3607615730481386,
            0.4679139345726910,
            0.4679139345726910,
            0.3607615730481386,
            0.1713244923791704
        ])
    )


def GQ7():
    return SegmentRule(
        np.array([
            -0.9491079123427585,
            -0.7415311855993945,
            -0.4058451513773972,
             0.,
             0.4058451513773972,
             0.7415311855993945,
             0.9491079123427585
        ]),
        np.array([
            0.1294849661688697,
            0.2797053914892766,
            0.3818300505051189,
            0.4179591836734694,
            0.3818300505051189,
            0.2797053914892766,
            0.1294849661688697
        ])
    )


def GQ8():
    return SegmentRule(
        np.array([
            -0.9602898564975363,
            -0.7966664774136267,
            -0.5255324099163290,
            -0.1834346424956498,
             0.1834346424956498,
             0.5255324099163290,
             0.7966664774136267,
             0.9602898564975363
        ]),
        np.array([
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.3626837833783620,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763
        ])
    )


def GQ9():
    return SegmentRule(
        np.array([
            -0.9681602395076261,
            -0.8360311073266358,
            -0.6133714327005904,
            -0.3242534234038089,
             0.,
             0.3242534234038089,
             0.6133714327005904,
             0.8360311073266358,
             0.9681602395076261
        ]),
        np.array([
            0.0812743883615744,
            0.1806481606948574,
            0.2606106964029354,
            0.3123470770400029,
            0.3302393550012598,
            0.3123470770400029,
            0.2606106964029354,
            0.1806481606948574,
            0.0812743883615744
        ])
    )


def GQ10():
    return SegmentRule(
        np.array([
            -0.9739065285171717,
            -0.8650633666889845,
            -0.6794095682990244,
            -0.4333953941292472,
            -0.1488743389816312,
             0.1488743389816312,
             0.4333953941292472,
             0.6794095682990244,
             0.8650633666889845,
             0.9739065285171717
        ]),
        np.array([
            0.0666713443086881,
            0.1494513491505806,
            0.2190863625159820,
            0.2692667193099963,
            0.2955242247147529,
            0.2955242247147529,
            0.2692667193099963,
            0.2190863625159820,
            0.1494513491505806,
            0.0666713443086881
        ])
    )
