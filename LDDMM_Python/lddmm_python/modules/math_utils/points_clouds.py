import numpy as np

def decimate_indices(q, r) :
	def rec(l) :
		if l == [] :
			return []
		else :
			new_index = l.pop()
			new_point = q[new_index]
			s = []
			for i in l :
				curr_point = q[i]
				if sum((curr_point - new_point)**2) > r**2 :
					s.append(i)
			m = rec(s)
			m.append(new_index)
			return m
	
	return rec([j for j in range(q.shape[0])])
	
def decimate(q, r) :
	return np.array([q[i] for i in decimate_indices(q,r)])

