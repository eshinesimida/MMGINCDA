import numpy as np
Y = np.loadtxt(r"association1.txt", dtype=float)
rng = np.random.default_rng(seed=99)
pos_samples = np.where(Y == 1)
known = []
for i in range(len(pos_samples[0])):
    a = []
    a.append(pos_samples[0][i])
    a.append(pos_samples[1][i])
    known.append(a)

print(known, len(known))
np.savetxt('known-D1.txt', known,fmt ='%d', delimiter='\t')

pos_samples_shuffled = rng.permutation(pos_samples, axis=1)


# get the edge of negative samples
# rng = np.random.default_rng(seed=42)
neg_samples = np.where(Y == 0)

unknown = []
for i in range(len(neg_samples[0])):
    a = []
    a.append(neg_samples[0][i])
    a.append(neg_samples[1][i])
    unknown.append(a)

print(unknown, len(unknown))
np.savetxt('unknown-D1.txt', unknown,fmt ='%d', delimiter='\t')
# neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
#
# edge_idx_dict = dict()
# n_pos_samples = pos_samples_shuffled.shape[1]