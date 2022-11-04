import pickle
import numpy as np
results = pickle.load(open('res_dict.pkl', 'rb'))
errors = []
for key, result in results.items():
    print(result['rul']['true'].shape)
    errors.append(abs(result['rul']['true'][100]-result['rul']['transfer'][100]))
print(len(errors), np.mean(errors), np.var(errors))