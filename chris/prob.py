import numpy as np

probs = []
for seed in range(40):
    print(seed)
    np.random.seed(seed)
    balls = np.zeros(30)
    balls[7] = 1
    balls[1] = 2
    balls[23] = 3
    balls[3] = 4
    success, trial = 0, 0

    while trial < 100000:
        trial += 1
        chosen = np.zeros(4)
        for i in range(4):
            value = np.random.choice(balls)
            chosen[i] = value

        non_zero_count = np.count_nonzero(chosen)
        if non_zero_count == 2:
            success +=1
            
    probs.append(success/trial)
    
print(np.mean(probs))
