import numpy as np

np.random.seed(123)

# Initialize Random_Walk
random_walk = [0]

for x in range(100):
	step = random_walk[-1]

	# Roll the Dice
	dice = np.random.randint(1,7)

	# Determine Next Step
	if dice <= 2:
		step = max(0,step - 1)
	elif dice in [3,4,5]:
		step += 1
	else:
		step += np.random.randint(1,7)

	# Append Step to Random Walk
	random_walk.append(step)

print(random_walk)