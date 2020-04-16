import numpy as np

np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1,7)

# Finish the control construct
if dice <= 2 :
    step = step - 1
elif dice in [3,4,5] :
    step += 1
else :
    step = step + np.random.randint(1,7)

# Print out dice and step
print(dice)
print(step)

# You threw a 6, so the code for the else statement was executed. You threw again, and apparently you threw 3, causing you to take three steps up: you're currently at step 53.