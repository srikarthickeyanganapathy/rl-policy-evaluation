# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
The Bandit Slippery Walk problem is a Reinforcement Learning (RL) problem in which the agent must learn to navigate a slippery environment to reach the goal state.

1. we are tasked with creating an RL agent to solve the "Bandit Slippery Walk" problem.

2. The environment consists of Seven states representing discrete positions the agent can occupy.

3. The agent must learn to navigate this environment while dealing with the challenge of slippery terrain.

4. Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## STATE
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.Five Transition states / Non-terminal States including S: The starting state.

## Actions
The agent can take two actions: R (move right) and L (move left). 

The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.

## REWARD
The agent receives a reward of +1 for reaching the goal state and a reward of 0 for all other states.

## GRAPHICAL REPRESENTATION
![Graph](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/39233bd8-d7fc-44ca-b981-475977ed383b)

## FORMULA
![form](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/01443f3a-5f3a-4a5a-8f04-9764f555f94f)


## POLICY EVALUATION FUNCTION
~~~python

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
# code  to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
      return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing the two policies
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
~~~
## OUTPUT:
### POLICY 1
![P1](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/27415880-1365-4184-a0bc-8f86cf478c76)
![P2](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/57da26d4-f53a-42d5-b921-2fd7db215d8a)
![P3](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/be95d100-264f-4274-ba10-aaf02e5d506d)


### POLICY 2
![PP1](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/4c5e456e-5e22-4d77-bd4b-bb9c76663da0)
![PP2](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/88f8678f-55e5-4d3b-beae-7c70c1aeae68)
![PP3](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/526d9599-88ce-4e38-879c-a7126fde9321)

### COMPARISON
![C1](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/d4b262df-e6aa-4275-947d-773ef1a67967)


### CONCLUSION
![CC1](https://github.com/Manojrathinavelu/rl-policy-evaluation/assets/119560395/1530c7f8-50e6-408b-9f8b-bd0241b8d99a)



## RESULT:
Thus, This program will evaluate the given policy in the Bandit Slippery Walk environment and predict the expected reward of the policy.
