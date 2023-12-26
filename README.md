# Reinforcement-Learning-in-Preventive-Maintenance-and-production-scheduling
<details open="open">
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#problem-description">Problem Description</a>
    </li>
    <li>
      <a href="#methodology">Methodology</a>
      <ul>
        <li><a href="#gr-learning">GR-Learning</a></li>
        <li><a href="#r-learning">R-Learning</a></li>
        <li><a href="#hR-learning-heuristic-r-learning">HR-Learning</a></li>
      </ul>
    </li>
    <li><a href="#experiment-result">Experiment Result</a></li>
      <ul>
        <li><a href="#convergence-of-reinforcement-learning-algorithm">Convergence of Reinforcement Learning Algorithm</a></li>
        <li><a href="#scheduling-visualization">Scheduling Visualization</a></li>
      </ul>
    <li><a href="#conclusion">Conclusion</a></li>
      <ul>
        <li><a href="#conclusion">Conclusion</a></li>
        <li><a href="#contributions">Contributions</a></li>
        <li><a href="#limitation">Limitation</a></li>
        <li><a href="#future-work">Future work</a></li>
      </ul>
  </ol>
</details>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Introduction
  In this project, we reference the paper: "Joint optimization of preventive maintenance and production scheduling for multi-state production systems based on reinforcement learning" by Hongbing Yang, Wenchao Li, and Bin Wang from Soochow University, Jiangsu   University, and Jiangsu Academy of Safety Science and Technology, China. We explore the integration of production scheduling and preventive maintenance aimed at improving the reliability and productivity of manufacturing systems. Traditional production scheduling methods often overlook the impact of maintenance activities on operational efficiency, and our approach through reinforcement learning, specifically the integration of Markov Decision Processes (MDP) and Heuristic Reinforcement Learning, effectively addresses the optimization of long-term average rewards in multi-state single-machine production systems.

Our research is focused on how to integrate preventive maintenance strategies into production scheduling, taking into account the effect of deterioration over time, which includes the gradual decline in machine efficiency and quality, leading to increased production and maintenance costs. Our study proposes an innovative integrated model that includes planning decisions for preventive maintenance as well as single-machine production scheduling decisions, thereby making the entire production process more efficient and economical.

The goal of our research is to provide an operational guideline to help industrial practitioners better understand and implement integrated strategies for production scheduling and preventive maintenance. Through this project, we aim to provide meaningful insights and practical solutions to advance the manufacturing industry.

## Problem Description
  We concentrate on a production system involving the processing of multiple products on a single machine, 
  and we assume that the machine can only handle one job at a time, and all job types in the machine's buffer are available at any moment. <br>
  <br>
  The machine's condition deteriorates as its usage time increases.
  We define M + 1 machine states, denoted by i, where i ∈ {0, 1, ⋯, M}; 0 and M signify the best and worst machine conditions, respectively. 
  When selecting a job for processing on the machine at stage k,
  the machine's condition transitions from state s<sub>k</sub> to s<sub>k+1</sub> at the succeeding stage k + 1 with a certain probability. 
  At each decision-making epoch, the agent (decision maker) determines an action from all possible actions, including all job and preventive maintenance task.
  Let A represent the set of N + 1 actions, denoted as A = {a<sub>1</sub>,..., a<sub>N</sub>, a<sub>N+1</sub>}, where a<sub>N+1</sub> represents a preventive maintenance activity, and a<sub>n</sub> refers to the choice of job type n for processing, where n ∈ {1, 2, ⋯, N}.<br>
  <br>
  We assume that the state transition follows a Markov process in the machine degradation process.
  In other words, the state transition is independent of past states and relies solely on the present state and the chosen action.<br>
  
  It is assumed that preventive maintenance restores the machine to its optimal condition and the likelihood of transitioning to a better state to be zero.
  Additionally, as the machine state worsens, the likelihood of transitioning to a worse succeeding state increases. 
  <br>
  ```py
  action_num = 12
  state_num = 6

  P_ij = np.array([[0.1,0.9,0,0,0,0],
                  [0,0.1,0.9,0,0,0],
                  [0,0,0.1,0.9,0,0],
                  [0,0,0,0.1,0.9,0],
                  [0,0,0,0,0.1,0.9],
                  [0,0,0,0,0,1]]
                  )

  maitenance_index = action_num-1
  ```
  Hence, the state transition process is calculated as follows:<br>
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/transition_prob..png" width="580" height="70">
  ```py
  P_aij = np.zeros((action_num,state_num,state_num))
  def cal_P_aij(P_aij):
      for a in range(action_num):
          for i in range(state_num):
              for j in range(state_num):
                  if a != maitenance_index and j>=i:  
                     P_aij[a][i][j] = P_ij[i][j]
                  if a == maitenance_index and j==0:  
                     P_aij[a][i][j] = 1
      return P_aij
  ```
  ```py
  def get_next_state(state,action):
      state_list = [i for i in range(state_num)]
      weights = P_aij[action][state]
      next_state = random.choices(state_list, weights=weights, k=1)[0]
      return next_state
  ```
  Deteriorating machine conditions result in higher maintenance costs.
  Therefore, the maintenance cost function is characterized as a non-decreasing function in state i. 
  Specifically, if i ≥ i<up>′</up>, then c<sub>m</sub>(i,a) ≥ c<sub>m</sub>(i<up>′</up> , a).
  The machine processing cost per time unit in state i is delineated as:<br>
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/maintainence_cost.png" width="580" height="60">
  ```py
  def cal_maintenance_cost(maintenance_cost):
      maintenance_cost = np.zeros((state_num,action_num))
      maintenance_cost[1][maitenance_index] = 1.5
      maintenance_cost[2][maitenance_index] = 1.8
      maintenance_cost[3][maitenance_index] = 2.1
      maintenance_cost[4][maitenance_index] = 2.4
      maintenance_cost[5][maitenance_index] = 2.9
      maintenance_cost[0][maitenance_index] = 1
  
      return maintenance_cost
  ```
  The degradation of the machine condition increases the processing cost per time unit.
  Hence, if i ≥ i′, then c<sub>p</sub>(i, a) ≥ c<sub>p</sub>(i′, a).
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/process_cost.png" width="550" height="80">
  ```py
  def cal_proc_cost(proc_cost):
      proc_cost = np.zeros((state_num,action_num))
      for i in range(state_num-2): 
          proc_cost[i][:] = round(random.uniform(i,i+1),1)
      for i in range(state_num-2,state_num): 
          proc_cost[i][:] = round(random.uniform(i+2,i+3),1)
      proc_cost[5][:]=5.1
      return proc_cost
  ```
  This machine can handle various job types, with t<sub>n</sub> representing the processing time for job type n, where n ∈ {1, 2, ⋯, N}, which is given randomly. 
  ```py
  def cal_proc_time(proc_time):
      proc_time = np.zeros((action_num))
      for a in range(action_num-1):
          proc_time[a] = round(random.uniform(0.8,2.6),1)
          # proc_time[a] = round(random.uniform(0.3,3.1),1)# increasing uncertainty, lower the efficiency of HR algorithm
  
      return proc_time
  ```
  The completion reward for all actions in state i is denoted by the function r<sub>o</sub>(i, a) as:
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/completion_reward.png" width="600" height="80"><br>
  r<sub>o</sub>(a) denotes the completion reward for a type of job when taking action a, which is given randomly. No reward is received during the execution of maintenance activities.
  ```py
  def cal_completion_reward(completion_reward):
      completion_reward = np.zeros((state_num,action_num))
      for a in range(action_num-1):
          reward = round(random.uniform(1.1,3.4),1)
          # reward = round(random.uniform(0.5,4),1)# increasing uncertainty, lower the efficiency of HR algorithm
          for i in range(state_num): 
              completion_reward[i][a] = reward
  
      return completion_reward
  ```
  Building upon the earlier considerations of preventive maintenance costs, processing costs, and completion rewards functions, the immediate rewards function at stage k in the production process is derived:<br>
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/reward_function.png" width="600" height="40">
  ```py
  def immediate_reward(state,action):
      return completion_reward[state][action] - maintenance_cost[state][action] - proc_cost[state][action]*proc_time[action]
  ```
## Methodology

### GR-Learning

GR-Learning (Gamma-Reward Learning) is an algorithm designed to integrate the concepts of Q-learning and R-learning, aiming at maximizing long-run expected average rewards in a given Markov Decision Process. This algorithm updates the Q-values by considering the immediate reward plus the discounted future state's maximum Q-value.

**Algorithm Steps**

**Variables Initialization**
1. `Q`: A state-action value function initialized to zeros for all state and action pairs.
2. `epsilon`: Exploration factor, starting at 0.9.
3. `avg_reward_list`: To track the average reward.
4. `history_state_list, history_action_list, history_reward_list, history_exp_avg_reward_list`: To keep the history of states, actions, and rewards.
5. `rho`: Average reward rate, initialized to 0.

**Function Definitions**
1. **Q_value_updating**: Updates the Q-value for a specific state and action. It considers the immediate reward, discounted maximum Q-value of the next state, average reward rate (rho), and the learning rate (`beta_1`).
   
   ```python
   def Q_value_updating(cur_state, action, next_state, Q, immediate_reward, rho):
       return (1 - beta_1) * Q[cur_state][action] + beta_1 * (immediate_reward(cur_state, action) - rho + gamma * np.max(Q[next_state]))

2. **epsilon_greedy_policy**: Select an action using the epsilon-greedy strategy. With probability epsilon, it selects a random action; otherwise, it selects the action with the highest Q-value.
     ```py 
     def epsilon_greedy_policy(state, Q, epsilon):
         if random.uniform(0, 1) < epsilon:
            return random.randint(0, action_num - 1)
         else:
            return np.argmax(Q[state])
    ```
### Main Algorithm Loop

The `GR_learning` function is the core of the GR-Learning algorithm, where the iterative process of learning and decision-making occurs. Here is how it works:

1. **Initialize the Q-table for state-action pairs**: A matrix `Q` is initialized with zeros for all state-action pairs, representing the expected rewards for taking an action in a given state.

2. **Iterate over a specified number of iterations `iterative_num`**: The algorithm runs for a predefined number of steps, refining its strategy at each step.

3. **At each iteration**:

   - **Select an action using the `epsilon_greedy_policy`**: The function decides whether to take the best-known action or explore a new action based on the value of `epsilon`.

   - **Observe the next state and reward after taking the action**: The environment returns the next state and the observed reward after performing the action.

   - **Update the Q-value for the current state and action**: Using the `Q_value_updating` function, the algorithm updates its Q-table with new knowledge gained from the latest action.

   - **Update the average reward rate (`rho`)**: The average reward rate is updated to reflect the latest information.

   - **Decay the `epsilon` value over time**: To balance exploration and exploitation, `epsilon` decreases with each step, making the policy more greedy as it learns more about the environment.

   - **Append the current state, action, and reward to the history lists**: Keeps track of the entire sequence of states, actions, and rewards for analysis and debugging.

### GR-Learning Function

```python
def GR_learning(initial_state, iterative_num):
    # Initializations
    Q = np.zeros((state_num, action_num))
    epsilon = 0.9  # Exploration factor
    rho = 0  # Average reward rate
    
    # History lists
    history_state_list = []
    history_action_list = []
    history_reward_list = []
    history_exp_avg_reward_list = []
    
    cur_state = initial_state
    for k in range(iterative_num):
        action = epsilon_greedy_policy(cur_state, Q, epsilon)
        next_state = get_next_state(cur_state, action)
        reward = immediate_reward(cur_state, action)
        
        Q[cur_state][action] = Q_value_updating(cur_state, action, next_state, Q, immediate_reward, rho)
        
        rho = (1-beta_2)*rho + beta_2*reward
        
        # Append to history
        history_state_list.append(cur_state)
        history_action_list.append(action)
        history_reward_list.append(reward)
        history_exp_avg_reward_list.append(sum(history_reward_list)/len(history_reward_list))
        
        cur_state = next_state
        epsilon *= 1 / (1 + beta_2 * k)  # Decay epsilon
        
    avg_reward = np.mean(history_reward_list)
    return history_state_list, history_action_list, history_exp_avg_reward_list, avg_reward, Q
```
<br>

### R-Learning
Before entering the algorithm, we first define several functions for the algorithm to use.
After calculating the immediate reward value R(i, a) at each decision point, we adjust the relative average reward R̄<sup>π</sup>(i, a) and the average reward ρ following the specified rules.

<img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/relative_average_reward.png" width="580" height="60">

 ```py
 beta_1 = 0.1 # learning rate
 def rel_avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward):
     return (1-beta_1)*rel_avg_reward[cur_state][action] + beta_1*(immediate_reward(cur_state,action)-avg_reward+max(rel_avg_reward[next_state]))
  ```

<img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/average_reward.png" width="580" height="60">

  ```py
  beta_2 = 0.01 # learning rate    
  def avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward):
      return (1-beta_2)*avg_reward + beta_2*(immediate_reward(cur_state,action)+max(rel_avg_reward[next_state])-max(rel_avg_reward[cur_state]))
  ```
 **R-learning algorithm** <br>
 **Step1:** Initialize the learning rates β₁, β₂, exploration factor ε₀, decaying factor ψ, average reward ρ, and relative average reward R̄(i,a). Calculate the immediate reward of each (state, action) pair, initialize some record list, and let the current machine state be 0.
  ```py
  def iterative_R_alg(initial_state,iterative_num):
      k = 0
      epsilon = 0.9 # exploration factor
      phi = 1.005 # decaying factor
      avg_reward = 0
      rel_avg_reward = np.zeros((state_num,action_num))
      for i in range(state_num): 
          for a in range(action_num):
              rel_avg_reward[i][a] = immediate_reward(i,a)
      history_state_list = []
      history_action_list = []
      history_reward_list = []
      history_exp_avg_reward_list = []
      cur_state = initial_state
      next_state = 0
      action = 0
  ```
  We update according rewards in iterative_num.<br>
  **Step2:**  Calculate exploration probability ε by using ε = ε<sub>0</sub> / ψᵏ.
  ```py 
      while k<iterative_num:
          # step2
          epsilon = epsilon/phi
  ```
  **Step3:** Choose and carry out the action a that has the highest R̄ᵏ(i,a) value with probability 1-ε, else, randomly choose other exploration action a with probability ε/(|A|-1), where |A| is the number of actions in action set A. Let the next state be j transferred from i. If the selected action a is a non-exploratory action, update the average reward ρ; otherwise, the average reward ρ is kept the same.
  ```py
          # step3
          exploitation = random.choices([0,1], weights=[epsilon,1-epsilon], k=1)[0]
          if exploitation == 1 :
              action = np.argmax(rel_avg_reward[cur_state])
              next_state = get_next_state(cur_state,action)
              avg_reward = avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
          else:
              action_candidate_list = [i for i in range(action_num) if i!=np.argmax(rel_avg_reward[cur_state])]
              action = random.choices(action_candidate_list, weights=[1/len(action_candidate_list) for i in range(len(action_candidate_list))], k=1)[0]
              next_state = get_next_state(cur_state,action)
  ```
 **Step4:** Calculate the immediate reward value R(i,a). Update the relative average reward R̄ᵏ(i,a).
  ```py 
          rel_avg_reward[cur_state][action] = rel_avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
          history_action_list.append(action)   
          history_state_list.append(next_state)
          history_reward_list.append(immediate_reward(cur_state,action))
          # print(immediate_reward(cur_state,action))
          history_exp_avg_reward_list.append(sum(history_reward_list)/len( history_reward_list))
  ```
  **Step5:** Set k=k+1, and the current state to be j. Repeat steps 2–5 until the terminal condition is reached.
  ```py 
          # step 5
          cur_state = next_state
          k += 1
          # print(avg_reward)
      return history_state_list, history_action_list, history_exp_avg_reward_list,rel_avg_reward
  ```
<br>

### HR-Learning (Heuristic R-Learning)
  Before introducing the steps of the algorithm, it is crucial to elucidate this heuristic property. This heuristic property explains that under specific conditions, preventive maintenance will be the best action to conduct. <br><br>
  **Property: If c<sub>m</sub>(i,a<sub>N+1</sub>) ≤ min<sub>a≠a<sub>N+1</sub></sub>[c<sub>p</sub>(i, a)t<sub>n</sub> − r<sub>o</sub>(i, a)] for all i, such that the preventive maintenance should be conducted in i under the optimal stationary policy.<br>**<br>
  To prove it, for any state i, if, <br>
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/property2_pf1.png" width="330" height="70"><br>
  then a<sub>N+1</sub> must be the best action to choose. And h(i) here represents the infinite horizon discounted expected reward function. According to the reference paper, it is a non-increasing function. So<br>
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/property2_pf2.png" width="120" height="25">.<br>
  Thus, when c<sub>m</sub>(i, a<sub>N+1</sub>) ≤ min<sub>a≠a<sub>N+1</sub></sub>[c<sub>p</sub>(i, a)t<sub>n</sub> − r<sub>o</sub>(i, a)] holds, there must be R(i, a<sub>N+1</sub>) ≥ R(i, a) holds, so that the first inequation hold, the proof complete.<br>
  <br>
  **With this property, we can incorporate the heuristic rule into the R learning algorithm mentioned above.**
  <br>
  <br>
  Firstly, we define a function for the algorithm to use. This function randomly generates eta to decide whether to exploit the heuristic rule.
  ```py
  def eta_generate():
      return random.uniform(0,1)
  ```
  **HR learning algorithm** <br>
  **Step1:** Initialize the learning rates β₁, β₂, exploration factor ε<sub>0</sub>, trigger factor η₀, decaying factor ψ, average reward ρ, and relative average reward R̄(i,a). Calculate the immediate reward of each (state, action) pair, initialize some record list, and let the current machine state be 0.
  ```py
  def iterative_HR_alg(initial_state,iterative_num):
      k = 0
      epsilon = 0.9 # exploration factor
      eta_0 = 0.1 # trigger factor
      phi = 1.005 # decaying factor
      avg_reward = 0
      rel_avg_reward = np.zeros((state_num,action_num))
    
      for i in range(state_num): 
          for a in range(action_num):
              rel_avg_reward[i][a] = immediate_reward(i,a)
      
      history_state_list = []
      history_action_list = []
      history_reward_list = []
      history_exp_avg_reward_list = []
      cur_state = initial_state
      next_state = 0
      action = 0
  ```
  Same as R Learning, we update according rewards in iterative_num.<br>
  **Step2:** Randomly generate a number η by above function. Then, calculate exploration probability ε by using ε = ε<sub>0</sub> / ψᵏ. 

  ```py
      while k<iterative_num:
          # step2
          eta = eta_generate()
          epsilon = epsilon/phi
  ```
  **Step3:** Calculate c<sub>m</sub>(i, a<sub>N+1</sub>). If c<sub>m</sub>(i, a<sub>N+1</sub>) ≤ min<sub>a≠a<sub>N+1</sub></sub>[c<sub>p</sub>(i, a)t<sub>n</sub> − r<sub>o</sub>(i, a)], and if η > η<sub>0</sub>(1− ε), perform preventive maintenance; else go to Step4. If we perform preventive maintenance, set the state goes to 0 and calculate the immediate reward value R(i,a), and update the relative average reward R̄<sup>π</sup>(i, a) and the average reward ρ.
  ```py
          # step3 
          if maintenance_cost[cur_state][maitenance_index] <= min(
              [(proc_cost[cur_state][action]*proc_time[action])-(completion_reward[cur_state][action]) for action in range(action_num-1)]) and eta>eta_0*(1-epsilon):
              action = maitenance_index
              next_state = 0
              rel_avg_reward[cur_state][action] = rel_avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
              avg_reward = avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
  ```
  **Step4:** Choose and carry out the action a that has the highest R̄ᵏ(i,a) value with probability 1-ε, else, randomly choose other exploration action a with probability ε/(|A|-1), where |A| is the number of actions in action set A. Let the next state be j transferred from i. If the selected action a is a non-exploratory action, update the average reward ρ; otherwise, the average reward ρ is kept the same.
  ```py
          else:
              # step 4
              exploitation = random.choices([0,1], weights=[epsilon,1-epsilon], k=1)[0]
              if exploitation == 1 :
                  action = np.argmax(rel_avg_reward[cur_state])
                  next_state = get_next_state(cur_state,action)
                  avg_reward = avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
              else:
                  action_candidate_list = [i for i in range(action_num) if i!=np.argmax(rel_avg_reward[cur_state])]
                  action = random.choices(action_candidate_list, weights=[1/len(action_candidate_list) for i in range(len(action_candidate_list))], k=1)[0]
                  next_state = get_next_state(cur_state,action)
  ```
  **Step5:** Calculate the immediate reward value R(i,a). Update the relative average reward R̄ᵏ(i,a).
  ```py
  
              # step 5
              rel_avg_reward[cur_state][action] = rel_avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
          history_action_list.append(action)   
          history_state_list.append(next_state)
          history_reward_list.append(immediate_reward(cur_state,action))
          # print(immediate_reward(cur_state,action))
          history_exp_avg_reward_list.append(sum(history_reward_list)/len( history_reward_list))
  ```
  **Step6:** Set k=k+1, and the current state to be j. Repeat steps 2–6 until the terminal condition is reached.
  ```py 
          # step 6
          cur_state = next_state
          k += 1
          # print(avg_reward)
      return history_state_list, history_action_list, history_exp_avg_reward_list,rel_avg_reward
  ```

##  Experiment Result
The numerical experiments aim to confirm the performance and effectiveness of three algorithms implemented on a single machine with six states. The machine is responsible for producing various types of jobs, and the transitions between states during the production process follow the Markov chain model. In this study, production costs primarily consist of preventive maintenance costs and job processing costs. The processing costs are contingent upon the processing time of each job type and the machine's processing cost per unit time in a given state. The processing time, denoted as t<sub>n</sub>, for each job type is generated using a uniform distribution. Upon the completion of a job, a corresponding completion reward is received, and this reward follows a uniform distribution tailored to different job types.
### Convergence of Reinforcement Learning Algorithm
<img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/converge.png"><br>
As depicted in the figure, it is evident that HR-learning exhibits faster convergence compared to R-learning. Furthermore, HR-learning consistently outperforms R-learning across the entire iterative process. Conversely, while GR-learning converges more rapidly than HR-learning, it demonstrates inferior performance compared to HR-learning throughout the iterative process. Remarkably, R-learning outperforms GR-learning after around 900 decision-making steps.

<br>

### Scheduling Visualization
<p float="left">
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/GR.png" width="330" height="220">
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/R.png" width="330" height="220">
  <img src="https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/HR.png" width="330" height="220">
</p>
As mentioned in the problem description, this is a problem about single machine with six states. Therefore, instead of representing the scheduling through a Gantt chart, our focus is on the selection and switch of actions and states. The above diagram illustrates the last 50 states/actions and schedules among 6000 decision points. As you can observe, due to algorithm convergence, the agent tends to choose a few actions, typically associated with higher rewards or shorter process times.<br>
Another noteworthy observation is that the agent tends to keep the machine in a healthier state. Once the machine enters a moderately deteriorated state, the agent conducts preventive maintenance. This is because in a less healthy state, choosing to conduct a job may result in a lower overall reward.

## Conclusion
### Conclusion
HR-Learning, as an advanced evolution of R-Learning, successfully addresses key challenges in reinforcement learning by incorporating domain knowledge and opportunity cost considerations. It not only mitigates the issue of poor initial returns due to aggressive exploration but also ensures better long-term convergence by weighing additional benefits relative to other decision-making behaviors. While GR-Learning offers quick convergence, HR-Learning proves to be superior in the long run, providing more stable and effective results, particularly when extended over longer running intervals. This makes HR-Learning a robust choice, accelerating convergence and enhancing post-convergence effectiveness, thereby solving some of the persistent issues in reinforcement learning and making it a more viable and stable option for complex decision-making scenarios.

<br>

### Contributions
**Theoretical Contributions** <br>
Advancement in Reinforcement Learning Algorithms: This study significantly contributes to the reinforcement learning field by advancing R-Learning through the introduction of HR-Learning. This new algorithm incorporates domain knowledge and opportunity costs into the learning process, addressing the challenge of poor initial returns and ensuring better long-term convergence.

Incorporation of Domain Knowledge: By embedding domain knowledge into the HR-Learning algorithm, the research showcases a methodological enhancement that accelerates the learning process and improves decision-making quality, which is a substantial theoretical contribution.

Optimization of Multi-State Single-Machine Production Systems: The study extends the application of reinforcement learning to the specific context of multi-state single-machine production systems, providing a nuanced understanding of how state transitions, processing costs, and maintenance strategies interact and can be optimized in a complex production environment.

Comprehensive Performance Analysis: The extensive comparison between HR-Learning, R-Learning, and GR-Learning provides a thorough theoretical analysis of the convergence patterns and performance metrics, contributing to a deeper understanding of reinforcement learning behaviors in different scenarios.

**Practical Contributions** <br>
Improved Production Scheduling and Maintenance: Practically, this research offers a viable solution to integrate preventive maintenance and production scheduling, leading to improved operational efficiency and reduced costs in real-world manufacturing settings.

Decision-Making Strategy for Industry Application: The HR-Learning algorithm serves as a robust decision-making strategy that industry professionals can adopt for better management of machine states and job scheduling, ensuring higher productivity and extended machine life due to timely preventive maintenance.

Algorithmic Efficiency and Stability: By demonstrating HR-Learning's superiority in long-term performance and stability, the study provides a practical tool that industries can rely on for consistent decision-making, especially in complex and dynamic production environments.

Sensitivity to State Transitions: The findings related to the sensitivity of the proposed algorithm to state transition probabilities offer valuable insights for practitioners to customize and fine-tune the approach according to specific operational dynamics, leading to more tailored and effective maintenance and scheduling strategies.

Both the theoretical and practical contributions of this study provide significant advancements in understanding and applying reinforcement learning to complex scheduling and maintenance tasks, offering pathways for both continued academic research and real-world industrial applications.

<br>

### Limitations:

This model suite might face several limitations, including:

1. **Real-Time Adaptability**: The model may not adapt quickly to sudden changes in machine conditions or unexpected breakdowns, especially if these events are not represented in the historical data.

2. **Dependency on Accurate Domain Knowledge**: 

The effectiveness of the model heavily relies on the accuracy of the incorporated domain knowledge about machine deterioration. Misestimations can lead to suboptimal decision-making.

3. **Maintenance Scheduling Complexity**: 

Integrating a dynamic deterioration model into the scheduling process increases the complexity of maintenance planning, potentially making it harder to predict maintenance windows and allocate resources efficiently.

Considering these limitations is crucial when applying or further developing the model suite to ensure its practicality and effectiveness in real-world settings.

<br>

### Future work:

For future work, we propose focusing on the dynamic aspects of machine deterioration, acknowledging that different actions can significantly influence the rate of wear and tear. Instead of a uniform transition probability, a variable model should be considered, which adapts the deterioration rate based on the specific tasks being performed. This involves understanding and quantifying how certain tasks may impose more strain on the machinery than others. By integrating a responsive deterioration model that adjusts probabilities according to the task type and intensity, we can achieve more accurate maintenance scheduling and extend the machinery's operational lifespan.

Such an approach will allow for more efficient production processes by customizing strategies to the unique impacts of each job type on machine degradation. This will enable a more sophisticated method to predict and mitigate risks associated with intensive operational demands. Implementing this model will require detailed data collection, advanced modeling techniques, and possibly real-time monitoring of machine conditions. This advancement could significantly enhance the sustainability and productivity of manufacturing systems, making this an important area of research and development.
