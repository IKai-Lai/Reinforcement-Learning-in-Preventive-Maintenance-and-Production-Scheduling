# Reinforcement-Learning-in-Preventive-Maintenance-and-production-scheduling
* Introduction
* Problem Description
* Methodology
  * GR-learning
  * R-learning
  * HR-learning
* Experiment Result
  * Convergence of Reinforcement Learning Algorithm
  * Scheduling Visualization
* Conclusion
  * Conclusion
  * Contribution
  * Limitation
  * Future work
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
      eposilon = 0.9 # exploration factor
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
          eposilon = eposilon/phi
  ```
  **Step3:** Choose and carry out the action a that has the highest R̄ᵏ(i,a) value with probability 1-ε, else, randomly choose other exploration action a with probability ε/(|A|-1), where |A| is the number of actions in action set A. Let the next state be j transferred from i. If the selected action a is a non-exploratory action, update the average reward ρ; otherwise, the average reward ρ is kept the same.
  ```py
          # step3
          exploitation = random.choices([0,1], weights=[eposilon,1-eposilon], k=1)[0]
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
--------------------------------------------------------------------------------------------------------------------
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
      eposilon = 0.9 # exploration factor
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
          eposilon = eposilon/phi
  ```
  **Step3:** Calculate c<sub>m</sub>(i, a<sub>N+1</sub>). If c<sub>m</sub>(i, a<sub>N+1</sub>) ≤ min<sub>a≠a<sub>N+1</sub></sub>[c<sub>p</sub>(i, a)t<sub>n</sub> − r<sub>o</sub>(i, a)], and if η > η<sub>0</sub>(1− ε), perform preventive maintenance; else go to Step4. If we perform preventive maintenance, set the state goes to 0 and calculate the immediate reward value R(i,a), and update the relative average reward R̄<sup>π</sup>(i, a) and the average reward ρ.
  ```py
          # step3 
          if maintenance_cost[cur_state][maitenance_index] <= min(
              [(proc_cost[cur_state][action]*proc_time[action])-(completion_reward[cur_state][action]) for action in range(action_num-1)]) and eta>eta_0*(1-eposilon):
              action = maitenance_index
              next_state = 0
              rel_avg_reward[cur_state][action] = rel_avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
              avg_reward = avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward)
  ```
  **Step4:** Choose and carry out the action a that has the highest R̄ᵏ(i,a) value with probability 1-ε, else, randomly choose other exploration action a with probability ε/(|A|-1), where |A| is the number of actions in action set A. Let the next state be j transferred from i. If the selected action a is a non-exploratory action, update the average reward ρ; otherwise, the average reward ρ is kept the same.
  ```py
          else:
              # step 4
              exploitation = random.choices([0,1], weights=[eposilon,1-eposilon], k=1)[0]
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
  ### Convergence of Reinforcement Learning Algorithm
  
  --------------------------------------------------------------------------------------------------------------------
  ### Scheduling Visualization
