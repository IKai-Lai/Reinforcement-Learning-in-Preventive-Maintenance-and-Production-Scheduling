# Reinforcement-Learning-in-Preventive-Maintenance-and-production-scheduling
* Introduction
* Problem Description
* Methodology
  *  GR-learning
  * R-learning
  * HR-learning
* Scheduling Visualization
* Conclusion
  * Conclusion
  * Contribution
  * Limitation
  * Future work
## Problem Description
  We concentrate on a production system involving the processing of multiple products on a single machine. 
  This machine can handle various job types, with t<sub>n</sub> representing the processing time for job type n, where n ∈ {1, 2, ⋯, N}. 
  We assume that the machine can only handle one job at a time, and all job types in the machine's buffer are available at any moment. 
  The machine's condition deteriorates as its usage time increases.
  We define M + 1 machine states, denoted by i, where i ∈ {0, 1, ⋯, M}; 0 and M signify the best and worst machine conditions, respectively. 
  In other words, the higher the value of state i, the more deteriorated the machine condition.
  When selecting a job for processing on the machine at stage k,
  the machine's condition transitions from state s<sub>k</sub> to s<sub>k+1</sub> at the succeeding stage k + 1 with a certain probability. 
  At each decision-making epoch, the agent (decision maker) determines an action from all possible actions, including all job and preventive maintenance task.
  Let A represent the set of N + 1 actions, denoted as A = {a<sub>1</sub>,...,a<sub>N</sub>,a<sub>N+1</sub>}, where a<sub>N+1</sub> represents a preventive maintenance activity, and an refers to the choice of job type n for processing, where n ∈ {1, 2, ⋯, N}.
  Additionally, we assume that the state transition follows a Markov process in the machine degradation process.
  In other words, the state transition is independent of past states and relies solely on the present state and the chosen action.
  Due to the degradation of the machine, there is an associated rise in the cost of processing jobs.
  To reduce these costs, preventive maintenance is usually carried out to enhance the machine's condition timely.
  It is assumed that preventive maintenance restores the machine to its optimal condition, essentially bringing it back to a state equivalent to being "as good as new".
  The state transition process is calculated as follows:
  In general, as the machine state worsens, the likelihood of transitioning to a worse succeeding state increases. 
  Therefore, we assume that the likelihood of transitioning to a better state.
  The preventive maintenance costs in state i are expressed by the following function:

  ```py
  action_num = 12
  state_num = 6

  maintenance_cost = np.zeros((state_num,action_num))
  proc_cost = np.zeros((state_num,action_num))
  proc_time = np.zeros((action_num))
  completion_reward = np.zeros((state_num,action_num))
  P_ij = np.array([[0.1,0.9,0,0,0,0],
                  [0,0.1,0.9,0,0,0],
                  [0,0,0.1,0.9,0,0],
                  [0,0,0,0.1,0.9,0],
                  [0,0,0,0,0.1,0.9],
                  [0,0,0,0,0,1]]
                  )

  maitenance_index = action_num-1
  ```

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
  Moreover, deteriorating machine conditions result in higher maintenance costs.
  Therefore, the maintenance cost function is characterized as a non-decreasing function in state i. 
  Specifically, if i ≥ i<up>′</up>, then c<sub>m</sub>(i,a) ≥ c<sub>m</sub>(i<up>′</up> , a).
  The machine processing cost per time unit in state i is delineated as:
  ![GITHUB](https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/maintainence_cost.png)

  ```py
  def cal_maintenance_cost(maintenance_cost):
      maintenance_cost[1][maitenance_index] = 1.5
      maintenance_cost[2][maitenance_index] = 1.8
      maintenance_cost[3][maitenance_index] = 2.1
      maintenance_cost[4][maitenance_index] = 2.4
      maintenance_cost[5][maitenance_index] = 2.9
      maintenance_cost[0][maitenance_index] = 1
  
      return maintenance_cost
  ```
  ```py

  def cal_proc_cost(proc_cost):
      for i in range(state_num-2): 
          proc_cost[i][:] = round(random.uniform(i,i+1),1)
      for i in range(state_num-2,state_num): 
          proc_cost[i][:] = round(random.uniform(i+2,i+3),1)
      proc_cost[5][:]=5.1
      return proc_cost
  ```
  ```py
  def cal_proc_time(proc_time):
      for a in range(action_num-1):
          proc_time[a] = round(random.uniform(0.8,2.6),1)
          # proc_time[a] = round(random.uniform(0.3,3.1),1)# 區間大，則HR沒較好
  
      return proc_time
  ```
  ![GITHUB](https://github.com/IKai-Lai/Reinforcement-Learning-in-Preventive-Maintenance-and-Production-Scheduling/blob/main/image/completion_reward.png)


  ```py
  def cal_completion_reward(completion_reward):
      for a in range(action_num-1):
          reward = round(random.uniform(1.1,3.4),1)
          # reward = round(random.uniform(0.5,4),1)# 不太影響演算法效能
          for i in range(state_num): 
              completion_reward[i][a] = reward
  
      return completion_reward
  ```
  ```py
  def eta_generate():
      return random.uniform(0,1)
  ```
  ```py
  def immediate_reward(state,action):
      return completion_reward[state][action] - maintenance_cost[state][action] - proc_cost[state][action]*proc_time[action]
  ```
  ```py
  def get_next_state(state,action):
      state_list = [i for i in range(state_num)]
      weights = P_aij[action][state]
      next_state = random.choices(state_list, weights=weights, k=1)[0]
      return next_state
  ```
  ```py    
  def avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward):
      return (1-beta_2)*avg_reward + beta_2*(immediate_reward(cur_state,action)+max(rel_avg_reward[next_state])-max(rel_avg_reward[cur_state]))
  ```
  ```py
  def rel_avg_reward_updating(cur_state,action,next_state,avg_reward,rel_avg_reward):
      return (1-beta_1)*rel_avg_reward[cur_state][action] + beta_1*(immediate_reward(cur_state,action)-avg_reward+max(rel_avg_reward[next_state]))
  ```
