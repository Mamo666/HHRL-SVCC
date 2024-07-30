"""
这版的Light无论如何只有一个网络(把TP去掉)
[T/P/tp/TP] [Gv/tgv/pgv/tpgv] [V/TV/PV/tpV]    # tpg:HATD3,v:worker,TPGV:TD3
"""

from collections import deque

import numpy as np

from algorithm import HATD3Triple, HATD3Double, HATD3, TD3Single, WorkerTD3, ManagerTD3

np.random.seed(3407)  # 设置随机种子


class IndependentLightAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.use_time = config['use_time']
        self.use_phase = config['use_phase']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None
        self.lstm_observe_every_step = config['lstm_observe_every_step']

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        if self.use_time and self.use_phase:
            self.network = HATD3(config)
        elif self.use_time:
            self.network = TD3Single(config, 'time')
        else:   # only phase or neither
            self.network = TD3Single(config, 'phase')
            # self.network = DQN(config, 'phase')
        self.save = lambda path, ep: self.network.save(path + 'light_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else 99
            self.network.load('../model/' + config['load_model_name'] + '/light_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.o_t = config['time']['obs_dim']
        self.T_t = config['time']['T']
        self.o_p = config['phase']['obs_dim']
        self.T_p = config['phase']['T']

        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow = config['yellow']
        self.red = config['red']

        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: config['min_green'] for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        tl, pl = [], []
        for light in self.light_id:
            t, p = self._step(env, light)
            tl.append(t)
            pl.append(p)
        return tl, pl
        # return tl[0], pl[0]  # 只向外展示第一个路口的动作

    def _step(self, env, light):
        if self.lstm_observe_every_step:
            o_t = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
            self.step_time_obs[light].append(o_t)                      # 存近10步obs(T, o_dim)
            o_p = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
            self.step_phase_obs[light].append(o_p)

        next_green, next_phase = None, None
        if self.time_index[light] == 0:
            if self.color[light] == 'y' and self.red != 0:  # 黄灯结束切红灯
                env.set_light_action(light, self.phase_list[light][-2] * 3 + 2, self.red)
                self.time_index[light], self.color[light] = self.red, 'r'
            elif self.color[light] == 'r' or (self.color[light] == 'y' and self.red == 0):  # 红灯结束或（黄灯结束且无全红相位）切绿灯
                env.set_light_action(light, self.phase_list[light][-1] * 3, self.green[light])
                self.time_index[light], self.color[light] = self.green[light], 'g'
            elif self.color[light] == 'g':
                if not self.lstm_observe_every_step:
                    o_t = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
                    self.step_time_obs[light].append(o_t)                      # 存近10步obs(T, o_dim)
                    o_p = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
                    self.step_phase_obs[light].append(o_p)

                # Choose next phase
                if not self.use_phase or (not self.train_model and not self.load_model):
                    a_p = (self.phase_list[light][-1] + 1) % 4     # 不控制时默认动作
                else:
                    o_p = np.array(self.step_phase_obs[light]).flatten().tolist()
                    self.o_p_list[light].append(o_p)       # 存最近两次决策obs(2, T*o_dim)

                    if self.network.__class__.__name__ in ('TD3Single', 'HATD3'):
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_p = np.random.random(self.network.a2_dim) * 2 - 1
                        else:
                            a_p = self.network.choose_phase_action(o_p)
                        if self.train_model:    # 加噪声
                            a_p = np.clip(np.random.normal(0, self.var, size=a_p.shape) + a_p, -1, 1)
                        result = np.zeros_like(a_p)
                        result[np.argmax(a_p)] = 1.
                        self.a_p_list[light].append(result)
                        a_p = np.argmax(a_p)    # 4个里面值最大的
                    else:   # DQN
                        a_p = self.network.choose_phase_action(o_p)
                        self.a_p_list[light].append(a_p)

                next_phase = int(a_p)
                self.phase_list[light].append(next_phase)

                # Decide next green time
                if not self.use_time or (not self.train_model and not self.load_model):
                    a_t = np.array([0.])     # 经过下面的处理最终会得到20s
                else:
                    o_t = np.array(self.step_time_obs[light]).flatten().tolist()
                    self.o_t_list[light].append(o_t)       # 存最近两次决策obs(2, T*o_dim)
                    if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                        a_t = np.random.random(self.network.a1_dim) * 2 - 1
                    else:
                        a_t = self.network.choose_time_action(o_t)
                    if self.train_model:    # 加噪声
                        a_t = np.clip(np.random.normal(0, self.var, size=a_t.shape) + a_t, -1, 1)

                next_green = round((a_t[0] + 1) / 2 * (self.max_green - self.min_green) + self.min_green)
                self.green[light] = next_green
                self.a_t_list[light].append(a_t)

                if self.train_model:
                    reward = env.get_light_reward(light)
                    self.reward_list.append(reward)
                    if self.use_time and self.use_phase:
                        if len(self.o_t_list[light]) >= 2 and len(self.o_p_list[light]) >= 2:
                            self.network.store_transition(self.o_t_list[light][-2], self.o_p_list[light][-2],
                                                          self.a_t_list[light][-2], self.a_p_list[light][-2], reward,
                                                          self.o_t_list[light][-1], self.o_p_list[light][-1])
                    elif self.use_time:
                        if len(self.o_t_list[light]) >= 2:
                            self.network.store_transition(self.o_t_list[light][-2], self.a_t_list[light][-2], reward,
                                                          self.o_t_list[light][-1])
                    else:  # only phase or neither
                        if len(self.o_p_list[light]) >= 2:
                            self.network.store_transition(self.o_p_list[light][-2], self.a_p_list[light][-2], reward,
                                                          self.o_p_list[light][-1])
                    if self.train_model and self.pointer >= self.learn_begin:
                        self.var = max(0.01, self.var * 0.99)  # 0.9-40 0.99-400 0.999-4000
                        self.network.learn()

                if self.phase_list[light][-2] == self.phase_list[light][-1]:
                    skip_yr = self.yellow + self.red + self.green[light]
                    env.set_light_action(light, self.phase_list[light][-1] * 3, skip_yr)   # 本该亮黄灯的继续亮绿灯
                    self.time_index[light], self.color[light] = skip_yr, 'g'
                else:
                    env.set_light_action(light, self.phase_list[light][-2] * 3 + 1, self.yellow)
                    self.time_index[light], self.color[light] = self.yellow, 'y'

        self.time_index[light] -= 1
        return next_green, next_phase

    def reset(self):
        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: self.min_green for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []


class ManagerLightAgent:
    def __init__(self, light_id, config, get_worker_oa, worker_choose_action):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)
        self.get_worker_oa = get_worker_oa
        self.worker_choose_action = worker_choose_action

        self.use_time = config['use_time']
        self.use_phase = config['use_phase']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None
        self.lstm_observe_every_step = config['lstm_observe_every_step']

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配
        config['vehicle']['obs_dim'] = config['vehicle']['obs_dim'] + len(self.light_id)    # note: 临时增加了路口编号作为g_obs！！！

        if self.use_time and self.use_phase:
            self.light_opt = 'both'
            self.network = HATD3Triple(config)  # tpg
        elif not self.use_time and not self.use_phase:
            self.light_opt = 'neither'
            self.network = ManagerTD3(config)
        else:
            self.light_opt = 'phase' if self.use_phase else 'time'
            self.network = HATD3Double(config, self.light_opt)

        self.save = lambda path, ep: self.network.save(path + 'light_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else 99
            self.network.load('../model/' + config['load_model_name'] + '/light_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.o_t = config['time']['obs_dim']
        self.a_t = config['time']['act_dim']
        self.T_t = config['time']['T']
        self.o_p = config['phase']['obs_dim']
        self.a_p = config['phase']['act_dim']
        self.T_p = config['phase']['T']
        self.o_g = config['vehicle']['obs_dim']
        self.a_g = config['vehicle']['act_dim']
        self.T_g = config['vehicle']['T']

        self.min_green = config['min_green']
        self.max_green = config['max_green']
        self.yellow = config['yellow']
        self.red = config['red']

        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: config['min_green'] for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_goal_obs = {light: deque([[0] * self.o_g for _ in range(self.T_g)], maxlen=self.T_g) for light in self.light_id}
        self.o_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []
        self.accumulate_reward_manager = {light: [] for light in self.light_id}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env):
        tl, pl, gl = [], [], []
        for light in self.light_id:
            t, p, g = self._step(env, light)
            tl.append(t)
            pl.append(p)
            gl.append(g)
        return tl, pl, gl
        # return tl[0], pl[0], gl[0]  # 只向外展示第一个路口的动作

    def _step(self, env, light):
        if self.lstm_observe_every_step:
            o_t = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
            self.step_time_obs[light].append(o_t)                      # 存近10步obs(T, o_dim)
            o_p = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
            self.step_phase_obs[light].append(o_p)

        # remain_green = (self.green if self.color in 'yr' else self.time_index) / (4 * env.base_cycle_length)
        remain_green = (self.green[light] if self.color[light] in 'yr' else self.time_index[light]) / (env.base_cycle_length / 4)
        # goal只看变灯时刻状态不合理，应该每秒都看
        add_light_id = np.eye(len(self.light_id))[int(self.light_id.index(light))].tolist()
        o_g = add_light_id + np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_goal_obs(light) + [remain_green]
        self.step_goal_obs[light].append(o_g)
        self.accumulate_reward_manager[light].append(env.get_manager_fluency_reward(light))

        next_green, next_phase, vehicle_goal = None, None, None
        if self.time_index[light] == 0:
            if self.color[light] == 'y' and self.red != 0:  # 黄灯结束切红灯
                env.set_light_action(light, self.phase_list[light][-2] * 3 + 2, self.red)
                self.time_index[light], self.color[light] = self.red, 'r'
            elif self.color[light] == 'r' or (self.color[light] == 'y' and self.red == 0):  # 红灯结束或（黄灯结束且无全红相位）切绿灯
                env.set_light_action(light, self.phase_list[light][-1] * 3, self.green[light])
                self.time_index[light], self.color[light] = self.green[light], 'g'
            elif self.color[light] == 'g':
                if not self.lstm_observe_every_step:
                    o_t = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
                    self.step_time_obs[light].append(o_t)                      # 存近10步obs(T, o_dim)
                    o_p = np.eye(4)[int(self.phase_list[light][-1])].tolist() + env.get_light_obs(light)
                    self.step_phase_obs[light].append(o_p)

                # Choose next phase
                if not self.use_phase or (not self.train_model and not self.load_model):
                    a_p = (self.phase_list[light][-1] + 1) % 4     # 不控制时默认动作
                else:
                    o_p = np.array(self.step_phase_obs[light]).flatten().tolist()
                    self.o_p_list[light].append(o_p)       # 存最近两次决策obs(2, T*o_dim)

                    if self.train_model:  # 加噪声
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_p = np.random.random(self.a_p) * 2 - 1
                        else:
                            a_p = self.network.choose_phase_action(o_p)
                        a_p = np.clip(np.random.normal(0, self.var, size=a_p.shape) + a_p, -1, 1)
                    else:
                        a_p = self.network.choose_phase_action(o_p)
                    result = np.zeros_like(a_p)
                    result[np.argmax(a_p)] = 1.
                    self.a_p_list[light].append(result)
                    a_p = np.argmax(a_p)    # 4个里面值最大的
                next_phase = int(a_p)
                self.phase_list[light].append(next_phase)

                # Decide next green time
                if not self.use_time or (not self.train_model and not self.load_model):
                    a_t = np.array([0.])     # 经过下面的处理最终会得到20s
                else:
                    o_t = np.array(self.step_time_obs[light]).flatten().tolist()
                    self.o_t_list[light].append(o_t)       # 存最近两次决策obs(2, T*o_dim)

                    if self.train_model:    # 加噪声
                        if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                            a_t = np.random.random(self.a_t) * 2 - 1
                        else:
                            a_t = self.network.choose_time_action(o_t)
                        a_t = np.clip(np.random.normal(0, self.var, size=a_t.shape) + a_t, -1, 1)
                    else:
                        a_t = self.network.choose_time_action(o_t)
                    self.a_t_list[light].append(a_t)
                next_green = round((a_t[0] + 1) / 2 * (self.max_green - self.min_green) + self.min_green)
                self.green[light] = next_green

                # Decide next vehicle goal
                o_g = np.array(self.step_goal_obs[light]).flatten().tolist()
                self.o_g_list[light].append(o_g)

                if self.train_model:    # 加噪声
                    if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                        a_g = np.random.random(self.a_g) * 2 - 1
                    else:
                        a_g = self.network.choose_goal(o_g)
                    a_g = np.clip(np.random.normal(0, self.var, size=a_g.shape) + a_g, -1, 1)
                else:
                    a_g = self.network.choose_goal(o_g)
                self.a_g_list[light].append(a_g)
                # advice_speed = (a_g + 1) / 2    # [0,1]           ##################################错误地弄成了0,1
                vehicle_goal = a_g

                reward = env.get_light_reward(light)

                g_reward = sum(self.accumulate_reward_manager[light]) / len(self.accumulate_reward_manager[light]) if (
                        len(self.accumulate_reward_manager[light]) > 0) else 0      # 平均每秒每车道的停止车辆数
                self.accumulate_reward_manager[light] = []  # 是否应该考虑把waiting time 也用于goal的reward？有点麻烦，先不了

                worker_obs, worker_act = self.get_worker_oa()
                if self.light_opt == 'both':    # 3Actor
                    reward = (2 * reward + g_reward) / 3
                    if len(self.o_t_list[light]) >= 2 and len(self.o_p_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                        self.network.store_transition(self.o_t_list[light][-2], self.o_p_list[light][-2], self.o_g_list[light][-2],
                                                      self.a_t_list[light][-2], self.a_p_list[light][-2], self.a_g_list[light][-2], reward,
                                                      self.o_t_list[light][-1], self.o_p_list[light][-1], self.o_g_list[light][-1])
                elif self.light_opt == 'time':
                    reward = (reward + g_reward) / 2
                    if len(self.o_t_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                        self.network.store_transition(self.o_t_list[light][-2], self.o_g_list[light][-2],
                                                      self.a_t_list[light][-2], self.a_g_list[light][-2], reward,
                                                      self.o_t_list[light][-1], self.o_g_list[light][-1])
                elif self.light_opt == 'phase':
                    reward = (reward + g_reward) / 2
                    if len(self.o_p_list[light]) >= 2 and len(self.o_g_list[light]) >= 2:
                        self.network.store_transition(self.o_p_list[light][-2], self.o_g_list[light][-2],
                                                      self.a_p_list[light][-2], self.a_g_list[light][-2], reward,
                                                      self.o_p_list[light][-1], self.o_g_list[light][-1])
                else:   # self.light_opt == 'neither':   # only goalTD3
                    reward = g_reward
                    if len(self.o_g_list[light]) >= 2:
                        self.network.store_transition(self.o_g_list[light][-2], self.a_g_list[light][-2], reward,
                                                      self.o_g_list[light][-1], worker_obs, worker_act)
                self.reward_list.append(reward)

                if self.train_model and self.pointer >= self.learn_begin:
                    self.var = max(0.01, self.var * 0.99)  # 0.9-40 0.99-400 0.999-4000
                    self.network.learn(self.worker_choose_action)

                if self.phase_list[light][-2] == self.phase_list[light][-1]:
                    skip_yr = self.yellow + self.red + self.green[light]
                    env.set_light_action(light, self.phase_list[light][-1] * 3, skip_yr)   # 本该亮黄灯的继续亮绿灯
                    self.time_index[light], self.color[light] = skip_yr, 'g'
                else:
                    env.set_light_action(light, self.phase_list[light][-2] * 3 + 1, self.yellow)
                    self.time_index[light], self.color[light] = self.yellow, 'y'

        self.time_index[light] -= 1
        return next_green, next_phase, vehicle_goal

    def reset(self):
        self.time_index = {light: 0 for light in self.light_id}
        self.green = {light: self.min_green for light in self.light_id}
        self.color = {light: 'g' for light in self.light_id}
        self.phase_list = {light: deque([0], maxlen=2) for light in self.light_id}

        self.step_time_obs = {light: deque([[0] * self.o_t for _ in range(self.T_t)], maxlen=self.T_t) for light in self.light_id}
        self.o_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_t_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_phase_obs = {light: deque([[0] * self.o_p for _ in range(self.T_p)], maxlen=self.T_p) for light in self.light_id}
        self.o_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_p_list = {light: deque(maxlen=2) for light in self.light_id}
        self.step_goal_obs = {light: deque([[0] * self.o_g for _ in range(self.T_g)], maxlen=self.T_g) for light in self.light_id}
        self.o_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.a_g_list = {light: deque(maxlen=2) for light in self.light_id}
        self.reward_list = []
        self.accumulate_reward_manager = {light: [] for light in self.light_id}


"""
    车辆智能体
"""


class IndependentCavAgent:
    """理论上整个路网只用一个cav智能体，但时间所限不改这个了，目前是一个分区一个cav智能体。"""
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.ctrl_all_lane = not config['only_ctrl_curr_phase']
        self.ctrl_lane_num = 8 if self.ctrl_all_lane else 2  # 每个时刻控制的入口车道数。每一时刻都控制所有方向的车道
        self.ctrl_all_cav = not config['only_ctrl_head_cav']

        self.use_CAV = config['use_CAV']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        self.network = TD3Single(config, 'cav')
        self.save = lambda path, ep: self.network.save(path + 'cav_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else 99
            self.network.load('../model/' + config['load_model_name'] + '/cav_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.T = config['cav']['T']

        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def step(self, env, goal, next_phase):
        real, next_a = [], []
        if self.use_CAV:
            global_income_cav = []
            for light in self.light_id:
                if self.ctrl_all_cav:
                    curr_cav = [car for lane in env.light_get_lane(light) for car in env.lane_get_cav(lane, head_only=False)]
                else:
                    curr_cav = env.light_get_head_cav(light, self.next_phase[light], curr_phase=not self.ctrl_all_lane)
                global_income_cav.extend(curr_cav)
                self.ctrl_cav[light].append(curr_cav)
            self.global_income_cav.append(global_income_cav)
        for light_idx, light in enumerate(self.light_id):
            r, n = self._step(env, goal[light_idx], next_phase[light_idx], light)
            real.append(r)
            next_a.append(n)
        return real, next_a

    def _step(self, env, goal, next_phase, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            if next_phase is not None:    # 说明上层切相位了，接下来是一对新车道的yrg
                self.next_phase[light] = next_phase

            # 对比两时刻头CAV，上时刻还有现在没了(可能切相位或驶出)的要reset一下跟驰
            for cav_id in self.ctrl_cav[light][-2]:
                if cav_id is not None and cav_id not in self.global_income_cav[-1]:
                    env.reset_head_cav(cav_id)
                    self.reward_list.append(self.trans_buffer[cav_id]['reward'])

                    del self.trans_buffer[cav_id]

            for cav_id in self.ctrl_cav[light][-1]:
                if cav_id:  # cav is not None
                    o_v = env.get_head_cav_obs(cav_id)  # list

                    if cav_id not in self.trans_buffer:  # == 0
                        self.trans_buffer[cav_id] = {'obs': [o_v],  # 存储车辆每一步的obs
                                                     'action': [],  # 每一步的action
                                                     'real_acc': [],  # 每一步的action
                                                     'reward': []}
                    else:  # >=1
                        self.trans_buffer[cav_id]['obs'].append(o_v)

                    cav_obs = self.trans_buffer[cav_id]['obs']
                    if len(cav_obs) >= self.T:  # 没存满就先不控制
                        if self.train_model:  # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_v = np.random.random(self.network.a_dim) * 2 - 1
                            else:
                                a_v = self.network.choose_action(cav_obs[-self.T:])
                            a_v = np.clip(np.random.normal(0, self.var, size=a_v.shape) + a_v, -1, 1)
                        else:
                            a_v = self.network.choose_action(cav_obs[-self.T:])
                        self.trans_buffer[cav_id]['action'].append(a_v)
                        next_acc = a_v[0]    # [-1,1]
                        real_a = cav_obs[-1][2]    # [-?,1]
                        self.trans_buffer[cav_id]['real_acc'].append(real_a)   # 获取的是上一时步的实际acc

                        reward = env.get_cav_reward(cav_obs[-1], self.trans_buffer[cav_id]['real_acc'][-2],
                                                    self.trans_buffer[cav_id]['action'][-2]) if len(cav_obs) >= 1 + self.T else 0
                        self.trans_buffer[cav_id]['reward'].append(reward)

                        if self.train_model and len(cav_obs) >= self.T + 1:
                            self.network.store_transition(np.array(cav_obs[-self.T - 1: -1]).flatten(),
                                                          # self.trans_buffer[cav_id]['action'][-2],
                                                          self.trans_buffer[cav_id]['real_acc'][-1],    # 当前时刻的real_acc存的是上一时刻动作的真实效果
                                                          self.trans_buffer[cav_id]['reward'][-1],
                                                          np.array(cav_obs[-self.T:]).flatten())

                        env.set_head_cav_action(cav_id, cav_obs[-1][1], next_acc)
                        # print('cav obs:', cav_obs[-1], 'a:', next_acc, 'r:', self.trans_buffer[cav_id]['reward'][-1])
            if self.train_model and self.pointer >= self.learn_begin:
                self.var = max(0.01, self.var * 0.999)  # 0.9-40 0.99-400 0.999-4000
                self.network.learn()
        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}

        self.trans_buffer = {}
        self.reward_list = []


class WorkerCavAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.ctrl_all_lane = not config['only_ctrl_curr_phase']
        self.ctrl_lane_num = 8 if self.ctrl_all_lane else 2  # 每个时刻控制的入口车道数。每一时刻都控制所有方向的车道
        self.ctrl_all_cav = not config['only_ctrl_head_cav']

        self.use_CAV = config['use_CAV']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        config['memory_capacity'] = config['memory_capacity'] * len(self.light_id)  # 控制多路口会导致存速翻倍，故扩大容量以匹配

        self.network = WorkerTD3(config)
        self.save = lambda path, ep: self.network.save(path + 'cav_agent_' + self.holon_name + '_ep_' + str(ep))
        if self.load_model:
            load_ep = str(config['load_model_ep']) if config['load_model_ep'] else 99
            self.network.load('../model/' + config['load_model_name'] + '/cav_agent_' + self.holon_name + '_ep_' + load_ep)

        self.var = config['var']
        self.T = config['cav']['T']
        self.alpha = config['alpha']

        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}
        self.lane_speed = {light: deque([[1.] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.goal = {light: deque([], maxlen=2) for light in self.light_id}

        self.trans_buffer = {}
        self.ext_reward_list = []
        self.int_reward_list = []
        self.reward_list = []
        self.for_manager = {'obs': [], 'act': []}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def get_oa(self):
        self.for_manager = {'obs': [[[-1] * 8 for _ in range(8)] for _ in range(25)],
                            'act': [[[-1] for _ in range(8)] for _ in range(25)]}   # note: 注意，这里使得OPC始终无效！！！！！
        obs_seq = self.for_manager['obs']
        act_seq = self.for_manager['act']
        self.for_manager = {'obs': [], 'act': []}
        return obs_seq, act_seq

    def step(self, env, goal, next_phase):
        real, next_a = [], []
        if self.use_CAV:
            global_income_cav = []
            for light in self.light_id:
                if self.ctrl_all_cav:
                    curr_cav = [car for lane in env.light_get_lane(light) for car in env.lane_get_cav(lane, head_only=False)]
                else:
                    curr_cav = env.light_get_head_cav(light, self.next_phase[light], curr_phase=not self.ctrl_all_lane)
                global_income_cav.extend(curr_cav)
                self.ctrl_cav[light].append(curr_cav)
            self.global_income_cav.append(global_income_cav)
        for light_idx, light in enumerate(self.light_id):
            r, n = self._step(env, goal[light_idx], next_phase[light_idx], light)
            real.append(r)
            next_a.append(n)
        return real, next_a

    def _step(self, env, goal: np.ndarray, next_phase, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            curr_headcav = env.light_get_head_cav(light, self.next_phase[light], curr_phase=not self.ctrl_all_lane)

            # curr_lane = env.light_get_ctrl_lane(self.light_id, self.next_phase, curr_phase=not self.ctrl_all_lane)    # 与g_dims是2还是8有关
            curr_lane = env.light_get_ctrl_lane(light, self.next_phase[light], curr_phase=False)
            if goal is not None:    # 说明上层切相位了，接下来是一对新车道的yrg
                self.lane_speed[light] = deque([[env.lane_get_speed(lane) / env.max_speed for lane in curr_lane]], maxlen=2)
                self.goal[light] = deque([goal.tolist()], maxlen=2)
                self.next_phase[light] = next_phase
            else:
                self.lane_speed[light].append([env.lane_get_speed(lane) / env.max_speed for lane in curr_lane])
                # self.goal.append([self.lane_speed[-2][i] + self.goal[-1][i] - self.lane_speed[-1][i] for i in range(len(self.lane_speed[-1]))])
                self.goal[light].append([max(min((self.lane_speed[light][-2][i] + self.goal[light][-1][i] - self.lane_speed[light][-1][i]), 1), -1)
                                         for i in range(len(self.lane_speed[light][-1]))])  # clip0719

            # 对比两时刻头CAV，上时刻还有现在没了(可能切相位或驶出)的要reset一下跟驰
            for cav_id in self.ctrl_cav[light][-2]:
                if cav_id is not None and cav_id not in self.global_income_cav[-1]:
                    env.reset_head_cav(cav_id)
                    self.ext_reward_list.append(self.trans_buffer[cav_id]['ext_reward'])
                    self.int_reward_list.append(self.trans_buffer[cav_id]['int_reward'])
                    self.reward_list.append(self.trans_buffer[cav_id]['reward'])

                    del self.trans_buffer[cav_id]

            curr_all_lane_obs, curr_all_lane_act = [], []   # 用于保存8车道头车的o & a
            for cav_id in self.ctrl_cav[light][-1]:
                o_v = env.get_head_cav_obs(cav_id)  # list

                a_v_for_manager = -1
                if cav_id:  # cav is not None
                    if cav_id not in self.trans_buffer:  # == 0
                        self.trans_buffer[cav_id] = {'obs': [o_v],  # 存储车辆每一步的obs
                                                     'action': [],  # 每一步的action
                                                     'real_acc': [],  # 每一步的action
                                                     'goal': deque(maxlen=2),
                                                     'ext_reward': [],
                                                     'int_reward': [],
                                                     'reward': []}
                    else:  # >=1
                        self.trans_buffer[cav_id]['obs'].append(o_v)

                    cav_obs = self.trans_buffer[cav_id]['obs']
                    if len(cav_obs) >= self.T:  # 没存满就先不控制
                        g_v = self.goal[light][-1][curr_lane.index(env.cav_get_lane(cav_id))]  # goal is advice_lane_speed_delta
                        self.trans_buffer[cav_id]['goal'].append(g_v)

                        if self.train_model:  # 加噪声
                            if self.pointer < self.learn_begin and not self.load_model:  # 随机填充
                                a_v = np.random.random(self.network.a_dim) * 2 - 1
                            else:
                                a_v = self.network.choose_action(cav_obs[-self.T:], g_v)
                            a_v = np.clip(np.random.normal(0, self.var, size=a_v.shape) + a_v, -1, 1)
                        else:
                            a_v = self.network.choose_action(cav_obs[-self.T:], g_v)
                        self.trans_buffer[cav_id]['action'].append(a_v)
                        a_v_for_manager = a_v[0]
                        next_acc = a_v[0]    # [-1,1]
                        real_a = cav_obs[-1][2]    # [-?,1]
                        self.trans_buffer[cav_id]['real_acc'].append(real_a)   # 获取的是上一时步的实际acc

                        int_reward = -np.sqrt(np.sum(g_v ** 2))
                        ext_reward = env.get_cav_reward(cav_obs[-1], self.trans_buffer[cav_id]['real_acc'][-2],
                                                        self.trans_buffer[cav_id]['action'][-2]) if len(cav_obs) >= 1 + self.T else 0
                        reward = (1 - self.alpha) * ext_reward + self.alpha * int_reward
                        self.trans_buffer[cav_id]['int_reward'].append(int_reward)
                        self.trans_buffer[cav_id]['ext_reward'].append(ext_reward)
                        self.trans_buffer[cav_id]['reward'].append(reward)

                        if self.train_model and len(cav_obs) >= self.T + 1:
                            self.network.store_transition(np.array(cav_obs[-self.T - 1: -1]).flatten(),
                                                          # self.trans_buffer[cav_id]['action'][-2],
                                                          self.trans_buffer[cav_id]['real_acc'][-1],    # 当前时刻的real_acc存的是上一时刻动作的真实效果
                                                          self.trans_buffer[cav_id]['goal'][-2],
                                                          self.trans_buffer[cav_id]['goal'][-1],    # next_goal
                                                          self.trans_buffer[cav_id]['reward'][-1],
                                                          np.array(cav_obs[-self.T:]).flatten())

                        env.set_head_cav_action(cav_id, cav_obs[-1][1], next_acc)
                        # print('cav obs:', cav_obs[-1], 'a:', next_acc, 'r:', self.trans_buffer[cav_id]['reward'][-1])

                if cav_id is not None and cav_id in curr_headcav:   # 若该车为头车，传出状态给opc
                    curr_all_lane_obs.append(o_v)
                    curr_all_lane_act.append([a_v_for_manager])
            for i in range(8 - len(curr_all_lane_act)):
                curr_all_lane_obs.append([-1]*8)
                curr_all_lane_act.append([-1])
            self.for_manager['obs'].append(curr_all_lane_obs)
            self.for_manager['act'].append(curr_all_lane_act)

            if self.train_model and self.pointer >= self.learn_begin:
                self.var = max(0.01, self.var * 0.999)  # 0.9-40 0.99-400 0.999-4000
                self.network.learn()

        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.ctrl_cav = {light: deque([[None] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.global_income_cav = deque([[], []], maxlen=2)
        self.next_phase = {light: 1 for light in self.light_id}
        self.lane_speed = {light: deque([[1.] * self.ctrl_lane_num], maxlen=2) for light in self.light_id}
        self.goal = {light: deque([], maxlen=2) for light in self.light_id}

        self.trans_buffer = {}
        self.ext_reward_list = []
        self.int_reward_list = []
        self.reward_list = []
        self.for_manager = {'obs': [], 'act': []}


class LoyalCavAgent:
    def __init__(self, light_id, config):
        if isinstance(light_id, str):
            self.holon_name = light_id
            self.light_id = [light_id]
        elif isinstance(light_id, (list, tuple)):
            self.holon_name = 'h_' + light_id[0]
            self.light_id = list(light_id)

        self.network = WorkerTD3(config)
        self.save = lambda path, ep: print('Loyal Agent no need to save')

        self.use_CAV = config['use_CAV']
        self.train_model = config['train_model']
        self.load_model = config['load_model_name'] is not None

        self.var = config['var']
        self.T = config['cav']['T']

        self.last_car_list = {light: [] for light in self.light_id}
        self.target_speed = {light: [] for light in self.light_id}
        self.reward_list = []
        self.for_manager = {'obs': [[[-1] * 8 for _ in range(8)] for _ in range(25)],
                            'act': [[[-1] for _ in range(8)] for _ in range(25)]}

    @property
    def pointer(self):
        return self.network.pointer

    @property
    def learn_begin(self):
        return self.network.learn_begin

    def get_oa(self):
        obs_seq = self.for_manager['obs']
        act_seq = self.for_manager['act']
        # self.for_manager = {'obs': [], 'act': []}
        return obs_seq, act_seq

    def step(self, env, goal, next_phase):
        for light in self.light_id:
            self._step(env, goal, next_phase, light)
        return [None] * len(self.light_id), [None] * len(self.light_id)

    def _step(self, env, goal, next_phase, light):
        next_acc, real_a = None, None

        if self.use_CAV:
            curr_car, curr_tar_v = [], []
            for lid, lane in enumerate(env.light_get_lane(light)):  # 必控制所有车道所有车(无论是不是CAV)
                lane_car = env.lane_get_all_car(lane)
                curr_car.extend(lane_car)
                lane_speed = env.lane_get_speed(lane) / env.max_speed

                if goal is not None:
                    # 注意，worker中g_v是直接传入网络，每一步递推goal时goal值域始终是[-1,1]，而这里的需要直接根据v+dv算出上层实际希望各车道的车速，而车速不能为负
                    target_v = max(min((goal[lid] + lane_speed), 1), 0) * env.max_speed
                    curr_tar_v.append(target_v)
                else:
                    curr_tar_v.append(self.target_speed[light][lid])

                for car in lane_car:
                    env.set_lane_act_speed(car, curr_tar_v[-1])

            self.target_speed[light] = curr_tar_v if goal is not None else self.target_speed[light]

            for vehicle in self.last_car_list[light]:
                if vehicle not in curr_car:
                    env.reset_head_cav(vehicle)
            self.last_car_list[light] = curr_car

        # self.for_manager['obs'].append([[-1] * 8 for _ in range(8)])
        # self.for_manager['act'].append([[-1] for _ in range(8)])

        return (real_a, next_acc) if not real_a or not next_acc else (real_a * env.max_acc, next_acc * env.max_acc)

    def reset(self):
        self.last_car_list = {light: [] for light in self.light_id}
        self.target_speed = {light: [] for light in self.light_id}
        self.reward_list = []
        self.for_manager = {'obs': [[[-1] * 8 for _ in range(8)] for _ in range(25)],
                            'act': [[[-1] for _ in range(8)] for _ in range(25)]}
