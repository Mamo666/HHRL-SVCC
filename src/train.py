
# -*- coding:utf-8 -*-
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils
from agent import IndependentLightAgent, ManagerLightAgent, IndependentCavAgent, WorkerCavAgent, LoyalCavAgent
from configs import env_configs, get_agent_configs
from environment import Environment

np.random.seed(3407)  # 设置随机种子


series_name = '0728_rou4'
MAX_EPISODES = 70  # 训练轮数
SUMO_GUI = False


def setting(base_key, change):
    experience_cfg_base = {
        'T': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': False, }}},
        'P': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': False, }}},
        'V': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'TV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'PV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'tp': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': False, }}},
        'tpV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'Gv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'tgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False, },
                            'cav': {'use_CAV': True, }}},
        'pgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
        'tpgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True, },
                            'cav': {'use_CAV': True, }}},
    }
    return utils.change_dict(experience_cfg_base[base_key], {'modify_dict': change})


experience_cfg = {
    # # Note：Done
    # 'Gv_noOPC_loyal_pretrain_manager': setting('Gv', {}),
    # 'Gv_noOPC_worker_pretrained_manager': setting('Gv', {
    #     'light': {'load_model_name': '0721_new_rou/Gv_noOPC_loyal_pretrain_manager'}}),
    # 'V_all8cav': setting('V', {'cav': {'only_ctrl_head_cav': False}}),
    # 'V_head8cav': setting('V', {'cav': {'only_ctrl_head_cav': True}}),
    # 'V_head8cav_T2': setting('V', {'cav': {'only_ctrl_head_cav': True, 'cav': {'T': 2}}}),
    # 'V_all8cav_T2': setting('V', {'cav': {'only_ctrl_head_cav': False, 'cav': {'T': 2}}}),
    # 'Gv_noOPC_worker_pretrained_manager': setting('Gv', {
    #     'light': {'train_model': False, 'load_model_name': '0721_new_rou/Gv_noOPC_loyal_pretrain_manager'},
    #     'cav': {'alpha': 1.}}),
    # 'Gv_noOPC_worker_pretrained_manager_head2cav__correct': setting('Gv', {
    #     'light': {'train_model': False, 'load_model_name': '0721_new_rou/Gv_noOPC_loyal_pretrain_manager'},
    #     'cav': {'alpha': 1., 'only_ctrl_curr_phase': True, 'only_ctrl_head_cav': True}}),  # memory size changed
    # 'Gv_noOPC_worker_pretrained_manager_head8cav__correct': setting('Gv', {
    #     'light': {'train_model': False, 'load_model_name': '0721_new_rou/Gv_noOPC_loyal_pretrain_manager'},
    #     'cav': {'alpha': 1., 'only_ctrl_curr_phase': False, 'only_ctrl_head_cav': True}}),  # memory size changed
    # 'Gv_noOPC_worker_pretrained_manager_head8cav_T2__correct': setting('Gv', {
    #     'light': {'train_model': False, 'load_model_name': '0721_new_rou/Gv_noOPC_loyal_pretrain_manager'},
    #     'cav': {'cav': {'T': 2}, 'alpha': 1., 'only_ctrl_curr_phase': False, 'only_ctrl_head_cav': True}}),
    # 'Gv_noOPC_worker_pretrained_manager_head8cav_T2_alpha05_correct': setting('Gv', {
    #     'light': {'train_model': False, 'load_model_name': '0721_new_rou/Gv_noOPC_loyal_pretrain_manager'},
    #     'cav': {'cav': {'T': 2}, 'alpha': .5, 'only_ctrl_curr_phase': False, 'only_ctrl_head_cav': True}}),

    # # Note：Doing
    # 'T': setting('T', {}),
    # 'tp': setting('tp', {}),
    # 'tpgv_noOPC_loyal_pretrain_manager': setting('tpgv', {}),
    'Gv_noOPC_loyal_pretrain_manager': setting('Gv', {}),

    # # Note: To do
    # 'tpgv_noOPC_worker_pretrained_manager': setting('tpgv', {
    #     'light': {'load_model_name': '0721_new_rou/tpgv_noOPC_loyal_pretrain_manager'}})
}
# flow_feat_id_list = [0, 1, 2, 3, 4, 5]
flow_feat_id_list = [4]


def launch_experiment(exp_cfg, save_model=True, single_flag=True, flow_feat_id=None):
    global MAX_EPISODES, SUMO_GUI

    exp_cfg['turn_on_gui_after_learn_start'] = True
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'])

    experiment_name = exp_cfg['experiment_name']
    writer = SummaryWriter('../log/' + experiment_name)  # './log/240226light_only'
    model_dir = '../model/' + experiment_name + '/'
    env = Environment(env_configs, single_flag)
    light_id_list = env.get_light_id()

    if exp_cfg['use_HRL']:
        if 'loyal' in exp_cfg['experiment_name']:   # 方便起见，以检索实验名中有无loyal字段来判断cav是否使用loyal
            cav_agent = dict([(light_id, LoyalCavAgent(light_id, cav_configs)) for light_id in light_id_list])
        else:
            cav_agent = dict([(light_id, WorkerCavAgent(light_id, cav_configs)) for light_id in light_id_list])
        light_agent = dict([(light_id, ManagerLightAgent(light_id, light_configs, cav_agent[light_id].get_oa,
                                                         cav_agent[light_id].network.policy)) for light_id in light_id_list])
    else:
        light_agent = dict([(light_id, IndependentLightAgent(light_id, light_configs)) for light_id in light_id_list])
        cav_agent = dict([(light_id, IndependentCavAgent(light_id, cav_configs)) for light_id in light_id_list])

    utils.txt_save('../log/' + str(experiment_name) + '/configs',
                   {'env': env_configs, 'light': light_configs, 'cav': cav_configs})

    for episode in range(MAX_EPISODES):
        # rou_file_num = np.random.randint(1, 16)  # 随机选取一个训练环境
        rou_file_num = np.random.randint(flow_feat_id * 5 + 1, flow_feat_id * 5 + 6)  # 随机选取一个训练环境
        print("Ep:", episode, "File:", env.rou_path, rou_file_num, '\t', time.strftime("%Y-%m-%d %H:%M:%S"))
        if (light_agent[light_id_list[0]].pointer > light_agent[light_id_list[0]].learn_begin and
                cav_agent[light_id_list[0]].pointer > cav_agent[light_id_list[0]].learn_begin):
            SUMO_GUI = exp_cfg['turn_on_gui_after_learn_start']
        env.start_env(SUMO_GUI, n_file=rou_file_num)

        waiting_time, halting_num, emission, fuel_consumption, mean_speed, time_loss = [], [], [], [], [], []

        for t in range(3000):
            for light_id in light_id_list:
                if light_agent[light_id].__class__.__name__ == 'ManagerLightAgent':
                    l_t, l_p, goal = light_agent[light_id].step(env)
                else:   # 'IndependentLightAgent'
                    l_t, l_p = light_agent[light_id].step(env)
                    goal = None
                real_a, v_a = cav_agent[light_id].step(env, goal, l_p)

                if l_t is not None:
                    writer.add_scalar('green time/' + str(episode), l_t, t)
                if l_p is not None:
                    writer.add_scalar('next phase/' + str(episode), l_p, t)
                if goal is not None:
                    writer.add_scalar('advice speed lane0/' + str(episode), goal[0] * env.max_speed, t)
                    # print(goal * env.max_speed)
                if v_a is not None:
                    writer.add_scalar('head CAV action/' + str(episode), v_a, t)
                    writer.add_scalar('head CAV acc_real/' + str(episode), real_a, t)

            env.step_env()

            if t % 10 == 0:  # episode % 10 == 9 and
                w, h, e, f, v, timeLoss = env.get_performance()
                waiting_time.append(w)
                halting_num.append(h)
                emission.append(e)
                fuel_consumption.append(f)
                mean_speed.append(v)
                time_loss.append(timeLoss)

            print('\r', t, '\t', light_agent[light_id_list[0]].pointer, cav_agent[light_id_list[0]].pointer, flush=True, end='')

        ep_light_r = sum(sum(light_agent[light_id].reward_list) for light_id in light_id_list)
        ep_cav_r = sum(sum(sum(sublist) for sublist in cav_agent[light_id].reward_list) for light_id in light_id_list)
        ep_wait = sum(waiting_time)
        ep_halt = sum(halting_num)
        ep_emission = sum(emission)
        ep_fuel = sum(fuel_consumption)
        ep_speed = sum(mean_speed) / len(mean_speed)
        ep_timeloss = sum(time_loss)

        writer.add_scalar('light reward', ep_light_r, episode)
        writer.add_scalar('cav reward', ep_cav_r, episode)
        writer.add_scalar('waiting time', ep_wait, episode)
        writer.add_scalar('halting count', ep_halt, episode)
        writer.add_scalar('carbon emission', ep_emission, episode)
        writer.add_scalar('fuel consumption', ep_fuel, episode)
        writer.add_scalar('average speed', ep_speed, episode)
        writer.add_scalar('time loss', ep_timeloss, episode)
        writer.add_scalar('collision', env.collision_count, episode)

        print('\n', episode,
              '\n\tlight:\tpointer=', light_agent[light_id_list[0]].pointer, '\tvar=', light_agent[light_id_list[0]].var, '\treward=', ep_light_r,
              '\n\tcav:\tpointer=', cav_agent[light_id_list[0]].pointer, '\tvar=', cav_agent[light_id_list[0]].var, '\treward=', ep_cav_r,
              '\n\twait=', ep_wait, '\thalt=', ep_halt,
              '\tspeed=', ep_speed, '\tcollision=', env.collision_count,
              '\temission=', ep_emission, '\tfuel_consumption=', ep_fuel, '\ttime_loss=', ep_timeloss)

        # 重置智能体内暂存的列表, 顺便实现每10轮存储一次模型参数
        for light_id in light_id_list:
            light, cav = light_agent[light_id], cav_agent[light_id]
            if save_model:
                utils.mkdir(model_dir)
                if episode % 10 == 9:
                    light.save(model_dir + 'light_agent_' + light_id + '_ep_' + str(episode))
                    cav.save(model_dir + 'cav_agent_' + light_id + '_ep_' + str(episode))
            light.reset()
            cav.reset()
        env.end_env()


if __name__ == "__main__":
    for ffi in flow_feat_id_list:
        for key in experience_cfg:
            series_name = series_name + '/' if series_name[-1] != '/' else series_name
            experience_cfg[key]['experiment_name'] = series_name + key + '_' + str(ffi)
            print(experience_cfg[key]['experiment_name'], 'start running')
            launch_experiment(experience_cfg[key], save_model=True, single_flag=True, flow_feat_id=ffi)

