
# -*- coding:utf-8 -*-
import time

from torch.utils.tensorboard import SummaryWriter

import utils
from agent import IndependentLightAgent, ManagerLightAgent, IndependentCavAgent, WorkerCavAgent
from configs import env_configs, get_agent_configs
from environment import Environment

env_configs['single']['sumocfg_path'] = '../sumo_sim_env/collision_env_tmp2.sumocfg'    # 防止两边同时运行修改时撞车
to_be_tested = '0726_various_rou'
TEST_ROU_PATH = 'single/rou_test/'


def setting(base_key, change):
    experience_cfg_base = {
        'baseline': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': None, },
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'T': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/T', },
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'P': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/P', },
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'V': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': None, },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/V', }}},
        'TV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/TV', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/TV', }}},
        'PV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/PV', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/PV', }}},
        'tp': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/tp', },
                            'cav': {'use_CAV': False,
                                    'train_model': False,
                                    'load_model_name': None, }}},
        'tpV': {
            'use_HRL': False,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/tpV', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/tpV', }}},
        'Gv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/Gv', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/Gv', }}},
        'tgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': False,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/tgv', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/tgv', }}},
        'pgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': False,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/pgv', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/pgv', }}},
        'tpgv': {
            'use_HRL': True,
            'modify_dict': {'light': {'use_time': True,
                                      'use_phase': True,
                                      'train_model': False,
                                      'load_model_name': to_be_tested + '/tpgv', },
                            'cav': {'use_CAV': True,
                                    'train_model': False,
                                    'load_model_name': to_be_tested + '/tpgv', }}},
    }
    return utils.change_dict(experience_cfg_base[base_key], {'modify_dict': change})


experience_cfg = {
    # # Note：Done


    # # Note：Doing
    'baseline': setting('baseline', {}),


    # # Note: To do
    # 'T': setting('T', {}),
    # 'P': setting('P', {}),
    # 'V': setting('V', {}),
    # 'tp': setting('tp', {}),
    # 'tpV': setting('tpV', {}),
    # 'tpgv': setting('tpgv', {}),
}


def launch_test(exp_cfg, single_flag=True):
    light_configs, cav_configs = get_agent_configs(exp_cfg['modify_dict'])

    experiment_name = exp_cfg['experiment_name']
    writer = SummaryWriter('../log/' + experiment_name)

    env_configs['single']['rou_path'] = TEST_ROU_PATH
    env = Environment(env_configs, single_flag)
    light_id_list = env.get_light_id()

    if exp_cfg['use_HRL']:  # Loyal仅用于训练上层，不能真正用于控制
        cav_agent = dict([(light_id, WorkerCavAgent(light_id, cav_configs)) for light_id in light_id_list])
        light_agent = dict([(light_id, ManagerLightAgent(light_id, light_configs, cav_agent[light_id].get_oa,
                                                         cav_agent[light_id].network.policy)) for light_id in light_id_list])
    else:
        light_agent = dict([(light_id, IndependentLightAgent(light_id, light_configs)) for light_id in light_id_list])
        cav_agent = dict([(light_id, IndependentCavAgent(light_id, cav_configs)) for light_id in light_id_list])

    utils.txt_save('../log/' + str(experiment_name) + '/configs',
                   {'env': env_configs, 'light': light_configs, 'cav': cav_configs})

    evaluate_index = ['wait', 'halt', 'emission', 'fuel', 'speed', 'timeloss', 'collision']
    ep_performance = {_: [] for _ in evaluate_index}
    for episode in range(15):
        rou_file_num = episode + 1
        print("Ep:", episode, "File:", env.rou_path, rou_file_num, '\t', time.strftime("%Y-%m-%d %H:%M:%S"))
        env.start_env(False, n_file=rou_file_num)

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

            print('\r', t, flush=True, end='')

        ep_wait = sum(waiting_time)
        ep_halt = sum(halting_num)
        ep_fuel = sum(fuel_consumption)
        ep_emission = sum(emission)
        ep_timeloss = sum(time_loss)
        ep_speed = sum(mean_speed) / len(mean_speed)
        ep_collision = env.collision_count

        print('\n', episode,
              '\n\twait=', ep_wait, '\thalt=', ep_halt,
              '\tspeed=', ep_speed, '\tcollision=', ep_collision,
              '\temission=', ep_emission, '\tfuel_consumption=', ep_fuel, '\ttime_loss=', ep_timeloss)

        writer_scalar_name = ['waiting time', 'halting count', 'carbon emission', 'fuel consumption',
                              'average speed', 'time loss', 'collision']
        ep_list = [ep_wait, ep_halt, ep_emission, ep_fuel, ep_speed, ep_timeloss, ep_collision]
        for i in range(len(evaluate_index)):
            ep_performance[evaluate_index[i]].append(ep_list[i])
            writer.add_scalar(writer_scalar_name[i], ep_list[i], episode)
        utils.txt_save('../log/' + experiment_name + '/performance_index', ep_performance)

        # 重置智能体内暂存的列表
        for light_id in light_id_list:
            light_agent[light_id].reset()
            cav_agent[light_id].reset()
        env.end_env()


if __name__ == "__main__":
    for key in experience_cfg:
        experience_cfg[key]['experiment_name'] = 'test/' + to_be_tested + '/' + key
        print(experience_cfg[key]['experiment_name'], 'start testing')
        launch_test(experience_cfg[key], single_flag=True)

