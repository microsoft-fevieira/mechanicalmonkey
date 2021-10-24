import h5py
import numpy as np
import os
import time

class Writer:
    def __init__(self, file_name_template, file_path="trajectories"):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.path = file_path
        self.file_name_template = file_name_template
        self.episode_count = 0
        self.step = 0

    def start_episode(self, obs):
        self.episode_count += 1
        self.step = 0

    def log(self, act, obs, rew, done, info):
        self.step += 1

    def end_episode(self, discard=False):
        pass


class SimpleNpyWriter(Writer):
    def __init__(self, file_name_template, file_path="trajectories"):
        super().__init__(file_name_template, file_path)
        self.file = None

    def start_episode(self, obs):
        if self.file:
            self.end_episode(discard=True)
        super().start_episode(obs)
        self.file_name = os.path.join(self.path, f"{self.file_name_template}_{self.episode_count}_{time.time()}.npy")
        self.file = open(self.file_name, "wb")
        self.log_obs(obs)

    def log(self, act, obs, rew, done, info):
        if not self.file:
            return
        super().log(act, obs, rew, done, info)
        np.save(self.file, np.array(act), allow_pickle=False, fix_imports=False)
        np.save(self.file, np.array([rew, done, info["success"]]), allow_pickle=False, fix_imports=False)
        self.log_obs(obs)
        self.file.flush()

    def log_obs(self, obs):
        np.save(self.file, obs["cameras"][0], allow_pickle=False, fix_imports=False)

    def end_episode(self, discard=False):
        self.file.close()
        self.file = None
        if discard:
            os.remove(self.file_name)

def readSimpleNpy(file_name):
    episode = {}
    episode['images'] = []
    episode['actions'] = []
    episode['rewards'] = []
    episode['dones'] = []
    episode['successes'] = []
    with open(file_name, 'rb') as f:
        try:
            while(True):
                episode['images'].append(np.load(f, allow_pickle=True, fix_imports=False))
                episode['actions'].append(np.load(f, allow_pickle=True, fix_imports=False))
                rew, done, success = np.load(f, allow_pickle=True, fix_imports=False)
                episode['rewards'].append(rew)
                episode['dones'].append(done)
                episode['successes'].append(success)
        except IOError:
            pass

    return episode


class RobosuiteWriter(Writer):
    def __init__(self, file_name_template, file_path="trajectories"):
        super().__init__(file_name_template, file_path)
        self.episode_data = None

    def start_episode(self, obs):
        super().start_episode(obs)
        self.episode_data = {}
        self.episode_data['images'] = []
        self.episode_data['proprios'] = []
        self.episode_data['actions'] = []
        self.episode_data['rewards'] = []
        self.episode_data['dones'] = []
        self.episode_data['successes'] = []

    def log(self, act, obs, rew, done, info):
        if not self.episode_data:
            return

        super().log(act, obs, rew, done, info)
        self.episode_data['images'].append(obs["image"])
        self.episode_data['proprios'].append(obs["proprio"])
        self.episode_data['actions'].append(act)
        self.episode_data['rewards'].append(rew)
        self.episode_data['dones'].append(done)
        self.episode_data['successes'].append(info["success"])

    def end_episode(self, discard=False):
        if not discard:
            file_name = os.path.join(self.path, f"{self.file_name_template}_{self.episode_count}_{time.time()}.hdf5")
            with h5py.File(file_name, 'w') as f:
                for k, v in self.episode_data.items():
                    f.create_dataset(k, data=np.stack(v))
        self.episode_data = None

def readRobosuite(file_name):
    episode = {}
    with h5py.File(file_name, 'r') as f:
        for k in f.keys():
            episode[k] = np.array(f.get(k))

    return episode
