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
        self.log_obs(obs)
        self.file.flush()

    def log_obs(self, obs):
        np.save(self.file, obs["images"][0], allow_pickle=False, fix_imports=False)

    def end_episode(self, discard=False):
        self.file.close()
        self.file = None
        if discard:
            os.remove(self.file_name)

class SimpleHdf5Writer(Writer):
    def __init__(self, file_name_template, file_path="trajectories"):
        super().__init__(file_name_template, file_path)
        self.episode_data = None

    def start_episode(self, obs):
        super().start_episode(obs)
        self.episode_data = {}
        for k, v in obs.items():
            self.episode_data[k] = [v]
        self.episode_data['actions'] = []
        self.episode_data['rewards'] = []
        self.episode_data['dones'] = []

    def log(self, act, obs, rew, done, info):
        if not self.episode_data:
            return

        super().log(act, obs, rew, done, info)
        for k, v in obs.items():
            self.episode_data[k].append(v)
        for k, v in info.items():
            vals = self.episode_data.get(k, [])
            vals.append(v)
            self.episode_data[k] = vals
        self.episode_data.get('actions', []).append(act)
        self.episode_data.get('rewards', []).append(rew)
        self.episode_data.get('dones', []).append(done)

    def end_episode(self, discard=False):
        if not discard:
            file_name = os.path.join(self.path, f"{self.file_name_template}_{self.episode_count}_{time.time()}.hdf5")
            with h5py.File(file_name, 'w') as f:
                for k, v in self.episode_data.items():
                    f.create_dataset(k, data=np.stack(v))
        self.episode_data = None


