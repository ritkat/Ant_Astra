from pybullet_envs.gym_locomotion_envs import AntBulletEnv, HalfCheetahBulletEnv, Walker2DBulletEnv, HumanoidBulletEnv, HopperBulletEnv, WalkerBaseBulletEnv
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
import numpy as np
import pybullet as p
from pybullet_envs.robot_locomotors import Hopper, Walker2D, HalfCheetah, Ant, Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder


class QDAntBulletEnv(AntBulletEnv):
    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(4)])
        self.desc_acc = np.array([0.0 for _ in range(4)])

        # print(f"The behavioural desciptor is {len(self.desc)}-dimentional",
        #       f"and defined as proportion of feet contact time with the ground in the order {self.robot.foot_list}")


    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(4)])
        self.desc_acc = np.array([0.0 for _ in range(4)])

        # p.setGravity(0, 0, -9.81)  # Reset gravity to default

        return r

    
    def step(self, a):
        # do this 100 times
        state, reward, done, info = super().step(a)
        self.desc_acc += self.robot.feet_contact
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info
    
    def change_env(self, xgrav,ygrav,zgrav):
       p.setGravity(xgrav, ygrav, zgrav)       

class WalkerBaseBulletEnv_fix(WalkerBaseBulletEnv):

  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=13.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene

class WalkerBaseBulletEnv_grav(WalkerBaseBulletEnv):

  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=13.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene
  
class WalkerBaseBulletEnv_move_target(WalkerBaseBulletEnv):

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
    ))  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    debugmode = 0
    if (debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [
        self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
    ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), {}

  def camera_adjust(self):
    x, y, z = self.robot.body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)

class AntBulletEnv_grav(WalkerBaseBulletEnv_grav):

  def __init__(self, render=False):
    self.robot = Ant()
    WalkerBaseBulletEnv_grav.__init__(self, self.robot, render)


class QDAntBulletEnv_grav(AntBulletEnv_grav):
    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(4)])
        self.desc_acc = np.array([0.0 for _ in range(4)])

        # print(f"The behavioural desciptor is {len(self.desc)}-dimentional",
        #       f"and defined as proportion of feet contact time with the ground in the order {self.robot.foot_list}")


    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(4)])
        self.desc_acc = np.array([0.0 for _ in range(4)])

        # p.setGravity(0, 0, -9.81)  # Reset gravity to default

        return r

    
    def step(self, a):
        # do this 100 times
        state, reward, done, info = super().step(a)
        self.desc_acc += self.robot.feet_contact
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info
    
class QDHalfCheetahBulletEnv(HalfCheetahBulletEnv):
    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        print(f"The behavioural desciptor is {len(self.desc)}-dimentional",
              f"and defined as proportion of feet contact time with the ground in the order {[self.robot.foot_list[0] , self.robot.foot_list[3]]}")


    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        return r

    
    def step(self, a):
        state, reward, done, info = super().step(a)
        self.desc_acc[0] += self.robot.feet_contact[0]
        self.desc_acc[1] += self.robot.feet_contact[3]
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info
    
