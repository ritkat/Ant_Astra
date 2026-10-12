import numpy as np
import gymnasium as gym
# from Ant_quant import *
env = gym.make('Ant-v5')
import torch as th
import zipfile
import imageio
import io
import mujoco
from scipy.spatial.transform import Rotation as R
# from compression_utils import *

from SPI_Interface_DroneLab import crazy_processor
import time

RUN_CONTINUOUS = False

astra = crazy_processor(
    instruction_file="~/Ant_Astra/instr_mem_bin.txt",
    parameter_file="~/Ant_Astra/param_mem_bin.txt",
)

# from Ant_quant import *
# model = PPO.load("/home/ritwik/MuJoCo_Quant/logs_ant/best_model_fixscale_1/best_model", env=env)

save_wt_scales_activations = True
NEURONS = 64
cfg = {
    "quant_act": True,
    "quant_weights": True,
    "activation": "relu",
}
# env = gym.make("Ant-v5")
# model_path = "/home/ritwik/MuJoCo_Quant/logs_ant_new/_fixscale_64_20250831_141022/Quant_ant_69.zip"



# def forward_pass(obs):
#     obs_quant = torch.round(torch.tensor(obs)*2**e0/m0).to(torch.float32)
#     obs_quant = torch.clip(obs_quant, -127, 127)
#     a1 = torch.round((torch.matmul(obs_quant,w1.T)+b1)*m1/2**e1)
#     # a1 = torch.relu(a1)
#     a1 = torch.clip(a1, 0, 255)
#     a2 = torch.round((torch.matmul(a1,w2.T)+b2)*m2/2**e2)
#     # a2 = torch.relu(a2)
#     a2 = torch.clip(a2, 0, 255)
#     a3 = torch.round((torch.matmul(a2,w3.T)+b3)*m3/2**e3)
#     a3 = torch.clip(a3, -127, 127)

#     return a3

# def forward_pass_astra(obs):
#     obs_quant = torch.round(torch.tensor(obs)*2**e0/m0).to(torch.float32)
#     obs_quant = torch.clip(obs_quant, -127, 127)
#     return astra.processor_run(obs_quant.cpu().numpy(), startAddress=232, outputLength=10)                      

# def save_forward_pass(obs):
#     obs_quant = torch.round(torch.tensor(obs)*2**e0/m0).to(torch.float32)
#     obs_quant = torch.clip(obs_quant, -127, 127)
#     a1 = torch.round((torch.matmul(obs_quant,w1.T)+b1)*m1/2**e1)
#     # a1 = torch.relu(a1)
#     a1 = torch.clip(a1, 0, 255)
#     a2 = torch.round((torch.matmul(a1,w2.T)+b2)*m2/2**e2)
#     # a2 = torch.relu(a2)
#     a2 = torch.clip(a2, 0, 255)
#     a3 = torch.round((torch.matmul(a2,w3.T)+b3)*m3/2**e3)
#     a3 = torch.clip(a3, -127, 127)

#     return obs_quant, a1, a2, a3, w1, w2, w3, b1, b2, b3, m0, m1, m2, m3, e0, e1, e2, e3




# def batch_frexp_refactored(inputs):
#     """
#     Decompose the scaling factor into mantissa and twos exponent, ensuring:
#     - Mantissa lies in [0, 2^16)
#     - Exponent remains meaningful (adjusted to match the scaled mantissa)

#     Parameters:
#     ----------
#     inputs: Tensor
#         Scaling factor

#     Returns:
#     -------
#     mantissa: Tensor
#     exponent: Tensor
#     """
#     shape_of_input = inputs.size()

#     # Flatten the input tensor to a 1D tensor
#     inputs = inputs.view(-1)

#     # Decompose into mantissa and exponent
#     output_m, output_e = np.frexp(inputs.cpu().numpy())

#     # Prepare adjusted mantissa and exponent lists
#     tmp_m = []
#     tmp_e = []

#     # Define the bias to scale mantissa to [0, 2^16)
#     scale_factor = 16  # 2^16

#     for idx, (m, e) in enumerate(zip(output_m, output_e)):
#         # Scale mantissa from [0.5, 1) to [0, 2^16)
#         int_m_shifted = int(Decimal(m * (2 ** scale_factor)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))

#         # Adjust the exponent to account for the scaling of the mantissa
#         e -= scale_factor
#         e= e*-1
#         e = max(0, min(255, e))
#         # Ensure mantissa and exponent are within valid bounds
#         if int_m_shifted >= 2 ** 16:  # Cap mantissa at maximum (65535)
#             int_m_shifted = 2 ** 16 - 1

#         elif inputs[idx] >= 65535:
#             int_m_shifted = 65535
#             e = 0

#         tmp_m.append(int_m_shifted)
#         tmp_e.append(e)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # tensor = torch.tensor(0.).to(device)

#     return torch.from_numpy(np.array(tmp_m)).to(device).view(shape_of_input), \
#            torch.from_numpy(np.array(tmp_e)).to(device).view(shape_of_input)


e0_n = 18
m0_n = 39451

def signed_to_u8(values):
    return [int(x) & 0xff for x in values]

def u8_to_signed(values):
    return [int(x) - 256 if int(x) > 127 else int(x) for x in values]

def forward_pass_astra(obs):
	# d
    print("obs: ", obs)
    obs_quant = np.round(np.array(obs)*(2**e0_n/m0_n))
    obs_quant = np.clip(obs_quant, -127, 127)
    print("input: ", obs_quant)
    obs_quant = signed_to_u8(obs_quant)
    print("obs_quant: ", u8_to_signed(astra.processor_run(obs_quant, startAddress=232, outputLength=10)[2:]))
    # exit()
    return u8_to_signed(astra.processor_run(obs_quant, startAddress=232, outputLength=10)[2:])                      

env = gym.make('Ant-v5', render_mode="human")
obs = env.reset(seed=42)

Sx3 = np.array([0.0245])


import imageio
action_space_high = env.action_space.high
action_space_low = env.action_space.low
frames = []
cd_list = []
for _ in range(10):
    timestep=0
    contact_counts = {f: 0 for f in ["front_left_foot","front_right_foot","back_left_foot","back_right_foot"]}
    contacts_over_time = []
    all_contact_logs = []
    obs = env.reset()
    model = env.unwrapped.model
    data = env.unwrapped.data
    total_reward= 0
    done = False
    while not done and timestep < 1000:
        # if timestep==0:
        #     action = forward_pass(obs[0])*Sx3
        # else:
        #     action = forward_pass(obs)*Sx3
        if timestep==0:
            action = forward_pass_astra(obs[0])*Sx3
        else:
            action = forward_pass_astra(obs)*Sx3
        
        # action = torch.tanh(action) * torch.tensor(env.action_space.high)
        action = action.cpu().numpy()
        # action[0], action[1], action[2] = 0, 0, 0
        
        action = np.clip(action, action_space_low, action_space_high)
        # if timestep ==0:
        #     action, _ = model.policy.predict(obs[0], deterministic=True)
        # else:
        #     action, _ = model.policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print("rendered")
        obs_req = obs
        # break
        done = terminated or truncated
        
        # if done:
            # print(timestep)
        total_reward += reward
        timestep+=1
    print(total_reward)
    # proportions = {f: contact_counts[f]/total_steps for f in contact_counts}
    # print(proportions)
# video_path = f"env_trial_quant.mp4"
# imageio.mimsave(video_path, frames, fps=30)
# print(f"Saved video to {video_path}")
