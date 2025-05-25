import os
RLHF_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), *[".."] * 4)
print(f"RLHF_ROOT_DIR: {RLHF_ROOT_DIR}")