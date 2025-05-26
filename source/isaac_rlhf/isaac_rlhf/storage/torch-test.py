import torch

policy_id_tensor = torch.tensor([[3,4],[3,4],[11,10],[11,10]])
policy_id_pairs = [tuple(pair.tolist()) for pair in policy_id_tensor]

unique_pairs = list(set(policy_id_pairs))
pair_preference_map = {}
for pair in unique_pairs:
    id0, id1 = pair

    # Your LLM preference function (assumed to return 1 or 0)
    preference = 1 if id0>id1 else 0
    pair_preference_map[pair] = preference

y_new_list = []
for pair in policy_id_pairs:
    preference = pair_preference_map[pair]
    y_new_list.append(preference)

# 6. Convert to torch tensor
y_new = torch.tensor(y_new_list)
print("y_new:", y_new)