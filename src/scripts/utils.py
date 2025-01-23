import yaml, os


def read_config(path="config.yaml"):
    # with open("config.yaml") as stream:
    #     try:
    #         print(yaml.safe_load(stream))
    #         config = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    assert os.path.exists(path)
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


# def load_individual_weights():

#     start_memory_tracking()

#     with torch.device("meta"):
#         model = GPTModel(BASE_CONFIG)

#     model = model.to_empty(device=device)

#     # print_memory_usage()
#     param_dir = "model_parameters"

#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             weight_path = os.path.join(param_dir, f"{name}.pt")
#             if os.path.exists(weight_path):
#                 param_data = torch.load(weight_path, map_location="cpu", weights_only=True)
#                 param.copy_(param_data)
#                 del param_data  # Free memory
#             else:
#                 print(f"Warning: {name} not found in {param_dir}.")


# model = GPTModel(BASE_CONFIG).to(device)

# state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

# print_memory_usage()

# # Sequentially copy weights to the model's parameters
# with torch.no_grad():
#     for name, param in model.named_parameters():
#         if name in state_dict:
#             param.copy_(state_dict[name].to(device))
#         else:
#             print(f"Warning: {name} not found in state_dict.")