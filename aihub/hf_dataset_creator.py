import os
DIR_PATH = os.path.dirname(__file__)
from datasets import load_dataset

# low_call_voices_prepreocessed = load_dataset("maxseats/meeting_valid_preprocessed", cache_dir=DIR_PATH)
# print(type(low_call_voices_prepreocessed))
# print(low_call_voices_prepreocessed)
# print('-'*40)
# print(type(low_call_voices_prepreocessed["train"]))
# print(low_call_voices_prepreocessed["train"])
# print('-'*40)
# print(type(low_call_voices_prepreocessed["train"]["input_features"]))
# print(low_call_voices_prepreocessed["train"]["input_features"])
# # print('-'*40)
# # print(type(low_call_voices_prepreocessed["train"]["labels"]))
# # print(low_call_voices_prepreocessed["train"]["labels"])