import json
import glob
import os
import tiktoken
import numpy as np
np.random.seed(32)

import json
import tiktoken
import numpy as np
from collections import defaultdict

import ipdb
st = ipdb.set_trace

# Upload fine-tuning files
import openai
import os

openai.api_key = os.getenv("AZURE_OPENAI_KEY") 
openai.api_base =  os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-12-01-preview' # This API version or later is required to access fine-tuning for turbo/babbage-002/davinci-002

#Retrieve fine_tuned_model name
job_id = "ftjob-c5cdd4d36f7943a79da87805835ee94c"
response = openai.FineTuningJob.retrieve(job_id)

print(response)
fine_tuned_model = response["fine_tuned_model"]

assert(False) # safeguard

base_path = "/home/gsarch/repo/teach-continual-memory/output/teach_skill_learning_fullmemlearning_idm_00"

save_files = False

if save_files:
    with open('prompt/prompt_plan_input_traingpt35.txt') as f:
        prompt_input_plan = f.read()

    with open('prompt/prompt_plan_output_traingpt35.txt') as f:
        prompt_output_plan = f.read()

    with open('prompt/api_primitives_nodefinitions.py') as f:
        api = f.read()

    # prompt = prompt_plan
    prompt_input_plan = prompt_input_plan.replace('{API}', api)

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    validation_set = []
    train_set = []

    total_prompt_length = 0
    all_prompt_lens = []
    # jsonl = []
    for file_idx in range(len(glob.glob(os.path.join(base_path, "successful_skill_functions", "*")))):
        python_function_file = os.path.join(base_path, "successful_skill_functions", f"skill_func_{file_idx}.txt")
        state_file = os.path.join(base_path, "successful_skill_states_filtered", f"skill_state_filtered_{file_idx}.txt")
        plan_file = os.path.join(base_path, "successful_skill_plan", f"skill_plan_{file_idx}.txt")
        summary_file = os.path.join(base_path, "successful_skill_summary", f"skill_summ_{file_idx}.txt")
        command_file = os.path.join(base_path, "successful_skill_commands", f"skill_command_{file_idx}.txt") 

        with open(python_function_file) as f:
            python_function = f.read()

        with open(state_file) as f:
            state = f.read()

        with open(command_file) as f:
            command = f.read()

        input = prompt_input_plan
        input = input.replace('{STATE}', state)
        input = input.replace('{command}', command)
        # prompt = prompt.replace('{summary}', summary)
        # prompt = prompt.replace('{plan}', plan)

        output = prompt_output_plan
        output = output.replace('{script}', python_function)

        prob = np.random.uniform()
        if prob<0.2:
            validation_set.append({"messages": [{"role": "user", "content": input}, {"role": "assistant", "content": output}]})
        else:
            train_set.append({"messages": [{"role": "user", "content": input}, {"role": "assistant", "content": output}]})
        
        prompt_token_length = len(enc.encode(input)) + len(enc.encode(output))
        all_prompt_lens.append(prompt_token_length)
        total_prompt_length += prompt_token_length

    print(np.mean(all_prompt_lens), np.max(all_prompt_lens), total_prompt_length)
    print(0.0017 * total_prompt_length / 1000)

    with open(os.path.join(base_path, 'train.jsonl'), 'w') as outfile:
        for entry in train_set:
            json.dump(entry, outfile)
            outfile.write('\n')

    with open(os.path.join(base_path, 'validation.jsonl'), 'w') as outfile:
        for entry in validation_set:
            json.dump(entry, outfile)
            outfile.write('\n')

# Load the training set
with open(os.path.join(base_path, 'train.jsonl'), 'r', encoding='utf-8') as f:
    training_dataset = [json.loads(line) for line in f]

# Training dataset stats
print("Number of examples in training set:", len(training_dataset))
print("First example in training set:")
for message in training_dataset[0]["messages"]:
    print(message)

# Load the validation set
with open(os.path.join(base_path, 'validation.jsonl'), 'r', encoding='utf-8') as f:
    validation_dataset = [json.loads(line) for line in f]

# Validation dataset stats
print("\nNumber of examples in validation set:", len(validation_dataset))
print("First example in validation set:")
for message in validation_dataset[0]["messages"]:
    print(message)

encoding = tiktoken.get_encoding("cl100k_base") # default encoding used by gpt-4, turbo, and text-embedding-ada-002 models

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

files = [os.path.join(base_path, 'train.jsonl'), os.path.join(base_path, 'validation.jsonl')]

for file in files:
    print(f"Processing file: {file}")
    with open(file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    total_tokens = []
    assistant_tokens = []

    for ex in dataset:
        messages = ex.get("messages", {})
        total_tokens.append(num_tokens_from_messages(messages))
        assistant_tokens.append(num_assistant_tokens_from_messages(messages))
    
    print_distribution(total_tokens, "total tokens")
    print_distribution(assistant_tokens, "assistant tokens")
    print('*' * 50)

training_file_name = os.path.join(base_path, 'train.jsonl')
validation_file_name = os.path.join(base_path, 'validation.jsonl')

st()

# Upload the training and validation dataset files to Azure OpenAI with the SDK.

if (0):
    training_response = openai.File.create(
        file=open(training_file_name, "rb"), purpose="fine-tune", user_provided_filename="training_set.jsonl"
    )
    training_file_id = training_response["id"]

    validation_response = openai.File.create(
        file=open(validation_file_name, "rb"), purpose="fine-tune", user_provided_filename="validation_set.jsonl"
    )
    validation_file_id = validation_response["id"]
else:
    training_file_id = "file-d15bcd9932564a589c59fc017020561a"
    validation_file_id = "file-4ae07733ebb0480989dcb17a081e7f3b"

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

# submit job for training
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-35-turbo-1106",
)

job_id = response["id"]

# You can use the job ID to monitor the status of the fine-tuning job.
# The fine-tuning job will take some time to start and complete.

print("Job ID:", response["id"])
print("Status:", response["status"])
print(response)

# Track training status

from IPython.display import clear_output
import time

start_time = time.time()

# Get the status of our fine-tuning job.
response = openai.FineTuningJob.retrieve(job_id)

status = response["status"]

# If the job isn't done yet, poll it every 10 seconds.
while status not in ["succeeded", "failed"]:
    time.sleep(10)
    
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)
    print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
    status = response["status"]
    print(f'Status: {status}')
    clear_output(wait=True)

print(f'Fine-tuning job {job_id} finished with status: {status}')

# List all fine-tuning jobs for this resource.
print('Checking other fine-tune jobs for this resource.')
response = openai.FineTuningJob.list()
print(f'Found {len(response["data"])} fine-tune jobs.')