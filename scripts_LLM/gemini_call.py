import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
import os
import argparse
import time
from datetime import datetime
import csv
import numpy as np

vertexai.init(project="augmented-slice-419913", location="europe-west3")

def gemini_call(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

def main_gemini():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=1, type=int, help="1: vf, 2: aut brick, 3: aut paperclip")
    parser.add_argument("--output_dir", default='../data_LLM/', type=str, help="Output directory for results")
    args = parser.parse_args()

    temperatures = [0.4, 0.9]      # [1]
    num_repetitions = 3

    filename = f'{args.output_dir}/results_task{args.task}_finalmodels.csv'
    if os.path.exists(filename):
        exists = True
        f = open(filename, 'a')
    else:
        f = open(filename, 'w')
        exists = False

    writer = csv.writer(f)
    if not exists:
        header = ['timestamp', 'task', 'model', 'temp', 'response', 'num_responses']
        writer.writerow(header)
    else:
        writer.writerow("\n")

    assert(args.task in [1,2,3])
    if args.task == 2:
        task = 'brick'
        num_responses = 20
    elif args.task == 3:
        task = 'paperclip'
        num_responses = 20
    elif args.task == 1:
        task = 'animals'
        num_responses = 30
    
    if args.task > 1:
        prompt = f"Think of {num_responses} creative uses for '{task}'.\nAnswer in short phrases of 1 to 5 words. Give your responses as a comma-separated list in square braces '[]'. For example, if your creative uses are a,b,c,d,.., output: [a,b,c,d,..]\nDo not give any other words or text.\n\nOutput: ["
    elif args.task == 1:
        prompt = f"Think of {num_responses} {task}.\nGive your responses as a comma-separated list in square braces '[]'. For example, if your {task} are a,b,c,d,.., output: [a,b,c,d,..]\nDo not give any other words or text.\n\nOutput: ["

    ### DATA COLLECTION
    for ti, temp in enumerate(temperatures):       
        for ri in range(num_repetitions):
            model = GenerativeModel("gemini-1.0-pro-002", generation_config={"temperature":temp})
            chat = model.start_chat()

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            try:
                response = gemini_call(chat, prompt)
            except Exception as e:
                print("Error:", e)
                response = 'NA'
                
            # sleep to prevent rate limit
            time.sleep(8)

            # create row
            row = [timestamp, task, "gemini-1.0-pro-002", temp, response, num_responses]
            # write to csv
            writer.writerow(row)
            print(f'_____ Temp: {temp} ({ti+1}/{len(temperatures)})', f'Rep: {ri} ({ri+1}/{num_repetitions}) _____')
            print(response)


if __name__ == "__main__":
    main_gemini()