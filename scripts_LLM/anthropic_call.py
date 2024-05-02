import anthropic
import os
import argparse
import csv
import numpy as np
import time
from datetime import datetime

anthropic.api_key = os.environ.get("ANTHROPIC_API_KEY")

def anthropic_call(prompt: str, temp: float) -> str:
    output = anthropic.Anthropic().messages.create(
        model="claude-3-opus-20240229",
        max_tokens=512,
        temperature=temp,
        stop_sequences = ["]", "].", "]\n", "],"],
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return output.content[0].text


def main_anthropic():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=1, type=int, help="1: vf, 2: aut brick, 3: aut paperclip")
    parser.add_argument("--output_dir", default='../data_LLM/', type=str, help="Output directory for results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]      # [1]
    num_repetitions = 5 # generate 5 responses per model per temp
    model = "claude-3-opus-20240229"

    ### OPEN CSV WRITER
    # open csv files to write gpt responses and conversation log to, create writer and logger
    filename = f'{args.output_dir}/results_task{args.task}_finalmodels.csv'
    if os.path.exists(filename):
        f = open(filename, 'r')
        reader = csv.reader(f)
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
    
    
    ### DATA COLLECTION
    for ti, temp in enumerate(temperatures):        
        for ri in range(num_repetitions):
            if args.task > 1:
                prompt = f"Think of {num_responses} creative uses for '{task}'.\nAnswer in short phrases of 1 to 5 words. Give your responses as a comma-separated list in square braces '[]'. For example, if your creative uses are a,b,c,d,.., output: [a,b,c,d,..]\nDo not give any other words or text.\n\nOutput: ["
            elif args.task == 1:
                prompt = f"Think of {num_responses} {task}.\nGive your responses as a comma-separated list in square braces '[]'. For example, if your {task} are a,b,c,d,.., output: [a,b,c,d,..]\nDo not give any other words or text.\n\nOutput: ["

            # timestamp data collection
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # collect data with model
            try:
                response = anthropic_call(prompt, temp)
            except Exception as e:
                print("Error:", e)
                response = 'NA'
                
            # sleep to prevent rate limit
            time.sleep(5)

            # create row
            row = [timestamp, task, model, temp, response, num_responses]
            # write to csv
            writer.writerow(row)
            print(f'_____ Temp: {temp} ({ti+1}/{len(temperatures)})', f'Rep: {ri} ({ri+1}/{num_repetitions}) _____')
            print(response)

    # close csv writer and file
    f.close()
    print("Finished")

if __name__ == "__main__":
    main_anthropic()