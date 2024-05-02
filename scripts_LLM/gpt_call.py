'''
Script for retrieving the Together AI LLM AUT.
Results are written to a seperate csv file.
'''
import argparse
import os
import csv
import numpy as np
import time
from datetime import datetime
import openai
from scipy.stats import lognorm, norm
np.random.seed(20)

# load API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

def gpt_call(prompt: str, model: str, temp: float) -> str:
    '''
    Calls OPENAI API with input and system prompt.
    '''
    response = openai.OpenAI().chat.completions.create(
        model = model,
        temperature = temp,
        max_tokens = 512,
        stop = ["]", "].", "]\n", "],"],
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def main_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=1, type=int, help="1: vf, 2: aut brick, 3: aut paperclip")
    parser.add_argument("--output_dir", default='../data_LLM/', type=str, help="Output directory for results")
    args = parser.parse_args()

    temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]      # [1]
    num_repetitions = 5

    ### DATA COLLECTION PREP GET MODELS
    chosen_models =  [
                            "gpt-4-turbo-2024-04-09"
                    ]
    
    print("number of models:", len(chosen_models))

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
        row_count = 0
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
    for idx, model in enumerate(chosen_models):
        
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
                    response = gpt_call(prompt, model, temp)
                except Exception as e: 
                    print("Error:", e)
                    response = 'NA'
                    
                # sleep to prevent rate limit
                time.sleep(2)

                # create row
                row = [timestamp, task, model, temp, response, num_responses]
                # write to csv
                writer.writerow(row)
                print(f'_____ Model: {model} ({idx+1}/{len(chosen_models)})', f'Temp: {temp} ({ti+1}/{len(temperatures)})', f'Rep: {ri} ({ri+1}/{num_repetitions}) _____')
                print(response)

    # close csv writer and file
    f.close()
    print("Finished")

if __name__ == "__main__":
    main_gpt()