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
# import openai
import together
from scipy.stats import lognorm, norm
np.random.seed(20)

# load API key from environment variable
together.api_key = os.environ.get("TOGETHER_API_KEY")

def together_call(prompt: str, model: str, temp: float) -> str:
    '''
    Calls TOGETHER API with input prompt.
    '''
    output = together.Complete.create(
        prompt = prompt,
        model = model,
        temperature = temp,
        max_tokens = 512,
        stop = ["]", "].", "]\n", "],"]
    )
    return output['choices'][0]['text']


def main_together():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=2, type=int, help="1: vf, 2: aut brick, 3: aut paperclip")
    parser.add_argument("--output_dir", default='../data_LLM/', type=str, help="Output directory for results")
    args = parser.parse_args()
    
    # get timestamp of the current date and time
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    temperatures = [0.8]
    num_repetitions = 5 # generate 5 responses per model per temp

    ### DATA COLLECTION PREP GET MODELS
    final_models =  [
                            "meta-llama/Llama-3-70b-chat-hf",
                            "mistralai/Mistral-7B-Instruct-v0.2",
                            "snorkelai/Snorkel-Mistral-PairRM-DPO",
                            "upstage/SOLAR-10.7B-Instruct-v1.0",
                            "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
                    ]
    
    print("number of models:", len(final_models))      # 169

    ### OPEN CSV WRITER
    # open csv files to write gpt responses and conversation log to, create writer and logger

    # filename = f'{args.output_dir}/results_task{args.task}_finalmodels.csv'
    # if os.path.exists(filename):
    #     exists = True
    #     f = open(filename, 'a')
    # else:
    #     f = open(filename, 'w')
    #     exists = False

    # writer = csv.writer(f)
    # if not exists:
    #     header = ['timestamp', 'task', 'model', 'temp', 'response', 'num_responses']
    #     writer.writerow(header)
    # else:
    #     writer.writerow("\n")


    assert(args.task in [1,2,3])
    if args.task == 2:
        task = 'brick'
        # num_responses = np.floor(lognorm.rvs(*(0.5676690838374294, 1.168052015273098, 15.927918311237551), size=len(final_models))).astype(int)
        num_responses = 30
    elif args.task == 3:
        task = 'paperclip'
        # num_responses = np.floor(lognorm.rvs(*(0.5498563871643265, 1.3641677548168816, 15.761105902350707), size=len(final_models))).astype(int)
        num_responses = 20
    elif args.task == 1:
        task = 'animals'
        # num_responses = np.floor(norm.rvs(*(28.53125, 8.431917796279462), size=len(final_models))).astype(int)
        num_responses = 30
    
    
    ### DATA COLLECTION
    for idx, model in enumerate(final_models):
        
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
                    response = together_call(prompt, model, temp)
                except Exception as e: 
                    print("Error:", e)
                    response = 'NA'
                    
                # sleep to prevent rate limit
                time.sleep(2)

                # create row
                row = [timestamp, task, model, temp, response, num_responses]
                # write to csv
                # writer.writerow(row)
                print(f'_____ Model: {model} ({idx+1}/{len(final_models)})', f'Temp: {temp} ({ti+1}/{len(temperatures)})', f'Rep: {ri} ({ri+1}/{num_repetitions}) _____')
                print(response)

    # close csv writer and file
    # f.close()
    print("Finished")

if __name__ == "__main__":
    main_together()