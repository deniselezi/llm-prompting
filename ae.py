import json
import time
from cohere import Client

API_KEY = ""

with open('data/ae/laptop/test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

client = Client(api_key=API_KEY)

def create_prompt(sentence, mode):
    prompt = None

    task_description = "You will be given a sentence expressing an opinion about an object (which we will call the aspect). This object could span multiple words (e.g. 'tech support' spans 2 words). Your goal is to generate a new sentence made up of individual letters separated by spaces. Each letter should correspond to a word in the original sentence. If the word is the start of the aspect, the corresponding letter should be 'B'. If the word is a continuation of the aspect, the corresponding letter should be 'I' (e.g. if 'tech support' is the aspect, 'tech' is 'B' and 'support' is 'I'). If the word is not part of the aspect then the corresponding letter should be 'O'. There should be the same number of letters in this new sentence as there are words in the original sentence. The generated sentence should be in this format: 'O O O O O B I I O O'"
    if mode=="zero-shot":
        prompt = f"{task_description}. Here is the sentence: [{sentence}]"
    elif mode=="one-shot":
        example_sentence = ""
        example_response = ""
        prompt = f"{task_description}. Here is the sentence: [{sentence}]. Example sentence: [{example_sentence}]. Correct example response: [{example_response}]."
    else:  # few shot
        examples = [
        ]
        shots = [f"Example sentence: [{s}]. Correct example response: [{r}]. " for s, r in examples]
        assert(len(shots) == 4)  # 4-shot
        prompt = f"{task_description}. Here is the sentence: [{sentence}]. " + shots[0] + shots[1] + shots[2] + shots[3]
    return prompt

def run(data, mode):
    # results = {}
    # for question_id in data:
    #     for paragraph in item["paragraphs"]:
    #         context = paragraph["context"]
    #         for qa in paragraph["qas"]:
    #             question = qa["question"]
    #             question_id = qa["id"]
    #             prompt = create_prompt(context, question, mode)
    #             response = client.v2.chat(
    #                 model="command-r",
    #                 messages=[
    #                      {"role": "user", "content": prompt}
    #                 ]
    #             ).message.content[0].text
    #             print(response)
    #             response = response.split(' ')
    #             results[question_id] = response
    #             print(response)
                
    #             # Rate limiting: Sleep for 1.5 seconds to not exceed 40 calls per minute
    #             time.sleep(1.5)
    
    # with open(f'ae_results_{mode}.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(results, outfile, indent=4)
    pass

run(json_data, mode='zero-shot')
run(json_data, mode='one-shot')
run(json_data, mode='few-shot')
