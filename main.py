import json
import time
from cohere import Client

with open("key.txt", "r", encoding='utf-8') as keyfile:
    key = keyfile.readline()

API_KEY = key

with open('data/rrc/laptop/test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

client = Client(api_key=API_KEY)

def create_prompt(context, question, mode):
    if mode=="zero-shot":
        prompt = "Here is the context:[" + context + "]. Here is the question: [" + question + "]. You should answer the question using only words which are contained in the context itself. Use only as many words as necessary to answer the question. Do not reorder the words or add any words not contained in the context."
    elif mode=="one-shot":
        prompt = "Here is the context:[" + context + "]. Here is the question: [" + question + "]. You should answer the question using only words which are contained in the context itself. Use only as many words as necessary to answer the question. Do not reorder the words or add any words not contained in the context." + " Here is an example to guide you. Example context: [This is a great value for the money . We purchased this as a back up computer after our more expensive HP needed to be repaired . This is a great computer . We have n't had any problems with it at all . The body is a bit cheaply made so it will be interesting to see how long it holds up . Overall though , for the money spent it 's a great deal .]. Example question: [how is the value ?]. Correct example response: [great]."
    return prompt
        
def run(data, mode):
    results = {}
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                question_id = qa["id"]
                prompt = create_prompt(context, question, mode)
                response = client.v2.chat(
                    model="command-r",
                    messages=[
                         {"role": "user", "content": prompt}
                    ]
                ).message.content[0].text
                
                results[question_id] = response
                print(response)
                
                # Rate limiting: Sleep for 1.5 seconds to not exceed 40 calls per minute
                time.sleep(1.5)
    
    with open('results.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)
                

run(json_data, mode='zero-shot')
