import json
import time
from cohere import Client

with open("key.txt", "r", encoding='utf-8') as keyfile:
    key = keyfile.readline()

API_KEY = key

with open('data/asc/rest/test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

client = Client(api_key=API_KEY)

def create_prompt(sentence, aspect, mode):
    prompt = None

    task_description = "You will be given a sentence and an object in the sentence. You need to classify the sentiment presented towards that object in the sentence with a one-word response: 'positive' if the sentiment is positive, 'negative' if the sentiment is negative and 'neutral' if it is neutral.  Do not respond with any words other than these three. Respond with one word."
    if mode=="zero-shot":
        prompt = f"{task_description}. Here is the sentence: [{sentence}]. Here is the object: [{aspect}]"
    elif mode=="one-shot":
        example_sentence = "The service is excellent and always informative without an air."
        example_object = "service"
        example_response = "positive"
        prompt = f"{task_description}. Here is the sentence: [{sentence}]. Here is the object: [{aspect}]. Example sentence: [{example_sentence}]. Example object: [{example_object}]. Correct example response: [{example_response}]."
    else:  # few shot
        examples = [
            ("The service is excellent and always informative without an air.", "service", "positive"),
            ("Perfectly al dente pasta, not drowned in sauce -- generous portions.", "sauce", "neutral"),
            ("Finally let into the store 5 at a time, to buy expensive slices from a harried staff.", "slices", "negative")
        ]
        shots = [f"Example sentence: [{s}]. Example object: [{o}]. Correct example response: [{r}]. " for s, o, r in examples]
        assert(len(shots) == 3)  # 3-shot
        prompt = f"{task_description}. Here is the sentence: [{sentence}]. Here is the object: [{aspect}]. " + shots[0] + shots[1] + shots[2]
    return prompt

def run(data, mode):
    results = {}
    for question_id in data:
        item = data[question_id]
        # print(item)
        sentence = item["sentence"]
        aspect = item["term"]
        # print(sentence)
        # print(aspect)
        # print(item['polarity'])
        # question_id = item["id"]
        prompt = create_prompt(sentence, aspect, mode)
        response = client.v2.chat(
            model="command-r",
            messages=[
                    {"role": "user", "content": prompt}
            ]
        ).message.content[0].text
        
        results[question_id] = response
        print(response)
        
        # Rate limiting: Sleep for 1.5 seconds to not exceed 40 calls per minute
        # time.sleep(1.5)
    
    with open(f'asc_results_{mode}.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

run(json_data, mode='zero-shot')
run(json_data, mode='one-shot')
run(json_data, mode='few-shot')
