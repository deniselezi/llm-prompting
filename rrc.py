import json
import time
from cohere import Client

with open("key.txt", "r", encoding='utf-8') as keyfile:
    key = keyfile.readline()

API_KEY = key

with open('data/rrc/rest/test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

client = Client(api_key=API_KEY)

def create_prompt(context, question, mode):
    prompt = None

    task_description = "You will be given a context and a question. Your goal is to answer the question using a single substring of the context (find the substring in the context which answers the question). Use only as many words as necessary to answer the question. Do not remove or reorder words in the substring you selected. The substring should be short."
    if mode=="zero-shot":
        prompt = f"{task_description}. Here is the context: [{context}]. Here is the question: [{question}]"
    elif mode=="one-shot":
        example_context = "I love Indian food and consider myself to be quite an expert on it . Chennai Garden is my favorite Indian restaurant in the city . They have authentic Indian at amazin prices . This restaurant is VEGETARIAN ; there are NO MEAT dishes whatsoever . The seats are uncomfortable if you are sitting against the wall on wooden benches . It 's a rather cramped and busy restaurant and it closes early ."
        example_question = "does this restaurant have a dish with meat ?"
        example_response = "NO"
        prompt = f"{task_description}. Here's an example context: [{example_context}]. Here's an example question: [{example_question}]. The correct response is: {example_response}. Here is your context: [{context}. Here is your question: [{question}]. Return the response."
    else:  # few shot
        examples = [
            ("I love Indian food and consider myself to be quite an expert on it . Chennai Garden is my favorite Indian restaurant in the city . They have authentic Indian at amazin prices . This restaurant is VEGETARIAN ; there are NO MEAT dishes whatsoever . The seats are uncomfortable if you are sitting against the wall on wooden benches . It 's a rather cramped and busy restaurant and it closes early .", "does this restaurant have a dish with meat ?", "NO"),
            ("Awsome Pizza especially the Margheritta slice . Always busy but fast moving . Great atmoshere and worth every bit . Open late ( well as late as I ever got there and I 'm a night person )", "how is their pizza ?", "Awsome"),
            ("This is some really good , inexpensive sushi . It costs $ 2 extra to turn a regular roll into an inside-out roll , but the roll more than triples in size , and that 's not just from the rice . The spicy Tuna roll is huge and probably the best that I 've had at this price range . The Yellowtail was particularly good as well . I have reservations about the all you can eat deal , however -- the choices are fairly limited and you can probably order more food than you can eat for less than $ 18 by just going off the menu . In any event , this is a place I 'll be sure to stop by again when I 'm in this part of town .", "do they have rolls with tuna ?", "spicy Tuna roll")
        ]
        shots = [f"Here's an example context: [{c}]. Here's an example question: [{q}]. The correct response is: {r}. " for c, q, r in examples]
        assert(len(shots) == 3)  # 3-shot
        prompt = f"{task_description}. " + shots[0] + shots[1] + shots[2] + f"Here is your context: [{context}]. Here is your question: [{question}]. Return the response."
    return prompt

def run(data, mode):
    results = {}
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                question_id = qa["id"]
                answer = qa["answers"][0]["text"]
                # print("--------")
                # print(context)
                # print(question)
                # print(answer)
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
                # time.sleep(0.5)
    
    with open(f'rrc_results_{mode}.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

# run(json_data, mode='zero-shot')
# run(json_data, mode='one-shot')
run(json_data, mode='few-shot')
