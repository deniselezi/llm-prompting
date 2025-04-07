import json
import time
from cohere import Client

with open("key.txt", "r", encoding='utf-8') as keyfile:
    key = keyfile.readline()

# print(key)

API_KEY = key

with open('data/ae/laptop/test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

client = Client(api_key=API_KEY)

# def create_prompt(sentence, count, mode):
#     prompt = None

#     task_description = "You will be given a series of space-delimited tokens forming a review sentence. Anything which is delimited by a space should count as a separate token, including punctuation like '.', ',' or '...'. Spaces do not count as tokens, and only serve as delimiters. Your goal is to identify which tokens in the review are terms which point to aspects of a larger entity (e.g. in the case of a restaurant review, terms would be 'waiter', 'food' and 'price'). These terms could span multiple tokens (if a term is made up of multiple words). Generate another sentence of space-delimited letters with as many letters as there are tokens in the input sentence. In this new sentence, a letter is 'B' if the corresponding token is the first word of a term, 'I' if the corresponding token is a word of a term other than the first, or the letter 'O' if the corresponding token is not part of a term. Each letter should correspond to a token in the original sentence. The generated sentence should be in this format: 'O O O O O B I I O O'. Do not respond with anything other than the three letters 'B', 'I' and 'O'"
#     if mode=="zero-shot":
#         prompt = f"{task_description}. Here is the review: [{sentence}]. Generate a sentence identifying the terms of the review as previously described. Your generated sentence should have {count} letters"
#     elif mode=="one-shot":
#         example_sentence = "Keyboard is great but primary and secondary control buttons could be more durable ."
#         example_response = "B O O O O O O B I O O O O O"
#         prompt = f"{task_description}. Here is the sentence: [{sentence}]. Example sentence: [{example_sentence}]. Correct example response: [{example_response}]."
#     else:  # few shot
#         examples = [
#             ("Keyboard is great but primary and secondary control buttons could be more durable .", "B O O O O O O B I O O O O O"),
#             ("I bought this laptop about a month ago to replace my compaq laptop .", "O O O O O O O O O O O O O O"),
#             ("So what am I supposed to do ? The LG service center can not provide me the `` service `` when it is called the `` service center `` .", "O O O O O O O O O B I I O O O O O O B O O O O O O O B I O O"),
#         ]
#         shots = [f"Example sentence: [{s}]. Correct example response: [{r}]. " for s, r in examples]
#         assert(len(shots) == 3)  # 4-shot
#         prompt = f"{task_description}. Here is the sentence: [{sentence}]. " + shots[0] + shots[1] + shots[2]
#     return prompt


# def run(data, mode):
#     results = {}
#     for question_id in data:
#         item = data[question_id]
#         tokens = item["sentence"]
#         count = len(tokens)
#         sentence = ' '.join(tokens)
#         label = item["label"]
#         print(label)
#         label = ' '.join(label)
#         # print(sentence)
#         # print(sentence)
#         prompt = create_prompt(sentence, count, mode)
#         response = client.v2.chat(
#             model="command-r",
#             messages=[
#                     {"role": "user", "content": prompt}
#             ]
#         ).message.content[0].text
#         response = response.split(' ')
#         print(response)

#         # response = response.split(' ')
#         # results[question_id] = response
#         # print(response)

#         # Rate limiting: Sleep for 1.5 seconds to not exceed 40 calls per minute
#         time.sleep(1.5)
    
#     with open(f'ae_results_{mode}.json', 'w', encoding='utf-8') as outfile:
#         json.dump(results, outfile, indent=4)
#     pass


def create_prompt(sentence, mode):
    prompt = None
    task_description = "You will be given a review sentence. In this review, you need to extract all the terms, such as 'waiter', 'food' and 'price' in the case of restaurant reviews, whch point to aspects of a larger entity (in this example, the larger entity would be 'restaurant'). Terms can be made up of multiple words. Your answer should be a list of terms separated by commas, like this: term1,term2,term3. You must ONLY use words that are exactly in the sentence. Do not alter words from the sentence or introduce new words to 'create' terms. If you find no terms, return 'No terms were found.'."
    if mode=="zero-shot":
        prompt = f"{task_description}. Here is your sentence: [{sentence}]. Identify the terms."
    elif mode=="one-shot":
        example_sentence = "Keyboard is great but primary and secondary control buttons could be more durable ."
        example_response = "keyboard,secondary control"
        prompt = f"{task_description}. Here's an example sentence: [{example_sentence}]. Here are the terms of the sentence: {example_response}. Here is your sentence: [{sentence}]. Identify the terms."
    else:  # few shot
        examples = [
            ("Keyboard is great but primary and secondary control buttons could be more durable .", "keyboard,secondary control"),
            ("I bought this laptop about a month ago to replace my compaq laptop .", "No terms were found."),
            ("So what am I supposed to do ? The LG service center can not provide me the `` service `` when it is called the `` service center `` .", "LG service center,service center"),
        ]
        shots = [f"Here's an example sentence: [{s}]. Here are the terms of the sentence: {r}. " for s, r in examples]
        assert(len(shots) == 3)  # 3-shot
        prompt = f"{task_description}. " + shots[0] + shots[1] + shots[2] + f"Here is your sentence: [{sentence}]. Identify the terms."
    return prompt


def run(data, mode):
    def label_tokens(tokens, terms, labels):
        if len(terms) == 1 and terms[0] == "":  # no terms were fonud
            return labels
        substrings_split = [[word.lower() for word in s.split()] for s in terms]

        for substring in substrings_split:
            sub_len = len(substring)
            for i in range(len(tokens) - sub_len + 1):
                if tokens[i:i + sub_len] == substring:
                    labels[i] = 'B'
                    for j in range(1, sub_len):
                        labels[i + j] = 'I'
        return labels

    results = {}
    for question_id in data:
        item = data[question_id]
        tokens = item["sentence"]
        sentence = ' '.join(tokens)
        label = item["label"]
        # print(label)
        label = ' '.join(label)
        # print(sentence)
        # print(tokens)
        prompt = create_prompt(sentence, mode)
        terms = client.v2.chat(
            model="command-r",
            messages=[
                    {"role": "user", "content": prompt}
            ]
        ).message.content[0].text
        if terms[-1] == ".":  # .
            terms = terms[:-1]
        if 'no terms were found' in terms.lower():
            terms = ""
        terms = terms.split(',')
        terms = [t.strip() for t in terms]
        terms = [t.lower() for t in terms]
        tokens = [t.lower() for t in tokens]
        labels = ['O'] * len(tokens)
        response = label_tokens(tokens, terms, labels)
        print(question_id, response)
        # print(response)

        # response = response.split(' ')
        results[question_id] = response
        # print(response)

        # Rate limiting: Sleep for 1.5 seconds to not exceed 40 calls per minute
        # time.sleep(1.5)
    
    with open(f'ae_results_{mode}.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)
    pass

# run(json_data, mode='zero-shot')
# run(json_data, mode='one-shot')
run(json_data, mode='few-shot')
