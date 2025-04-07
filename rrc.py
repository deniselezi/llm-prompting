import json
import time
from cohere import Client

API_KEY = ""

with open('data/rrc/laptop/test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

client = Client(api_key=API_KEY)

def create_prompt(context, question, mode):
    prompt = None

    task_description = "You will be given a context and a question. Your goal is to answer the question using a substring of the context (find the substring in the context which answers the question). Use only as many words as necessary to answer the question. Do not remove or reorder words in the substring you selected."
    if mode=="zero-shot":
        prompt = f"{task_description}. Here is the context: [{context}]. Here is the question: [{question}]"
    elif mode=="one-shot":
        example_context = "This is a great value for the money . We purchased this as a back up computer after our more expensive HP needed to be repaired . This is a great computer . We have n't had any problems with it at all . The body is a bit cheaply made so it will be interesting to see how long it holds up . Overall though , for the money spent it 's a great deal ."
        example_question = "how is the value ?"
        example_response = "great"
        prompt = f"{task_description}. Here is the context: [{context}]. Here is the question: [{question}]. Example context: [{example_context}]. Example question: [{example_question}]. Correct example response: [{example_response}]."
    else:  # few shot
        examples = [
            ("This is a great value for the money . We purchased this as a back up computer after our more expensive HP needed to be repaired . This is a great computer . We have n't had any problems with it at all . The body is a bit cheaply made so it will be interesting to see how long it holds up . Overall though , for the money spent it 's a great deal .", "how is the value ?", "great"), ("Right out of the box , this little netbook did everything I asked of it , including streaming the everyday video you 're bound to encounter checking mail and websites ( my biggest complaint previously ) . It even has a great webcam , and Skype works very well . The fact that you can spend over $ 100 on just a webcam underscores the value of this machine . The Windows 7 Starter is , in my opinion , a great way to think about using your netbook : basics , basics , basics . I wiped nearly everything off of it , installed OpenOffice and Firefox , and I am operating an incredibly efficient and useful machine for a great price . This netbook is a perfect supplementary computer to another laptop or desktop ( my wife and I have another laptop ) , or if you are a user who uses the computer for simple tasks . I use this for my tutoring business , and since I 'm always bouncing from student to student , it is ideal for portability and battery life ( yes , it gets the 8 hours as advertised ! ) . Finally , I should note that I took the 2GB RAM stick from my old EeePC and installed it before I even powered on for the first time . ASUS has done an outstanding job of evolving their netbooks , and I would recommend this to anyone who both understands their needs and how netbooks can fit them .", "does the webcam work with skype ?", "very well"), ("Purchased it for my birthday and love it ! I 've been a loyal Dell user for quite some time . But every six months my Dell computer crashes . Since I also do a lot of writing and illustrations using Adobe products , I thought I would try a Mac . I 'm so glad that I have . It 's been a very easy transition moving my Windows created files over to the Mac . So far I have n't lost any quality or information . I have a little bit of a learning curve with the keyboard shortcuts . But other than that , I really am loving the quality and speed .", "is it easy to learn the keyboard shortcuts of a mac ?", "little bit of a learning curve"), ("STOPPED BOOTING UP less than a week after my one-year warranty was up . Toshiba knows there is a manufacturing defect but will not acknowledge it . Summary : After doing some investigation on the web , I have found that Toshiba NB205s are not chronically not booting . Apparently there is a manufacturing defect , something with the amount of thermal paste . TOSHIBA WILL NOT ACKNOWLEDGE THIS PROBLEM . When I called Toshiba , they would not do anything and even tried to charge me $ 35 for the phone call , even though they did n't offer any technical support . I loved the netbook ( minus the fact that it was windows OS ) until this started happening . But if you ca n't make your product last more than a year , you will not get my business again .", "anything wrong with this computer ?", "STOPPED BOOTING UP less than a week after my one-year warranty was up")
        ]
        shots = [f"Example context: [{c}]. Example question: [{q}]. Correct example response: [{r}]. " for c, q, r in examples]
        assert(len(shots) == 4)  # 4-shot
        prompt = f"{task_description}. Here is the context: [{context}]. Here is the question: [{question}]. " + shots[0] + shots[1] + shots[2] + shots[3]
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
    
    with open(f'rrc_results_{mode}.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

run(json_data, mode='zero-shot')
run(json_data, mode='one-shot')
run(json_data, mode='few-shot')
