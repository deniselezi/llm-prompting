import json

rrc_actual = "data"
rrc_results = ["rrc_results_zero-shot.json", "rrc_results_one-shot.json", "rrc_results_few-shot.json"]
asc_results = ["asc_results_zero-shot.json", "asc_results_one-shot.json", "asc_results_few-shot.json"]

with open('data/rrc/laptop/test.json', 'r') as f:
    data_json = json.load(f)

# mapping from qas id to the expected answer text
id_to_answer = {}

for entry in data_json["data"]:
    for para in entry["paragraphs"]:
        for qa in para["qas"]:
            qa_id = qa["id"]
            if qa["answers"]:
                # assuming one answer per QA for simplicity
                id_to_answer[qa_id] = qa["answers"][0]["text"]


def rrc_loss(results_json):
    with open('results.json', 'r') as f:
        results_json = json.load(f)

    for result_id, predicted_answer in results_json.items():
        gold_answer = id_to_answer.get(result_id)
        if gold_answer is None:
            print(f"{result_id}: No matching QA found.")
        else:
            is_match = predicted_answer.strip() == gold_answer.strip()
            print(f"{result_id}: Match = {is_match} | Predicted = '{predicted_answer}' | Expected = '{gold_answer}'")


for i in rrc_results:
    loss = rrc_loss(i)
    print(f"For the {i} experiment, the loss is {loss}")
