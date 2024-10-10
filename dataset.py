import os
import json
import shutil
from prompt2 import get_ask_prompt2
SAMPLE_DATA_PATH  = "/home/zhaoyang/projects/siwei/Project/data/ManytoMany_sample"
ALL_DATA_PATH  = "/home/zhaoyang/projects/siwei/Project/data/ManytoMany_all"

def get_dataset(dataset):
    data_path = os.path.join(SAMPLE_DATA_PATH,dataset,'eval_sample_fb.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for d in data:
        processed_data_dict = {}
        id = d['id']
        model = d['model']
        question = d['question']
        answer = d['answer']
        gt_answer = d['gt_answer'][0]['text']
        feedback = d['feedback']
        score = d['score']
        question_text = ""
        answer_text = ""
        image_no = 0
        images = []

        for q in question:
            if q['text'] is not None:
                question_text+=q['text']+'\n'
            if q['image'] is not None:
                image_path = os.path.join(SAMPLE_DATA_PATH,dataset,q['image'])
                if os.path.exists(image_path):
                    question_text+="<image>"
                    image_no+=1
                    images.append(image_path)
                
        for a in answer:
            if a['text'] is not None:
                answer_text+=a['text']+'\n'
            if a['image'] is not None:
                image_path = os.path.join(SAMPLE_DATA_PATH,dataset,a['image'])
                if os.path.exists(image_path):
                    answer_text+="<image>"
                    image_no+=1
                    images.append(image_path)

        processed_data_dict['id'] = id
        processed_data_dict['model'] = model
        processed_data_dict['question'] = question_text
        processed_data_dict['answer'] = answer_text
        processed_data_dict['gt_answer'] = gt_answer
        processed_data_dict['images'] = images
        processed_data_dict['feedback'] = feedback
        processed_data_dict['score'] = score

        processed_data.append(processed_data_dict)

    return processed_data

def get_question_dataset(dataset):
    data_path = os.path.join(SAMPLE_DATA_PATH,dataset,'eval_sample_fb.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for d in data:
        processed_data_dict = {}
        id = d['id']
        question = d['question']
        gt_answer = d['gt_answer'][:3]
        question_text = ""
        image_no = 0
        images = []

        for q in question:
            if q['text'] is not None:
                question_text+=q['text']+'\n'
            if q['image'] is not None:
                image_path = os.path.join(SAMPLE_DATA_PATH,dataset,q['image'])
                # destination_dir = os.path.dirname(q['image'])
                # os.makedirs(destination_dir, exist_ok=True)
                # shutil.copy(image_path, q['image'])
                if os.path.exists(image_path):
                    question_text+="<image>"
                    image_no+=1
                    images.append(image_path)

        # for g in gt_answer:
        #     if g['image'] is not None:
        #         image_path = os.path.join(SAMPLE_DATA_PATH,dataset,g['image'])
        #         destination_dir = os.path.dirname(g['image'])
        #         os.makedirs(destination_dir, exist_ok=True)
        #         shutil.copy(image_path, g['image'])

        processed_data_dict['id'] = id
        processed_data_dict['question'] = question
        processed_data_dict['gt_answer'] = gt_answer
        processed_data_dict['question_text'] = question_text
        processed_data_dict['images'] = images

        processed_data.append(processed_data_dict)

    return processed_data

def get_question_anser_dataset(dataset,filename):
    data_path = os.path.join(ALL_DATA_PATH,dataset,f'{filename}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for d in data:
        processed_data_dict = {}
        id = d['id']
        question = d['question']
        question_text = ""
        question_images = []

        for q in question:
            if q['text'] is not None:
                question_text+="<text>"+q['text']+'\n'+"</text>"
            if q['image'] is not None:
                image_path = os.path.join(ALL_DATA_PATH,dataset,q['image'])
                if os.path.exists(image_path):
                    question_text+="<image>"
                    images.append(image_path)
        system_prompt = "<text>"+get_ask_prompt2(dataset)+"</text>"
        question_text += system_prompt

        question_anwer_text = question_text
        question_anwer_images = question_images.copy()

        for a in answer[:2]:
            if a['text'] is not None:
                question_anwer_text+="<text>"+a['text']+'\n'+"</text>"
            if a['image'] is not None:
                image_path = os.path.join(ALL_DATA_PATH,dataset,a['image'])
                if os.path.exists(image_path):
                    question_anwer_text+="<image>"
                    question_anwer_images.append(image_path)
                else:
                    print(f"{image_path} not found!")

        processed_data_dict['id'] = id
        processed_data_dict['question'] = question
        processed_data_dict['answer'] = answer[:2]
        processed_data_dict['question_text'] = question_text
        processed_data_dict['question_anwer_text'] = question_anwer_text
        processed_data_dict['question_images'] = question_images
        processed_data_dict['question_anwer_images'] = question_anwer_images

        processed_data.append(processed_data_dict)

    return processed_data

def get_dataset_model(dataset, model):
    temp_path = f'eval_all/temp/{dataset}_{model}_eval_result_temp.json'
    data_path = os.path.join(ALL_DATA_PATH,dataset,f'{model}_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for d in data:
        processed_data_dict = {}
        id = d['id']
        model = d['model']
        question = d['question']
        answer = d['answer']
        gt_answer = d['gt_answer'][0]['text']
        question_text = ""
        answer_text = ""
        image_no = 0
        images = []

        for q in question:
            if q['text'] is not None:
                question_text+=q['text']+'\n'
            if q['image'] is not None:
                image_path = os.path.join(ALL_DATA_PATH,dataset,q['image'])
                if os.path.exists(image_path):
                    question_text+=f"Image-{image_no}: <image>\n"
                    image_no+=1
                    images.append(image_path)
                else:
                    print(f"{image_path} not found!")
                
        for a in answer[:2]:
            if a['text'] is not None:
                answer_text+=a['text']+'\n'
            if a['image'] is not None:
                image_path = os.path.join(ALL_DATA_PATH,dataset,a['image'])
                if os.path.exists(image_path):
                    answer_text+=f"Image-{image_no}: <image>\n"
                    image_no+=1
                    images.append(image_path)
                else:
                    print(f"{image_path} not found!")

        processed_data_dict['id'] = id
        processed_data_dict['model'] = model
        processed_data_dict['question'] = question_text
        processed_data_dict['answer'] = answer_text
        processed_data_dict['gt_answer'] = gt_answer
        processed_data_dict['images'] = images
        if len(images) <= 4:
            processed_data.append(processed_data_dict)

    if os.path.exists(temp_path):
        print("Find temp data!")
        with open(temp_path, 'r') as f:
            temp_data = json.load(f)
        i = 0
        for td in temp_data:
            if 'gpt_feedback' in td:
                assert(processed_data[i]['id'] == td['id'])
                processed_data[i]['gpt_feedback'] = td['gpt_feedback']
                i+=1
            else:
                break

    return processed_data
