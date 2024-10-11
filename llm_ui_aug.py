import pandas as pd
import pickle
import os
import time
import tqdm
import requests
import torch
import transformers

from transformers import AutoModelForCausalLM, LlamaTokenizer

file_path = "/home/dyt/coldstart/RecBole/"
dataset_path = "/home/dyt/coldstart/RecBole/dataset/ml-100k/"
# model_dir = "/home/dyt/coldstart/llama/llama-2-7b-chat-hf"
model_dir = "/home/dyt/coldstart/Meta-Llama-3-8B-Instruct"

torch.set_default_tensor_type(torch.cuda.HalfTensor)

max_threads = 5
cnt = 0

# seperator = '::'
seperator = '||'
# MovieLens
def construct_prompting(uid, user_attribute, item_attribute, item_list, candidate_list):
    uid = uid - 1
    # make history string
    history_string = "User history:\n"
    for index in item_list:
        # index = index - 1
        history_string = generate_item_prompt(history_string, index, item_attribute)
    # make candidates
    candidate_string = "Candidates:\n"
    for index in candidate_list:
        # index = index - 1
        index = int(index)
        candidate_string = generate_item_prompt(candidate_string, index, item_attribute)
    # output format
    output_format = f"You Must output the index of user\'s favorite and least favorite movie only from candidate, " \
                    "but not user history.You Must output the index of user\'s favorite and least favorite movie only from candidate, " \
                    "but not user history.You Must output the index of user\'s favorite and least favorite movie only from candidate, " \
                    "but not user history. Please get the index from candidate, at the beginning of each line.\n" \
                    f"Output format:\nTwo numbers separated by '{seperator}'. Nothing else. You must just give the index of " \
                    "candicates, remove [] (just output the digital value), please do not output other thing else, " \
                    "do not give reasoning.\n\nYou must give me only Two numbers only from Candidates. You must give " \
                    "me only Two numbers only from Candidates. You must give me only Two numbers only from Candidates." \
                    " You must give me only Two numbers only from Candidates. "
    # make prompt
    u_age = user_attribute['age'][uid]
    u_gender = 'male' if user_attribute['gender'][uid] == 'M' else 'female'
    u_occupation = user_attribute['occupation'][uid]

    prompt = f"You are a movie recommendation system and required to recommend user who is {u_age} year " \
             f"old {u_gender} person and the occupation is {u_occupation} with movies based on user history " \
             f"that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt


def generate_item_prompt(candidate_string, index, item_attribute):
    index = index - 1
    title = item_attribute['title'][index]
    genre = item_attribute['gender'][index]
    year = item_attribute['year'][index]
    candidate_string += "["
    candidate_string += str(index + 1)
    candidate_string += "] "
    candidate_string += title + ", "
    candidate_string += year + ", "
    candidate_string += genre + "\n"
    return candidate_string


# # Netflix
# def construct_prompting(item_attribute, item_list, candidate_list): 
#     # make history string
#     history_string = "User history:\n" 
#     for index in item_list:
#         year = item_attribute['year'][index]
#         title = item_attribute['title'][index]
#         history_string += "["
#         history_string += str(index)
#         history_string += "] "
#         history_string += str(year) + ", "
#         history_string += title + "\n"
#     # make candidates
#     candidate_string = "Candidates:\n" 
#     for index in candidate_list:
#         year = item_attribute['year'][index.item()]
#         title = item_attribute['title'][index.item()]
#         candidate_string += "["
#         candidate_string += str(index.item())
#         candidate_string += "] "
#         candidate_string += str(year) + ", "
#         candidate_string += title + "\n"
#     # output format
#     output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
#     # make prompt
#     # prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
#     prompt = ""
#     prompt += history_string
#     prompt += candidate_string
#     prompt += output_format
#     return prompt




# # chatgpt
def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                augmented_sample_dict):
    if index in augmented_sample_dict:
        print(f"g:{index}")
        return 0
    else:
        try:
            print(f"{index}")
            prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])
            # url = "http://llms-se.baidu-int.com:8200/chat/completions"
            # url = "https://api.openai.com/v1/completions"
            url = "https://api.openai.com/v1/chat/completions"

            headers = {
                # "Content-Type": "application/json",
                # "Authorization": "Bearer your key"

            }
            # params={
            #     "model": model_type,
            #     "prompt": prompt,
            #     "max_tokens": 1024,
            #     "temperature": 0.6,
            #     "stream": False,
            # }

            params = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "system",
                              "content": "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"},
                             {"role": "user", "content": prompt}]
            }

            response = requests.post(url=url, headers=headers, json=params)
            message = response.json()

            content = message['choices'][0]['message']['content']
            # content = message['choices'][0]['text']
            print(f"content: {content}, model_type: {model_type}")
            samples = content.split(seperator)
            pos_sample = int(samples[0])
            neg_sample = int(samples[1])
            augmented_sample_dict[index] = {}
            augmented_sample_dict[index][0] = pos_sample
            augmented_sample_dict[index][1] = neg_sample
            # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
            # pickle.dump(augmented_sample_dict, open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_sample_dict','wb'))
            pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict', 'wb'))

        # except ValueError as e:
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            # time.sleep(40)
        except ValueError as ve:
            print("An error occurred while parsing the response:", str(ve))
            # time.sleep(40)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict)
        except KeyError as ke:
            print("An error occurred while accessing the response:", str(ke))
            # time.sleep(40)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            # time.sleep(40)


        # except ValueError as e:
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(8)
            # print(content)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                        augmented_sample_dict)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            time.sleep(10)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # print(content)
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                        augmented_sample_dict)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            time.sleep(10)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # print(content)
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                        augmented_sample_dict)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            time.sleep(10)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # # print(content)
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict)
            # return 1
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                        augmented_sample_dict)
        except EOFError as ke:
            print("EOFError: : Ran out of input error occurred while accessing the response:", str(ke))
            time.sleep(10)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # print(content)
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                        augmented_sample_dict)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(10)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # print(content)
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type,
                        augmented_sample_dict)
        return 1

def LLM_request(uid, interaction_info, user_attribute, item_attribute, candidate_indices_dict, pipeline, terminators, cnt):
    candidate_list = candidate_indices_dict[str(uid)]

    cnt += 1
    if cnt >= 10:
        with open(error_name, 'a') as file:
            file.write(f'{uid}: {candidate_list}' + '\n')
        print(f'Errors when augmenting for user {uid}')
        return
    item_list, _ = interaction_info
    # 如果电影长度超过100，则取最后100个
    if len(item_list) > 100:
        item_list = item_list[-100:]


    prompting = construct_prompting(uid, user_attribute, item_attribute, item_list, candidate_list)

    messages = [
        {"role": "system", "content": "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"},
        {"role": "user", "content": prompting},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=20000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    rst = outputs[0]["generated_text"][-1]['content']

    try:
        llm_aug_rst = rst.split(seperator)
        if len(llm_aug_rst) != 2:
            raise Exception("返回异常")
        else:
            for aug_iid in llm_aug_rst:
                if aug_iid not in candidate_list:
                    raise Exception("返回异常")

        with open(file_name, 'a') as file:
            file.write(f'{uid}: {rst}' + '\n')
        print(f'Results of user {uid} have been written')
    except Exception as ex:
        print(f"An return {rst} error occurred")
        LLM_request(uid, interaction_info, user_attribute, item_attribute, candidate_indices_dict, pipeline, terminators, cnt)

### read candidate
candidate_indices_dict = pickle.load(open(file_path + 'candidate_items_BPR', 'rb'))
aug_err_user_list = []
with open(file_path + 'ml_100k_error.txt', 'r') as file:
    for line in file:
        data_str = line.strip()
        key, value_str = data_str.split(': ', 1)
        aug_err_user_list.append(key)

### read item_attribute
user_attribute = pd.read_csv(dataset_path + 'ml-100k.user',
                             names=['user_id:token', 'age:token', 'gender:token', 'occupation:token', 'zip_code:token'],
                             sep="\t",
                             skiprows=1)[['user_id:token', 'age:token', 'gender:token', 'occupation:token']]\
                    .rename(columns={'user_id:token': 'uid', 'age:token': 'age',
                                     'gender:token': 'gender', 'occupation:token': 'occupation', })\
                    .to_dict()
item_attribute = pd.read_csv(dataset_path + 'ml-100k.item',
                             names=['item_id:token', 'movie_title:token_seq', 'release_year:token', 'class:token_seq'],
                             sep="\t",
                             skiprows=1)[['item_id:token', 'movie_title:token_seq',
                                          'class:token_seq', 'release_year:token']]\
                    .rename(columns={'item_id:token': 'iid', 'movie_title:token_seq': 'title',
                                     'class:token_seq': 'gender', 'release_year:token': 'year'})\
                    .to_dict()

### read user interactions
user_interactions = pd.read_csv(dataset_path + 'ml-100k.inter',
                                names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'],
                                sep="\t",
                                skiprows=1)\

interactions = user_interactions[['user_id:token', 'item_id:token', 'rating:float']]\
                .rename(columns={'user_id:token': 'uid', 'item_id:token': 'iid', 'rating:float': 'rating'})

grouped = interactions.groupby('uid').apply(lambda x: [x['iid'].tolist(), x['rating'].tolist()]).to_dict()

file_name = '/home/dyt/coldstart/RecBole/ml_100k_augmented_inter_p1_n1.txt'
error_name = '/home/dyt/coldstart/RecBole/ml_100k_error.txt'

rst_dict = {}

#正常的生成过程
# for k, v in grouped.items():
    # if k <= 450:
    #     LLM_request(k, v, user_attribute, item_attribute, candidate_indices_dict)

    # if k > 19:
    #     pipeline = transformers.pipeline(
    #         "text-generation",
    #         model=model_dir,
    #         model_kwargs={"torch_dtype": torch.float16},
    #         device_map="auto",
    #     )
    #     terminators = [
    #         pipeline.tokenizer.eos_token_id,
    #         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #     ]
    #     cnt = 0
    #     LLM_request(k, v, user_attribute, item_attribute, candidate_indices_dict, pipeline, terminators, cnt)


# 对生成失败的进行重试
for uid in aug_err_user_list:
    uid = int(uid)
    v = grouped[uid]
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto",
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    cnt = 0
    LLM_request(uid, v, user_attribute, item_attribute, candidate_indices_dict, pipeline, terminators, cnt)

