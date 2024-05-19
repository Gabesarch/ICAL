import os
import base64
import requests
from io import BytesIO
import os
from openai import AzureOpenAI
import ipdb
st = ipdb.set_trace

# Get OpenAI API Key from environment variable
# api_key = os.environ["OPENAI_API_KEY"]
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}"
# }

client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),  
        api_version = "2023-05-15",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

metaprompt = '''
- For any marks mentioned in your answer, please highlight them with []. For example, "the (blank) is in [1]", or "the (blank) is in [5], [12]". The marks associated with a mask will be directly centered on the mask it is associated with. In other words, the marks for the masks will not be next to the mask, but directly centered on the mask that it is associated with.
'''    

metaprompt = '''
- For any marks mentioned in your answer, please highlight them with [].
'''  

# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def prepare_inputs(message, image):

#     # # Path to your image
#     # image_path = "temp.jpg"
#     # # Getting the base64 string
#     # base64_image = encode_image(image_path)
#     base64_image = encode_image_from_pil(image)

#     payload = {
#         "model": "gpt-4-vision-preview",
#         "messages": [
#         {
#             "role": "system",
#             "content": [
#                 metaprompt
#             ]
#         }, 
#         {
#             "role": "user",
#             "content": [
#             {
#                 "type": "text",
#                 "text": message, 
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                 "url": f"data:image/jpeg;base64,{base64_image}"
#                 }
#             }
#             ]
#         }
#         ],
#         "max_tokens": 800
#     }

#     return payload

# def request_gpt4v(message, image):
#     payload = prepare_inputs(message, image)
#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#     res = response.json()['choices'][0]['message']['content']
#     return res

def prepare_inputs(message, image, in_context=[]):

    base64_image = encode_image_from_pil(image)

    if len(in_context)>0:
        
        messages = [
            {
                "role": "system",
                "content": [
                    metaprompt
                ]
            },
            # {
            #     "role": "user",
            #     "content": [
            #     {
            #         "type": "text",
            #         "text": "", 
            #     },
            #     {
            #         "type": "image_url",
            #         "image_url": {
            #         "url": f"data:image/jpeg;base64,{base64_image}"
            #         }
            #     }
            #     ]
            # }
        ]


        for in_context_idx in range(len(in_context)):
            # text = ""
            # if in_context_idx==0:
            #     text += 'Your task is to identify a query in the image. ' 
            query_ic = in_context[in_context_idx][0]
            answer = in_context[in_context_idx][2]
            text = query_ic #f'Question: Where can I find "{query_ic.lower()}"? Think step-by-step. The mark should be directly on the query in the image (not next to it, etc.). You should only output the mark of the location of the query itself.'
            base64_image_ic = encode_image_from_pil(in_context[in_context_idx][1])
            dict_ic = {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": text, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_ic}"
                    }
                }
                ]
            }
            
            messages.append(dict_ic)
            assistant_text = answer #f'Answer: [{answer}]'
            dict_feedback = {
                "role": "assistant", "content": assistant_text
            }
            messages.append(dict_feedback)
            print(f'in-context example #{in_context_idx}')
            print(text)
            print(assistant_text)

        base_dict = {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": message, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
        messages.append(base_dict)
        print(f'Question: {message}')

    else:
        messages = [
            {
                "role": "system",
                "content": [
                    metaprompt
                ]
            }, 
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": message, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ]

    return messages

def request_gpt4v(message, image, in_context=[]):
    messages = prepare_inputs(message, image, in_context)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=3000,
        temperature=0.,
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content
