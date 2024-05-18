import csv
import glob
import numpy as np

import ipdb
st = ipdb.set_trace

# print(glob.glob("./MT_images/*/*.png"))
import re

import os

seed = 42
np.random.seed(seed)

def format_class_name(name):
    if name=="TVStand":
        formatted = "television stand"
    elif name=="CounterTop":
        formatted = "countertop"
    else:
        formatted = re.sub(r"(?<=\w)([A-Z])", r" \1", name).lower()
    return formatted

save_dir = "./metrics" # save to metrics

root1 = 'tidy_task_TIDEE_full'
# root2 = 'tidy_task_TIDEE_nomemex'

folder1 = 'cleanup'
folder2 = 'cleanup'

url_path = "https://.s3.amazonaws.com" # path to aws S3

filename = f"{save_dir}/{root1}-{folder1}=0_{root2}-{folder2}=1.csv"

images1 = glob.glob(f"{root1}/{folder1}/*/*/*.png")
images2 = glob.glob(f"{root2}/{folder2}/*/*/*.png")

fields = ['name', 'url1', 'url2', 'rec1', 'rec2', 'order']

image_dict = {}

lines = []
for image1_name in images1:
    path_split = image1_name.split('/')
    obj_name1 = path_split[-1].split('-')[0]
    n1 = path_split[-2]
    for image2_name in images2:
        path_split = image2_name.split('/')
        obj_name2 = path_split[-1].split('-')[0]
        n2 = path_split[-2]
        if obj_name1==obj_name2 and n1==n2:
            # image_dict[image1_name] = image2_name
            obj_name = format_class_name(obj_name1.split('_')[0])
            rand_choice = np.random.uniform(0,1)
            rec1 = format_class_name(image1_name.split('-')[1].split('.')[0])
            rec2 = format_class_name(image2_name.split('-')[1].split('.')[0])
            image1_name = os.path.join(url_path, image1_name)
            image2_name = os.path.join(url_path, image2_name)
            if rand_choice>0.5:
                line = [obj_name, image1_name, image2_name, rec1, rec2, '[0,1]']
            else:
                line = [obj_name, image2_name, image1_name, rec2, rec1, '[1,0]']
            lines.append(line)
            break
    
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 

    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(lines)


# st()

