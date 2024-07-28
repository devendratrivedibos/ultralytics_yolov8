import os
import glob
from random import shuffle

def onlyJpg_with_Text(directory):
    l = []
    txt_names = {os.path.splitext(filename)[0] for filename in os.listdir(directory) if filename.endswith(".txt")}
    img_names = {os.path.splitext(filename)[0] for filename in os.listdir(directory) if filename.endswith(".jpg")}
    print(f"Text Length is {len(txt_names)} , images length {len(img_names)} in {directory} \n\n")
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            jpg_name = os.path.splitext(filename)[0] 
            if jpg_name in txt_names:
                filepath = os.path.join(directory, filename)
                l.append(filepath)
    return l

# V = D
# W = E
# X= F
# Y = K
# Z = C

filename = []
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/data_set_median/0a507f0f-684f-4be5-9ec3-405a6c732bd8'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/data_set_median/fad85044-a957-4740-bcf4-689f74cd5d0a'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/data_set_median/f0380ee0-4f22-4961-b6a2-2a7a2423092b'))
print(len(filename))

filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/data_set_median/d3c554b8-b7f3-4f73-ad13-3eed93b91045'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/data_set_median/7dbf58e5-8df2-42c3-a9ee-810d8ea3a055'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/data_set_median/7c522331-cea5-4438-9f34-f87882bfbec3'))
print(len(filename))

filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set_median/bb8e4dc9-fb2a-44be-9c1f-2a7d61e67724'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set_median/00d57cf3-b5a7-4eae-98d6-faaa4645f8d5'))
print(len(filename))

filename.extend(onlyJpg_with_Text(r'D:/0f9fa7a7-dd18-43d8-b437-b059b8141e40/data_set_median/4168c0f0-c0fa-414a-9c9a-3df867388238'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'D:/0f9fa7a7-dd18-43d8-b437-b059b8141e40/data_set_median/4f8b7bf3-2927-4560-ada7-0fe4b2f6b24d'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'D:/0f9fa7a7-dd18-43d8-b437-b059b8141e40/data_set_median/b6bf1b75-0cb8-4965-a353-7381a89d01f7'))
print(len(filename))

filename.extend(onlyJpg_with_Text(r'Y:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/data_set_median/20314658-9b00-47f4-ae21-f939e8091578'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'Y:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/data_set_median/d1677ddf-e380-4338-a86f-aeaea2cd02e9'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'Y:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/data_set_median/5b6d9356-388a-44d3-b2ce-8a712daffa07'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'Y:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/data_set_median/bf5eff81-fe74-4fd4-99d5-863d6a5f2e9b'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'Y:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/data_set_median/127409d7-7e5b-42b7-9686-e089de21bb4f'))
print(len(filename))
filename.extend(onlyJpg_with_Text(r'Y:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/data_set_median/f26f6dc6-3066-4a33-b269-a817650ede62'))
print(len(filename))

shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
trainF = int(0.8 * len(filename))
train = filename[:trainF]
val = filename[trainF:]


with open('Train_2mar_median.txt', 'w+') as file:
    for f in train:
        file.write(f + '\n')
file.close()

with open('Val_2mar_median.txt', 'w+') as file:
    for f in val:
        file.write(f + '\n')
file.close()