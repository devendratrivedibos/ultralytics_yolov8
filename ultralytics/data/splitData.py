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


filename = []
##K                ###y
#filename.extend(onlyJpg_withSpecificClass(r'X:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/47547a87-0dcd-47e1-ae30-42cc7b508d46/'))  ##F

filename.extend(onlyJpg_with_Text(r'K:/62280af4-b90c-4f80-b984-bce0aaf1cd9d/data_set/0341ae19-ccf7-4cf5-9a52-14f4f9319e0a/'))  ##K  ##Y
filename.extend(onlyJpg_with_Text(r'K:/62280af4-b90c-4f80-b984-bce0aaf1cd9d/data_set/472bc333-a91a-482d-9389-8dbe781659d3/'))  ##K  ##y
filename.extend(onlyJpg_with_Text(r'K:/62280af4-b90c-4f80-b984-bce0aaf1cd9d/data_set/37613c87-e457-432e-a1f1-001f547801fe/'))  ##K
filename.extend(onlyJpg_with_Text(r'K:/62280af4-b90c-4f80-b984-bce0aaf1cd9d/data_set/b7793911-1ac0-4c10-8d2d-75a17706ea95/'))  ##K

###E       #### W
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/0a507f0f-684f-4be5-9ec3-405a6c732bd8/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/2d5f2a9f-12a6-42c3-9cea-0344740f5681/data_set/'))  ##E
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/6c296df1-2a5b-467b-9954-e3e8875a6316/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/98fdd7c1-b9b5-42f4-8d6f-cb977088af08/data_set'))##E
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/220fd4ef-9cf3-401f-b1ba-f57d87fe3740/data_set'))##E
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/472649e4-adc4-4058-9d8c-a0e512279920/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/79180851-2989-4c49-bb65-022cb0346691/data_set'))##E

####X    F
filename.extend(onlyJpg_with_Text(r'F:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/data_set/c1797a98-401b-4780-84c9-578cdb40798d'))   ###F  X
filename.extend(onlyJpg_with_Text(r'F:/6d09b9ce-12aa-41ce-9e6b-2d240aba3f1d/data_set/9ab38a41-4c84-47a4-b173-91e3864a2968'))  

###E     ###W
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/0f2bd68e-ddd9-4345-905f-563a573cb181/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/1ee773ea-3eef-46f4-aa89-4fafc84079c4/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/1ff167d8-2c39-432d-bf6e-9ec86254f144/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/3da57355-5d6d-472b-b6c1-bde6b8da0afe/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/7c522331-cea5-4438-9f34-f87882bfbec3/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/7dbf58e5-8df2-42c3-a9ee-810d8ea3a055/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/7e90c075-a20c-4599-898a-6dc40e646ff8/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/8d376bba-4551-44da-ba6b-af1121e89140/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/9e95d4dd-22a4-4fda-8df1-91ea25d54a1c/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/24f62a37-8265-48ec-8d45-3f64a58c1ee8/data_set/'))##E
# filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/26b31163-95ce-44fe-aa0e-ea83b14c335c/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/29acd919-f31c-4391-85a9-c0f5abfc8815/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/447bc25b-6544-47f1-87ab-fb163328a743/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/04252263-7a1f-4bb8-8cc3-79b143ff0fd1/data_set/')*2)##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/34418854-f2b0-47a5-9e48-cfb2f09c8b85/data_set/')*2)##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/528be337-80cd-4ff9-bd3e-0914efc3b6a5/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/6294f9ed-66c8-4929-88f9-57a0d529aeeb/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/70175ead-5f4b-4cb9-bdf7-4ae80fa040f7/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/cdd0768c-5d7e-4203-9c9d-404eb746bbd8/data_set'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/d746fc53-8f6b-4bb1-ad95-942f1e6a3060/data_set'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/dff501cb-b397-4db5-b398-bee1628df8f9/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/e9a56dc7-a2d1-4fa0-b57e-e97d614700f1/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/f0380ee0-4f22-4961-b6a2-2a7a2423092b/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/f76997cc-f98f-454b-a22f-bb95256a434e/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'E:/86d22c2a-45f4-4f3c-8366-59c45ab1b046/fad85044-a957-4740-bcf4-689f74cd5d0a/data_set/'))##E

##F        ##X
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/00d57cf3-b5a7-4eae-98d6-faaa4645f8d5/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/7d007d45-2ed6-4af5-9312-94c8f0ab0bb4/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/9cd5e70a-9405-48e7-a214-6067812e50c1/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/13b4e6df-b9a2-4b17-bd92-a98d1d91a8ce/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/30a0f828-927f-4569-a140-1a28d96f976e/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/942e64ac-1cc5-419b-8ca6-e61e016c07b1/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/7509abb5-f4dd-483a-ab5b-951586a61a5b/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/aaa1dd5e-ed97-4a1e-b540-33e3099c92f7/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/97e33780-ef2a-4a9b-a0ac-3936be54547f/data_set/afc23a41-9e01-408d-8814-905a23e30f99/')*2) ##F

filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/2d8b0eb6-a52e-476b-af18-ac85966867f2/'))   ##F
filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/55fbc7eb-4e26-47ea-8305-b65049fb46d8/'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/47547a87-0dcd-47e1-ae30-42cc7b508d46'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/bb8e4dc9-fb2a-44be-9c1f-2a7d61e67724'))  ##F          SPECIFIC CLASSSS
filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/d1a169cd-d7cd-4a65-ae93-cd2005570d94'))  ##F
filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/d9022249-5ec8-4b46-b0d8-fcc7c2d897b8'))
filename.extend(onlyJpg_with_Text(r'F:/68e5daed-4f29-492f-988a-45e602ac946a/data_set/e66a54c3-298a-4147-ad4a-d88dec3fa9c0'))



filename.extend(onlyJpg_with_Text(r'F:/roadFurniture_lowdata/2e8e35cf-c5b6-4e8b-b91a-a047664af0ef'))
filename.extend(onlyJpg_with_Text(r'F:/roadFurniture_lowdata/010f2a49-f912-4f61-a667-319e17738efe'))
filename.extend(onlyJpg_with_Text(r'F:/roadFurniture_lowdata/0341ae19-ccf7-4cf5-9a52-14f4f9319e0a'))
filename.extend(onlyJpg_with_Text(r'F:/roadFurniture_lowdata/1181f41f-0263-48d0-8b0e-478677b08328'))
filename.extend(onlyJpg_with_Text(r'F:/roadFurniture_lowdata/01510007-7caf-4728-9be7-ed446f5d0569'))
filename.extend(onlyJpg_with_Text(r'F:/roadFurniture_lowdata/New folder'))

filename.extend(onlyJpg_with_Text(r'F:/ee74a59d-0345-480c-9853-18e3ef55d780/a5c63eb8-49e1-4932-b6ef-d5a65927c5ad/data_set/')*2)##E
filename.extend(onlyJpg_with_Text(r'F:/ee74a59d-0345-480c-9853-18e3ef55d780/8b38d31c-becb-445c-bb11-6eb5f61ed24d/data_set')*2)##E


filename.extend(onlyJpg_with_Text(r'K:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/03ddd'))        
filename.extend(onlyJpg_with_Text(r'K:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/010f2'))
filename.extend(onlyJpg_with_Text(r'K:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/73af_inner'))
filename.extend(onlyJpg_with_Text(r'K:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/20314'))
filename.extend(onlyJpg_with_Text(r'K:/c578c212-6800-4fe6-a5e7-bbd8192c23b2/bf5eff8'))

filename.extend(onlyJpg_with_Text(r'F:/56e1fce2-4e22-4db2-978f-6038d366b759/7345e1e3-9c93-4d52-a70c-5956edb8f04c/data_set/'))##E
filename.extend(onlyJpg_with_Text(r'F:/56e1fce2-4e22-4db2-978f-6038d366b759/d2294581-3d6b-4a97-a2ad-d655f556175b/data_set/'))##E




print(len(filename))
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
shuffle(filename)
train = filename[:26000]
val = filename[24000:]


with open('Train_22mar.txt', 'w+') as file:
    for f in train:
        file.write(f + '\n')
file.close()

with open('Val_22mar.txt', 'w+') as file:
    for f in val:
        file.write(f + '\n')
file.close()