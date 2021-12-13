from fastai.vision.all import *
import fastai
import timm
import torch
from sklearn.metrics import cohen_kappa_score,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import numpy as np
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,help="name of the user")
ap.add_argument("-i", "--iteration", required=True,help="iteration")
ap.add_argument("-b", "--bs", required=True,help="batch size")
ap.add_argument("-l", "--lr", required=True,help="learning rate")
args = vars(ap.parse_args())

device = 1
bs = int(args["bs"])
name = args["name"]
i = int(args["iteration"])-1
lr = float(args["lr"])
torch.cuda.set_device(device)

results = []
resultsTTA = []



df = pd.read_csv('../trainValid'+str(i)+'.csv')

def is_valid(path):
    name = path[1]
    return (df[df['nombre_foto']==name])['set'].values[0]=='validation'


db = DataBlock(blocks = (ImageBlock, CategoryBlock),
             splitter=FuncSplitter(is_valid),
             get_x = ColReader(1,pref="../amd/"),
             get_y=ColReader(3),
             item_tfms = [Resize(640)], # CropPad(200,200)
             batch_tfms=[*aug_transforms(size=512, min_scale=0.75,do_flip=True,flip_vert=True,
                  max_rotate=2.,max_zoom=1.1, max_warp=0.05,p_affine=0.9, p_lighting=0.8), 
                         Normalize.from_stats(*imagenet_stats)])

callbacks = [
    ShowGraphCallback(),
    EarlyStoppingCallback(patience=5),
    SaveModelCallback(fname=name+'_512b_'+str(i),monitor='f1_score'),
    #ReduceLROnPlateau(patience=2)
]

dls = db.dataloaders(df.values,bs=bs,num_workers=0)

learn = Learner(dls,timm.create_model(name,num_classes=2,pretrained=True,drop_rate=0.5),
            metrics=[accuracy,Precision(),Precision(pos_label=0),Recall(),Recall(pos_label=0),F1Score(),RocAucBinary(),CohenKappa()],
            cbs=callbacks,
            loss_func= FocalLossFlat()).to_fp16()
learn.load(name+'_'+str(i))
learn.fine_tune(100,base_lr=lr)


dfTest = pd.read_csv('../test'+str(i)+'.csv')
dfTest.set = dfTest.set.apply(lambda x : 'test')
dfTest = pd.concat([df[df.set=='validation'],dfTest])

def is_test(path):
    name = path[1]
    return (dfTest[dfTest['nombre_foto']==name])['set'].values[0]=='test'

dbTest = DataBlock(blocks = (ImageBlock, CategoryBlock),
             splitter=FuncSplitter(is_test),
             get_x = ColReader(1,pref="../amd/"),
             get_y=ColReader(3),
             item_tfms = [Resize(640)], # CropPad(200,200)
             batch_tfms=[*aug_transforms(size=512, min_scale=0.75,do_flip=True,flip_vert=True,
                  max_rotate=2.,max_zoom=1.1, max_warp=0.05,p_affine=0.9, p_lighting=0.8), 
                         Normalize.from_stats(*imagenet_stats)])
dlsTest = dbTest.dataloaders(dfTest.values,bs=8,num_workers=0)
learn.dls = dlsTest
res = learn.validate()

resTTA = learn.tta()
accuracy = accuracy_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist())
precision = precision_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist())
npv = precision_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist(),pos_label=0)
recall = recall_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist())
specificity = recall_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist(),pos_label=0)
f1score = f1_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist())
roc = roc_auc_score(resTTA[1],resTTA[0][:,1])
cohen = cohen_kappa_score(resTTA[1],np.argmax(resTTA[0],axis=1).tolist(),weights='quadratic')
resTTA = [name,i,accuracy,precision,npv,recall,specificity,f1score,roc,cohen]

df = pd.read_csv('resultsDMAE/resultsTTA512.csv')
df = df.append(pd.DataFrame([resTTA], columns=['name','iteration','accuracy','precision','npv','recall','specificity','f1-score','AUROC','Cohen']))
df.to_csv('resultsDMAE/resultsTTA512.csv',index=None)


df = pd.read_csv('resultsDMAE/results512.csv')
df = df.append(pd.DataFrame([np.append([name,i],res[1:])], columns=['name','iteration','accuracy','precision','npv','recall','specificity','f1-score','AUROC','Cohen']))
df.to_csv('resultsDMAE/results512.csv',index=None)

