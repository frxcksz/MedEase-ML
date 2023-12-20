import numpy as np
import pandas as pd
import cv2
from glob import glob
import spacy
import re
import string
import tensorflow as tf
import keras_ocr
import matplotlib.pyplot as plt
import os
import json

import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model_ner = spacy.load('./models/NER/spaCy')

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
   
    return str(removepunctuation)

def perform_ocr(input_image):
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [input_image]
    predictions = pipeline.recognize(images)
    return predictions[0]

def preprocess_image(image_path):
    resize = resize_images(image_path)  # 1920 x 1080
    saturation = adjust_saturation(resize)  # 2
    contrast = adjust_contrast(saturation)  # 2
    return contrast

def resize_images(input_image, target_size=(1920, 1080)):
    resized_image = tf.image.resize_with_pad(input_image, target_size[0], target_size[1])
    resized_image = tf.cast(resized_image, tf.uint8)
    return resized_image

def adjust_saturation(input_image):
    saturated_image = tf.image.adjust_saturation(input_image, saturation_factor=2)
    saturated_image = tf.cast(saturated_image, tf.uint8)
    return saturated_image

def adjust_contrast(input_image):
    contrast_image = tf.image.adjust_contrast(input_image, contrast_factor=2)
    contrast_image = tf.cast(contrast_image, tf.uint8)
    return contrast_image

# group the label
class groupgen():
    def __init__(self):
       self.id = 0
       self.text = ''
    def getgroup(self, text):
       if self.text == text:
          return self.id
       else:
          self.id += 1
          self.text = text
          return self.id

grp_gen = groupgen()

def getPredictions(image):
    img = keras_ocr.tools.read(image)
   
    ocrData = perform_ocr(img)
    text_from_ocr = " ".join([text_result[0] for text_result in ocrData])
    print(text_from_ocr)

    entities = dict(NAME=[], ING=[], TYPE=[], DES=[], ORG=[])

    if len(text_from_ocr) > 1:
        columns = ['image_path', 'text', 'boxes']
        df = pd.DataFrame(columns=columns)

        print(type(df))

        for text_result in ocrData:
            text = text_result[0]
            boxes = text_result[1]
            
            df = df.append({'image_path': image, 'text': text, 'boxes': boxes}, ignore_index=True)
        
        df.dropna(inplace=True)
        df['text'] = df['text'].apply(cleanText)
        df_clean = df[df['text'] != ""]

        doc = model_ner(text_from_ocr)
        for ent in doc.ents:
            print(f"Entity: {ent.text}, Label: {ent.label_}")

        docjson = doc.to_json()
        docjson.keys()

        doc_text = docjson['text']

        dataframe_tokens = pd.DataFrame(docjson['tokens'])
        dataframe_tokens['token'] = dataframe_tokens[['start', 'end']].apply(lambda x: doc_text[x[0]:x[1]], axis=1)

        right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
        dataframe_tokens = pd.merge(dataframe_tokens, right_table, how='left', on='start')
        dataframe_tokens.fillna('O', inplace=True)

        df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
        df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)

        dataframe_info = pd.merge(df_clean, dataframe_tokens[['start', 'token', 'label']], how='inner', on='start')

        bb_df = dataframe_info.query("label != 'O' ")

        def custom_sort(row):
            label = row['label']
            start = row['start']
            if label.startswith('B-NAME'):
                return (0, start, label)
            elif label.startswith('I-NAME'):
                return (1, start, label)
            else:
                return (2, start, label)
            
        bb_df['sorting_key'] = bb_df.apply(custom_sort, axis=1)
        df_sorted = bb_df.sort_values(by='sorting_key').drop('sorting_key', axis=1).reset_index(drop=True)

        df_sorted['label'] = df_sorted['label'].apply(lambda x: x[2:])
        df_sorted['group'] = df_sorted['label'].apply(grp_gen.getgroup)

        df_sorted[['left', 'top', 'right', 'bottom']] = pd.DataFrame(df_sorted['boxes'].apply(
            lambda boxes: (int(boxes[0][0]), int(boxes[0][1]), int(boxes[2][0]), int(boxes[2][1]))
        ).tolist(), index=df_sorted.index)

        col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
        group_tag_img = df_sorted[col_group].groupby(by='group')

        img_tagging = group_tag_img.agg({
            'left': min,
            'right': max,
            'top': min,
            'bottom': max,
            'label': lambda x: ', '.join(np.unique(x)),
            'token': lambda x: " ".join(x)
        })

        img_bb = img.copy()

        for l, r, t, b, label, token in img_tagging.values:
            cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(img_bb, str(label), (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        
        img_bb = cv2.cvtColor(img_bb, cv2.COLOR_BGR2RGB)
        
        info_array = img_tagging[['token', 'label']].values

        for token, label in info_array:
            label_tag = label
            entities[label_tag].append(token)
    else:
        img_bb = img
    
    return img_bb, entities

image_path = './tmp/pred_img.jpg'
img_results, entities = getPredictions(image_path)

print(entities)

json_output_path = './result/pred.json'
with open(json_output_path, 'w') as json_file:
    json.dump(entities, json_file, indent=2)

print(f"Entities exported to {json_output_path}")

cv2.namedWindow('predictions')
cv2.imshow('predictions', img_results)
cv2.waitKey(0)
cv2.destroyAllWindows()