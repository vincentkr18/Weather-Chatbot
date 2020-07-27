#imports
from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import nltk
#from gtts import gTTS
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import os
language = 'en'
from keras.models import load_model
import termcolor
import tensorflow as tf
from keras.models  import model_from_json
import pandas as pd
import numpy
#import tflearn
#import tensorflow
import random

import spacy
from spacy import displacy
from collections import Counter
#import en_core_web_sm
#nlp = en_core_web_sm.load()
import pprint
import requests

import pyodbc
import yaml

from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Text, String, DateTime, Float
import time
import pyodbc
import keras

import warnings
import warnings
warnings.filterwarnings("ignore")

import spacy
from spacy import displacy
import en_core_web_sm

nlp = en_core_web_sm.load()
import json

app = Flask(__name__)

import json

with open('indents_2.json') as file:
    data = json.load(file)

# In[3]:


words = []
labels = []
docs_x = []
docs_y = []

# In[4]:


for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# In[5]:


words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# In[6]:


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words1 = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words1]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return pd.DataFrame(numpy.array(bag)).T, s_words1


# In[36]:


# api.openweathermap.org/data/2.5/weather?q={0}&appid={1}
def weather_extract(city, filter_value, *args):
    API_TOKEN = ''

    response = requests.get("http://api.openweathermap.org/data/2.5/weather?q={0}&appid={1}".format(city, API_TOKEN))
    if filter_value == 'WindSpeed':
        output = response.json()['wind']
        if 'deg' in response.json()['wind']:
            output = "The Wind Speed at {} is {} m/s and its direction is {} degree".format(city.title(), str(
                response.json()['wind']['speed']), str(response.json()['wind']['deg']))
        else:
            output = "The Wind Speed at {} is {} m/s".format(city.title(), str(response.json()['wind']['speed']))
    else:
        value = round(int(response.json()['main']['temp']) - 273.15, 2)
        value = str(value) + '°C'
        output = "The weather in {} is {}".format(city.title(), value)
    # else filter_value =  'Generic':
    # output = response.json()
    # response.json()
    return output, response.json()


# In[ ]:


def weather_options(inp, s_words):
    inp = inp.lower()
    windspeed = ["wind", "speed"]
    temperature = ["solar", 'temperature']

    if any(x in inp for x in windspeed):
        filter_value = 'WindSpeed'
    elif any(x in inp for x in temperature):
        filter_value = 'temperature'
    else:
        filter_value = 'Generic'

    doc = nlp(inp)
    Named_Entity = [(X.text, X.label_) for X in doc.ents]
    location = []
    cardinal = []
    for i in Named_Entity:
        print(i[1])
        if i[1] == 'GPE':
            location.append(i[0])
        elif i[1] == 'CARDINAL':
            cardinal.append(i[0])
    #print('Locaion 0', location[0])
    if not location:
        output = 'I dont have an answer. I will update my team on this!'
    else:
        output, response = weather_extract(location[0], filter_value)

    return output


# In[ ]:


import pyodbc


# In[ ]:


def carbon_footprint_extract(emission):
    server =
    database =
    username =
    password =
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    engine = create_engine()

    df = pd.read_sql_table("", engine)
    print('Table has been read')
    if emission == 'Scope1':
        value = df['Scope1'][0]
        company = df['Company'][0]
        output = 'The overall Scope 1 emissions of {} is {} tonnes CO2'.format(company, value)
    elif emission == 'Scope2':
        value = df['Scope2'][0]
        company = df['Company'][0]
        output = 'The overall Scope 2 emissions of {} is {} tonnes CO2'.format(company, value)
    elif emission == 'Scope3':
        value = df['Scope3'][0]
        company = df['Company'][0]
        output = 'The overall Scope 3 emissions of {} is {} tonnes CO2'.format(company, value)
    elif emission == 'overall':
        value = df['Annual'][0]
        company = df['Company'][0]
        output = 'Your Net emissions ({}) is {} tonnes CO2'.format(company, value)
    else:
        output = 'Cannot find, please elaborate'
    return output


# In[ ]:


def carbon_footprint(inp, s_words):
    emission = []
    # for i in s_words:
    if '1' in inp:
        # print(i)
        emission = 'Scope1'
    elif '2' in inp:
        emission = 'Scope2'
    elif '3' in inp:
        emission = 'Scope3'
    else:
        emission = 'overall'
    # print(emission)
    output = carbon_footprint_extract(emission)

    return output


# In[ ]:


def execute_by_tag(tag, inp, s_words):
    if tag == 'weather':
        output = weather_options(inp, s_words)
    elif tag == 'carbonfootprint':
        output = carbon_footprint(inp, s_words)
    else:
        print('None')
    return output


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/get")
#function for the bot response
def get_bot_response():
    inp = request.args.get('msg')
    inp_df, s_words = bag_of_words(inp, words)

    global graph
    graph = tf.get_default_graph()

    model = load_model("Advaned_Chatbot_model.h5")

    with graph.as_default():
            results = model.predict([inp_df])

    results_index = numpy.argmax(results)

    tag = labels[results_index]

    custom = ['weather', 'carbonfootprint']
    if tag in custom:
        print(tag)
        output = execute_by_tag(tag, inp, s_words)
        #print(termcolor.colored('RoBot:', color='red', attrs=['bold']), output)
    else:
        doc = nlp(inp)
        # print([(X.text, X.label_) for X in doc.ents])
        #displacy.render(doc, jupyter=True, style='ent')

        for tg in data["intents"]:
            if tg['tag'] == tag:
                print(tag)
                responses = tg['responses']
                output = random.choice(responses)
    return str(output)

if __name__ == "__main__":
    app.run()