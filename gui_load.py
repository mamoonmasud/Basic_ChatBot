import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *

model = load_model('C:/Users/mamoo/Desktop/Python_Coding/ChatBot/chatbot_model.h5')

word_file = open('C:/Users/mamoo/Desktop/Python Coding/ChatBot/intents.json').read()
intents = json.loads(word_file)
words= pickle.load(open('C:/Users/mamoo/Desktop/Python_Coding/ChatBot/words.pkl', 'rb'))
classes = pickle.load(open('C:/Users/mamoo/Desktop/Python_Coding/ChatBot/classes.pkl', 'rb'))

#Defining a function to tokenize the input sentence, and then apply lemmatizer on the words!
def senten_clean (sentence):
    #Tokenizing the pattern & Splitting the words into array
    word_senten = nltk.word_tokenize(sentence)
    #Stemming every word
    word_senten = [lemmatizer.lemmatize(word.lower()) for word in word_senten]
    
    return word_senten

# Using the Bag of Words to find if the word exists in our vocabulary
def bag_of_words(sen, words, show_details= True ):
    
    sentence_words = senten_clean(sen)
    # bag of words - vocabulary matrix
    bow= [0]*len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word ==s:
                #We'll assign 1 if the current word is in the vocabulary matrix
                bow[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
                    
    return(np.array(bow))

# Predicting the class of the Input Sentence using the Deep Learning Model!
def predict_class(sentence):
    #Filter below threshold predictions
    bow = bag_of_words(sentence, words, show_details = False)
    predic = model.predict(np.array([bow]))[0]
    ERROR_THOLD = 0.25
    results = [[i, r] for i, r in enumerate(predic) if r >ERROR_THOLD]
    
    results.sort(key= lambda x:x[1], reverse = True)
    result_f = []
    
    for r in results:
        result_f.append({"intent": classes[r[0]], "probability": str(r[1])})
    return result_f

# After predicting the class, this will generate the appropriate response! 
def getResponse(inten, intents_json):
    tag = inten[0]['intent']
    intent_list = intents_json['intents']
    #print(intent_list)
    for i in intent_list:
        if (i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def send():
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)
    
    if msg != '':
        ChatBox.config(state = NORMAL)
        ChatBox.insert(END, "You: " + msg + "\n\n")
        ChatBox.config(foreground ="#456664", font= ("Lato", 12))
        
        ints = predict_class(msg)
        
        response = getResponse(ints, intents)
        #print(response)
        ChatBox.insert(END, "Bot: " + response + '\n\n')
        ChatBox.config(state= DISABLED)
        ChatBox.yview(END)
        
root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width = FALSE, height = FALSE)

#Chat Window Creation
ChatBox = Text (root, bd =0, bg = 'white', height = '9', width = '50', font = 'Lato',)
ChatBox.config(state=DISABLED)

#Bind scrollbar to chat window
scrollbar = Scrollbar(root, command = ChatBox.yview, cursor ="heart")
ChatBox["yscrollcommand"] = scrollbar.set

#Send Message Button Creation
SendB = Button(root, font = ("Verdana", 12, 'bold'), text = "Send", width = '12', height = 5, bd =0, bg = '#f8a602', activebackground = "#3c9d9b", fg = "#000000", command = send)

#Box for entering message
EntryBox = Text(root, bd =0, bg ='white', width = '29', height = '5', font = 'Arial')

#Placing all components on the screen
scrollbar.place(x =376, y =6, height = 386)
ChatBox.place(x=6, y=6, height=386, width = 370)
EntryBox.place(x = 128, y =401, height = 90, width =265)
SendB.place(x=6, y=401, height =90)

root.mainloop()
