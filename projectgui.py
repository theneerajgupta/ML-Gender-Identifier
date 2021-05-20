import pickle
import numpy as np
import tkinter as tk
import tensorflow as tf
from elt import translit
from tensorflow.keras.models import save_model, load_model
from keras.models import model_from_json


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# load machine learning model here
with open("model.json", 'r') as file :
	json = file.read()

file = open('model.json', 'r')
json_file = file.read()
file.close()
model = model_from_json(json_file)
model.load_weights("weights.h5")
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# load word mapping
to_hindi = translit('hindi')
file = open("mapping.pkl", "rb")
mapping = pickle.load(file)
file.close()


# create function to preprocess data and predict
def guess_gender(name) :
    name = (20-len(name)) // 2 * '0' + name + (20-len(name)) // 2 * '0' + len(name) % 2 * '0'
    encoded = np.array([[mapping[char] for char in name]]).reshape((1, 20, 1))
    gender = model.predict_classes(encoded)[0]
    if gender == 0 :
        gender = "BOY"
    else :
        gender = "GIRL"

    return gender




# Top level window
frame = tk.Tk()
frame.title("NLP Gender Identification Using LSTM")
frame.geometry('300x120')



# Function for getting Input from textbox and printing it at label widget
def printInput():
	inp = inputtxt.get(1.0, "end-1c")
	hindi = to_hindi.convert([inp])[0]
	gender = guess_gender(hindi)
	print(f"{inp} / {hindi} is a {gender}")
	lbl1.config(text = "NAME IN ENGLISH: " + inp.upper())
	lbl2.config(text = "NAME IN HINDI: " + hindi)
	lbl3.config(text = "GENDER: " + gender)



# TextBox Creation
inputtxt = tk.Text(frame, height = 1, width = 20)
inputtxt.pack()


# Button Creation
printButton = tk.Button(frame, text = "PREDICT GENDER", command = printInput)
printButton.pack()


# Label Creation
lbl1 = tk.Label(frame, text = "NAME IN ENGLISH: ")
lbl1.pack()
lbl2 = tk.Label(frame, text = "NAME IN HINDI: ")
lbl2.pack()
lbl3 = tk.Label(frame, text = "GENDER: ")
lbl3.pack()
frame.mainloop()
