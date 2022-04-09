import tkinter as tk
from tkinter import ttk
import numpy as np



import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from skimage.util import random_noise
from tkinter.filedialog import askopenfile
import tkinter.font as tkFont

import cv2
import math



# TODO


background_color = "#b65449"
border_color = "#54221d"
btn_names = ["Browse", "Loading..."]
with open("coco.names", "r") as f:
    classes = f.read().splitlines()
    #print(classes)
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
noises = ["Salt & Pepper", "Salt", "Pepper", "Gaussian", "Poison", "Speckle"]

def convert_img(img):
     RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     new_img = Image.fromarray(RGB_img)
     new_img.thumbnail((425, 425))
     converted_img = ImageTk.PhotoImage(image=new_img)
     return converted_img


def show_image():
    show_image.has_been_called = True
    browsing_text.set(btn_names[1])
    global raw_image
    global filename
    filename = askopenfile(parent=root, mode="rb", title="Choose a file", filetype=[("JPG File", ".jpg"), ("JPEG File", ".jpeg"), ("PNG File", ".png")])
    if filename:
        raw_image = cv2.imread(filename.name)
        display_image = convert_img(raw_image)
        image_label.configure(image=display_image)
        image_label.image = display_image
        if image_label.image:
            slider.grid(columnspan=2, row=1)
        browsing_text.set(btn_names[0])
        """
        
        blue, green, red = cv2.split(BRG_img)
        RGB_img = cv2.merge((red, green, blue))
        base_img = Image.fromarray(RGB_img)
        base_img.thumbnail((425, 425))
        display_img = ImageTk.PhotoImage(image=base_img)
        
                    cv2.namedWindow("Salt and Pepper")
        cv2.moveWindow("Salt and Pepper", 100,100)
            cv2.imshow("Salt and Pepper", resized)
        
        #base_img = ImageOps.expand(Image.open(filename), border=5, fill=border_color)
        base_img = Image.open((filename))
        base_img.thumbnail((425,425))
        display_img = ImageTk.PhotoImage(base_img)
        image_label.configure(image=display_img)
        image_label.image = display_img
        """


def callback(*args):
    global noise_image
    if show_image.has_been_called:
        noise_name = clicked.get()
        print(noise_name)
        print(get_slider_value())
        # change for a switch statement// Switch statement only exists in python 3.10 :(
        #Noises from skimage:
        #Gaussian, Poisson, Salt, Pepper, S&P, Speckle

        if noise_name == "Salt & Pepper":
            noise_image = random_noise(raw_image, mode="s&p", amount=float(get_slider_value()))

        elif noise_name == "Salt":
            noise_image = random_noise(raw_image, mode="salt", amount=float(get_slider_value()))

        elif noise_name == "Pepper":
            noise_image = random_noise(raw_image, mode="pepper", amount=float(get_slider_value()))

        elif noise_name == "Gaussian":
            noise_image = random_noise(raw_image, mode="gaussian", var=float(get_slider_value())) #Updates upon change and stays at the same value until changed again

        elif noise_name == "Poisson":
            noise_image = random_noise(raw_image, mode="poisson") #Poisson works but doesnt do anything

        elif noise_name == "Speckle":
            noise_image = random_noise(raw_image, mode="speckle", var=float(get_slider_value())) #Updates upon change and stays at the same value until changed again

        return noise_image


#Need to create 2 seaparate update functions
# 1) for Salt, Pepper, S&P with amount parameter
# 2) for Gaussian and Speckle with var(variance) parameter instead


def update_image(): #TODO add noise parameter
    if show_image.has_been_called:
        callback()
        global noised_image
        #if noise_image != None:
        update_image.has_been_called = True
        #noise_image = random_noise(raw_image, mode="s&p", amount=float(get_slider_value())) #TODO Change parameter to value from select option
        noised_image = np.array(255 * noise_image, dtype="uint8")
        filtered_image = convert_img(noised_image)
        image_label2.configure(image=filtered_image)
        image_label2.image = filtered_image
    else:
        print("You have to upload an image first!")


def get_slider_value():
    return '{: .2f}'.format(current_value.get())

def slider_changed(event):
    slide_val = math.floor(float(get_slider_value())*100)
    slider_label.configure(text=f" {slide_val}% Noise")
    return current_value.set(event)

def perform_detection():
    if show_image.has_been_called:
        classIds, scores, boxes = model.detect(raw_image, confThreshold=0.4, nmsThreshold=0.4) #base image prediction
        for (classId, score) in zip(classIds, scores): #TODO: Possibly get rid of for loop as were looking only for person class and its calling function twice xd
            if classId == 0: #Recognises only human now!
                prediction_outcome = [classes[classId], score]
                prediction_text = prediction_outcome[0] + " : " + str(round(prediction_outcome[1]*100, 2)) + "%"
                image_prediction_label1.config(text=prediction_text)
            if update_image.has_been_called:
                classIds_fil, scores_fil, boxes_fil = model.detect(noised_image, confThreshold=0.4, nmsThreshold=0.4) #filtered image prediction
                print(type(classIds_fil), len(scores_fil))  # when class is not recognised classIds_fill = "tuple" / len of scores_fill = 0
                #for (classId_fil, score_fil) in zip(classIds_fil, scores_fil):
                for i in range(len(classIds_fil)):
                    if classIds_fil[i] == 0:
                        print(classIds_fil, scores_fil)
                        prediction_outcome = [classes[classIds_fil[i]], scores_fil[i]]
                        prediction_text = prediction_outcome[0] + " : " + str(round(prediction_outcome[1] * 100, 2)) + "%"
                        image_prediction_label2.config(text=prediction_text)
                        # TODO: check the bookmarks on how to display plot in tkinter window
                        #slices = [3, 7, 8]
                        #plt.pie(slices, labels=["Gaussian", "S&P", "Poisson"], colors=["r", "y", "g"])
                        #plt.show()
                if len(scores_fil) == 0:
                    prediction_text = "Can't Recognise an Object"
                    image_prediction_label2.config(text=prediction_text)
                    print(prediction_text)

    else:
        print("You have to upload an image first!")
    """
    
        
         for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(BRG_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)

        text = '%s: %.2f' % (classes[classId], score)
        cv2.putText(BRG_img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=2)
    print(type(BRG_img))
    cv2.namedWindow("Image")
    cv2.moveWindow('Image', 100, 200)
    cv2.imshow('Image', BRG_img)
    """

def main_screen():
    fontStyle = tkFont.Font(family="Lucida Grande", size=18)
    start_screen.destroy()
    global clicked, browsing_text, root, image_label, image_label2, clicked, image_prediction_label1, image_prediction_label2, slider, slider_label, current_value
    root = tk.Tk()
    root.title("People Segmentation Application")
    show_image.has_been_called = False
    update_image.has_been_called = False
    canvas = tk.Canvas(root, width=1100, height=900, bg=background_color)
    canvas.grid(columnspan=2, rowspan=6)
    image_label = tk.Label(root)
    image_label.config(bg=background_color)
    image_label.grid(column=0, row=0)
    image_label2 = tk.Label(root)
    image_label2.config(bg=background_color)
    image_label2.grid(column=1, row=0)
    image_prediction_label1 = tk.Label(root, text=" ", font=fontStyle, bg=background_color)
    image_prediction_label1.grid(column=0, row=1)
    image_prediction_label2 = tk.Label(root, text=" ", font=fontStyle, bg=background_color)
    image_prediction_label2.grid(column=1, row=1)
    browsing_text = tk.StringVar()
    browse_btn = tk.Button(root, textvariable=browsing_text, command=show_image, height=2, width=10)
    browsing_text.set(btn_names[0])
    browse_btn.grid(columnspan=2, row=4)
    update_btn = tk.Button(root, text="Update", command=update_image, height=2, width=10)
    update_btn.grid(columnspan=2, row=3)
    slider_label = tk.Label(root, bg=background_color)
    slider_label.grid(columnspan=2, row=2)
    current_value = tk.DoubleVar()
    slider = ttk.Scale(root, from_=0, to=1, orient='horizontal', command=slider_changed, variable=current_value)
    clicked = tk.StringVar()
    clicked.set("Pick a Noise")
    clicked.trace("w", callback)
    dropdown_menu = tk.OptionMenu(root,
                                 clicked,
                                 *[key for key in noises]).grid(columnspan=2, row=6)

    submit_button = tk.Button(root, text="Submit", command=perform_detection, height=2, width=10)
    submit_button.grid(columnspan=2, row=5)
    root.mainloop()


def entry_screen():
    global start_screen
    screen_height = 300
    screen_width = 400
    start_screen = tk.Tk()
    start_screen.config(bg=background_color)
    start_screen.config(height=screen_height, width=screen_width)
    #start_screen.geometry("300x400")
    tk.Button(text ="Start", height="2", width="20", command=main_screen).place(x=screen_width/3, y=screen_height/3) #TODO: redirect to login screen
    start_screen.mainloop()




entry_screen()

