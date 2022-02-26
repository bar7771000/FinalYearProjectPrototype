import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk, ImageOps
from skimage import io, transform
from skimage.util import random_noise
from tkinter.filedialog import askopenfile
import tkinter.font as tkFont
from torchvision import models
from torchvision import transforms
import torch
import cv2
import math
#from perceptron.models.classification import PyTorchModel


background_color = "#b65449"
border_color = "#54221d"
btn_names = ["Browse", "Loading..."]
with open("coco.names", "r") as f:
    classes = f.read().splitlines()
    #print(classes)
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
noises = ["Salt & Pepper", "Gaussian", "Poison"]

def display_converted(img):
    blue, green, red = cv2.split(img)
    RGB_img = cv2.merge((red, green, blue))
    new_img = Image.fromarray(RGB_img)
    new_img.thumbnail((425, 425))
    display_img = ImageTk.PhotoImage(image=new_img)
    return display_img


def show_image():
    show_image.has_been_called = True
    browsing_text.set(btn_names[1])
    global BRG_img
    global filename
    filename = askopenfile(parent=root, mode="rb", title="Choose a file", filetype=[("JPG File", ".jpg"), ("JPEG File", ".jpeg"), ("PNG File", ".png")])
    if filename:
        BRG_img = cv2.imread(filename.name)
        display_img = display_converted(BRG_img)
        image_label.configure(image=display_img)
        image_label.image = display_img
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
    if show_image.has_been_called:
        noise_name = clicked.get()
        if noise_name == "Salt & Pepper":
            print("Salt & Pepper")
        elif noise_name == "Gaussian":
            print("Gaussian")
        elif noise_name == "Poison":
            print("Poison")


def update_image():
    if show_image.has_been_called:
        global noised_image
        update_image.has_been_called = True
        noise_image = random_noise(BRG_img, mode="s&p", amount=float(get_slider_value()))
        noised_image = np.array(255 * noise_image, dtype="uint8")

        filtered_image = display_converted(noised_image)
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
        classIds, scores, boxes = model.detect(BRG_img, confThreshold=0.4, nmsThreshold=0.4)
        for (classId, score) in zip(classIds, scores):
            prediction_outcome = [classes[classId], score]
            prediction_text = prediction_outcome[0] + " : " + str(round(prediction_outcome[1]*100, 2)) + "%"
            image_prediction_label1.config(text=prediction_text)
        if update_image.has_been_called:
            classIds_fil, scores_fil, boxes_fil = model.detect(noised_image, confThreshold=0.4, nmsThreshold=0.4)
            print(type(classIds_fil), len(scores_fil))
            for (classId, score) in zip(classIds_fil, scores_fil):
                print(classIds_fil, scores_fil)
                if len(scores_fil) != 0:  #TODO Figure this condition out
                    prediction_outcome = [classes[classId], score]
                    prediction_text = prediction_outcome[0] + " : " + str(round(prediction_outcome[1] * 100, 2)) + "%"
                    image_prediction_label2.config(text=prediction_text)
                    print("im still here")
                    print(score)
                else:
                    prediction_text = "Can't Recognise an Object"
                    image_prediction_label2.config(text=prediction_text)

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


"""
def login_screen():
    print("yello")
    #TODO: Implement Login/Register
"""

entry_screen()

