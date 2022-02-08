import tkinter as tk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from skimage import io, transform
from tkinter.filedialog import askopenfile
from torchvision import models
from torchvision import transforms
import torch
from perceptron.models.classification import PyTorchModel

def display_image():
    display_image.has_been_called = True
    browsing_text.set(btn_names[1])
    global filename
    filename = askopenfile(parent=root, mode="rb", title="Choose a file", filetype=[("JPG File", ".jpg"), ("JPEG File", ".jpeg"), ("PNG File", ".png")])
    if filename:
        img = Image.open(filename)
        print(type(img))
        img.thumbnail((425,425))
        img = ImageTk.PhotoImage(img)
        print(type(img))
        image_label.configure(image=img)
        image_label.image = img
    browsing_text.set(btn_names[0])
    return filename

def rgb_convolve(image, kern):
    new_image = np.empty_like(image)
    for dimension in range(image.shape[-1]):
        new_image[:, :, dimension] = sp.signal.convolve2d(image[:, :, dimension],
                                                          kern,
                                                          mode="same",
                                                          boundary="symm")
    return new_image


def callback(*args):
    if clicked.get() == "Random":
        KERNELS["Random"] = np.array([np.random.randint(-2, 4, 3), np.random.randint(-2, 4, 3), np.random.randint(-2, 4, 3)])
        print(KERNELS["Random"])
    if display_image.has_been_called:
        kernel_name = clicked.get()
        kernel = KERNELS[kernel_name]
        img_data = plt.imread(filename.name).astype(float) / 255.
        img_filtered = rgb_convolve(img_data, kernel)
        #print(type(img_filtered))
        #print(img_filtered)
        #new_image = Image.fromarray(img_filtered.astype('uint8'), 'RGB')
        new_image = Image.fromarray(np.uint8(img_filtered*255))
        #TODO: figure out the RGB conversion to get normal image
        new_image.thumbnail((425, 425))
        #print(type(new_image))
        #print(new_image)
        img = ImageTk.PhotoImage(new_image)
        image_label2.configure(image=img)
        image_label2.image = img
        img = transform(new_image)
        batch_t = torch.unsqueeze(img, 0)
        squeezenet.eval()
        #print(batch_t)
        out = squeezenet(batch_t)
        #print(out.shape)
        with open('classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        #print("Number of classes: {}".format(len(classes)))
        classes
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        print([(classes[idx], percentage[idx].item()) for idx in indices[0][:10]])
        #TODO: Rewrite prediction half as a separate function trigered on submitting


KERNELS = {
    "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Edge Detection 2": np.array([[2, -1, -2], [2, -2, -2], [-2, 2, 3]]),
    "Random": np.array([np.random.randint(-2, 4, 3), np.random.randint(-2, 4, 3), np.random.randint(-2, 4, 3)]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Intensified Sharpen": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
    "Gaussian Blur": np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]),
    "Grainy Sharpen": np.array([[-2, 3, -1], [-1, 0, 1], [-2, 2, 1]]),
    "Sharp Blur": np.array([[-2, 1, 0], [-1, -1, 3], [4, -2, -1]]),
    "Brighten Blur": np.array([[2, -1, -1], [-1, 3, 2], [-1, -1, 1]]),
    "Vertical Edge": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    "Horizontal Edge": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
    "Bottom Sobel": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Left Sobel": np.array([[1, 0, 1], [2, 0, -2], [1, 0, -1]]),
    "Right Sobel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Top Sobel": np.array([[1, 2, 1], [0, 0, 0], [-1, 2, -1]])
}

squeezenet = models.squeezenet1_1(pretrained=True) #sn1.1 has a bit better computation with a fewer parameters than 1.0 without looking accuracy (same as AlexNet)
print(squeezenet)

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

btn_names = ["Browse", "Loading..."]



root = tk.Tk()


display_image.has_been_called = False
canvas = tk.Canvas(root, width=1100, height=800, bg="#b65449")
canvas.grid(columnspan=2, rowspan=6)
image_label = tk.Label(root)
image_label.config(bg="#b65449")
image_label.grid(column=0, row=0)
image_label2 = tk.Label(root)
image_label2.config(bg="#b65449")
image_label2.grid(column=1, row=0)

browsing_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browsing_text, command=lambda: display_image(), height=2, width=10)
browsing_text.set(btn_names[0])
browse_btn.grid(columnspan=2, row=4)

clicked = tk.StringVar()
clicked.set("Pick a Filter")
clicked.trace("w", callback)
dropdown_menu = tk.OptionMenu(root,
                              clicked,
                              *[key for key in KERNELS.keys()]).grid(columnspan=2, row=3)

submit_button = tk.Button(root, text="Submit", height=2, width=10) #TODO Add command to this button
submit_button.grid(columnspan=2, row=5)
root.mainloop()