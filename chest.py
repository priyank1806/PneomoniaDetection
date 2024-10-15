import warnings
from PIL import Image, ImageEnhance
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image

import tkinter as tk
from tkinter import filedialog, Label, Button

class PneumoniaDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("PNEUMONIA Detection App")
        master.geometry("700x600")
        
        self.label = Label(master, text="Chest X-ray PNEUMONIA Detection", font=('Helvetica', 18, 'bold'))
        self.label.pack(pady=20)

        # Upload Image Button
        self.upload_button = Button(master, text="Upload Image", command=self.upload_image, bg='#DF582C', fg='white', font=('Helvetica', 12, 'bold'))
        self.upload_button.pack(pady=20)
        
        # Predict Button
        self.predict_button = Button(master, text="Prediction", command=self.predict_result, bg='#DF582C', fg='white', font=('Helvetica', 12, 'bold'))
        self.predict_button.pack(pady=20)

        self.result_label = Label(master, text="", font=('Helvetica', 16))
        self.result_label.pack(pady=20)

        self.filepath = r"C:\Users\rumjhum\Desktop\Pneumonia Detection\archive\chest_xray\chest_xray\val\PNEUMONIA\person1946_bacteria_4875.jpeg"  # Initialize filepath

    def upload_image(self):
        # File dialog to upload an image
        self.filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.filepath:  # Check if the user selected a file
            print(f"Image path: {self.filepath}")
            self.model = load_model(r"C:\Users\rumjhum\Desktop\Pneumonia Detection\archive\chest_xray\chest_xray\val\PNEUMONIA\person1946_bacteria_4875.jpeg")  # Load the model

            img_file = image.load_img(self.filepath, target_size=(224, 224))
            x = image.img_to_array(img_file)
            x = np.expand_dims(x, axis=0)
            self.img_data = preprocess_input(x)
            self.classes = self.model.predict(self.img_data)
            self.result_label.config(text="Image Uploaded Successfully!", fg='blue')
        else:
            self.result_label.config(text="No Image Selected", fg='red')

    def predict_result(self):
        if self.classes is not None:  # Ensure that an image has been processed
            if self.classes[0][0] > 0.5:
                self.result_label.config(text="Result: Normal", fg='green')
            else:
                self.result_label.config(text="Result: Affected by PNEUMONIA", fg='red')
        else:
            self.result_label.config(text="Please upload an image first!", fg='red')

if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaDetectionApp(root)
    root.mainloop()
