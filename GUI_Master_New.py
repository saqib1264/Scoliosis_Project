import tkinter as tk
from tkinter import ttk, LEFT, END, scrolledtext, messagebox
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
from keras.models import load_model
from keras.preprocessing import image
import CNNModel1  # Import your CNN model

global fn
fn = ""
basepath = "C:/Users/Tayyaba/Desktop/scoliosis detection"  # Adjust path

root = tk.Tk()
root.configure(background="#f0f0f0")  # Light gray background
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Scoliosis Detection System")

# Improved Title Label
title_font = ("Helvetica", 32, "bold")
title_label = tk.Label(root, text="Scoliosis Analysis System", font=title_font, bg="#4a6cd4", fg="white", pady=20)  # Darker blue
title_label.pack(fill=tk.X)

# Main Content Frame
main_frame = tk.Frame(root, bg="#e0e0e0", padx=20, pady=20)  # Lighter gray
main_frame.pack(expand=True, fill=tk.BOTH)

# Image Display Frames
image_frame = tk.Frame(main_frame, bg="#f0f0f0")
image_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)

original_img_label = tk.Label(image_frame, text="Original Image", relief=tk.RAISED, width=250, height=250, bg="white")
original_img_label.pack(pady=(0, 10))

gray_img_label = tk.Label(image_frame, text="Grayscale", relief=tk.RAISED, width=250, height=250, bg="white")
gray_img_label.pack(pady=(0, 10))

binary_img_label = tk.Label(image_frame, text="Binary", relief=tk.RAISED, width=250, height=250, bg="white")
binary_img_label.pack(pady=(0, 10))

spine_img_label = tk.Label(image_frame, text="Spine Highlight", relief=tk.RAISED, width=250, height=250, bg="white")
spine_img_label.pack(pady=(0, 10))

# Control and Report Frame
control_report_frame = tk.Frame(main_frame, bg="#e0e0e0", width=400)
control_report_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Control Buttons
button_frame = tk.Frame(control_report_frame, bg="#e0e0e0", pady=10)
button_frame.pack()

button_font = ("Arial", 14)
tk.Button(button_frame, text="Select Image", command=openimage, font=button_font, bg="#66bb6a", fg="white").pack(side=tk.LEFT, padx=5)  # Green
tk.Button(button_frame, text="Preprocess", command=convert_grey, font=button_font, bg="#ffb74d", fg="white").pack(side=tk.LEFT, padx=5)  # Orange
tk.Button(button_frame, text="Detect Spine", command=detect_spine, font=button_font, bg="#5c6bc0", fg="white").pack(side=tk.LEFT, padx=5)  # Indigo
tk.Button(button_frame, text="Analyze", command=test_model, font=button_font, bg="#29b6f6", fg="white").pack(side=tk.LEFT, padx=5)  # Light Blue
tk.Button(button_frame, text="Train Model", command=train_model, font=button_font, bg="#8d6e63", fg="white").pack(side=tk.LEFT, padx=5)  # Brown
tk.Button(button_frame, text="Exit", command=close_window, font=button_font, bg="#e57373", fg="white").pack(side=tk.LEFT, padx=5)  # Red


# Report Display
report_label = tk.Label(control_report_frame, text="Analysis Report", font=("Helvetica", 18, "bold"), bg="#e0e0e0")
report_label.pack(pady=(10, 0))

report_text = scrolledtext.ScrolledText(control_report_frame, wrap=tk.WORD, bg="white", height=10, font=("Courier New", 12))
report_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)


# --- Helper Functions ---

def update_report(str_T):
    report_text.insert(tk.END, str_T + "\n")
    report_text.see(tk.END)  # Autoscroll to the end

def openimage():
    global fn, original_img_label, gray_img_label, binary_img_label, spine_img_label
    fileName = askopenfilename(initialdir=basepath + '/testing', title='Select image for Analysis',
                               filetypes=[("all files", "*.*")])
    if fileName:
        fn = fileName
        img = Image.open(fn)
        img = img.resize((250, 250))
        imgtk = ImageTk.PhotoImage(img)
        original_img_label.config(image=imgtk)
        original_img_label.image = imgtk
        gray_img_label.config(image="")
        binary_img_label.config(image="")
        spine_img_label.config(image="")
        update_report("Image loaded: " + fileName)
    else:
        update_report("No image selected.")

def convert_grey():
    global fn, gray_img_label, binary_img_label
    if fn:
        try:
            img = cv2.imread(fn)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            gray_img = Image.fromarray(gray)
            gray_img = gray_img.resize((250, 250))
            gray_imgtk = ImageTk.PhotoImage(gray_img)
            gray_img_label.config(image=gray_imgtk)
            gray_img_label.image = gray_imgtk

            binary_img = Image.fromarray(binary)
            binary_img = binary_img.resize((250, 250))
            binary_imgtk = ImageTk.PhotoImage(binary_img)
            binary_img_label.config(image=binary_imgtk)
            binary_img_label.image = binary_imgtk
            update_report("Image Preprocessed.")
        except Exception as e:
            update_report(f"Error preprocessing image: {e}")
    else:
        update_report("Please select an image first.")

def test_model():
    global fn
    if fn:
        update_report("Analyzing image...")
        try:
            result = predict_image(fn)
            update_report("Analysis Result: " + result)
        except Exception as e:
            update_report(f"Error during analysis: {e}")
    else:
        update_report("Please select an image first.")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    model = load_model(basepath + '/scoliosis_model.h5')  # Changed model name
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    classes = {0: "Normal", 1: "Lumbar Scoliosis", 2: "Thoracic Scoliosis", 3: "Double Scoliosis"}  # Adjust classes to match your training

    return classes.get(class_index, "Unknown Scoliosis Type")

def detect_spine():
    global fn, spine_img_label
    if fn:
        try:
            img = cv2.imread(fn)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                spine_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                spine_img = spine_img.resize((250, 250))
                spine_imgtk = ImageTk.PhotoImage(spine_img)
                spine_img_label.config(image=spine_imgtk)
                spine_img_label.image = spine_imgtk
                update_report("Spine detected and highlighted.")
            else:
                update_report("Spine not detected.")
        except Exception as e:
            update_report(f"Error detecting spine: {e}")
    else:
        update_report("Please select an image first.")

def train_model():
    update_report("Model training started...")
    try:
        msg = CNNModel1.main()  # Call the training function
        update_report("Model training completed.\n" + msg)
        messagebox.showinfo("Training Complete", "Model training finished successfully!")
    except Exception as e:
        update_report(f"Error during model training: {e}")