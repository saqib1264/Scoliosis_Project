import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
from tkvideo import tkvideo

root = tk.Tk()
root.state('zoomed')
root.title("Scoliosis Detection System")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()

video_label = tk.Label(root)
video_label.pack()

player = tkvideo("2.mp4", video_label, loop=1, size=(w, h))

label_l1 = tk.Label(root, text="Welcome To \n Scoliosis Detection System", font=("Times New Roman", 20, 'bold'), fg="black", width=20, height=5)
label_l1.place(x=600, y=50)
label_l1.configure(bg='SystemButtonFace')  # Set background to transparent

def play_video():
    player.play()

# Schedule video playback after 100 milliseconds
root.after(100, play_video)
#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#def clear_img():
#    img11 = tk.Label(root, background='bisque2')
#    img11.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def cap_video():
    
#     video1.upload()
#     #from subprocess import call
#     #call(['python','video_second.py'])

def reg():
    from subprocess import call
    call(["python","registrationpage.py"])

def log():
    from subprocess import call
    call(["python","login.py"])
  
def window():
  root.destroy()


button1 = tk.Button(root, text="Login", command=log, width=15, height=1,font=('times', 20, ' bold '), bg="gray", fg="white")
button1.place(x=640, y=250)

button2 = tk.Button(root, text="Registration",command=reg,width=15, height=1,font=('times', 20, ' bold '), bg="gray", fg="white")
button2.place(x=640, y=320)

button3 = tk.Button(root, text="Exit",command=window,width=15, height=1,font=('times', 20, ' bold '), bg="black", fg="white")
button3.place(x=640, y=420)

root.mainloop()
