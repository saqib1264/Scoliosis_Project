import tkinter as tk
# from tkinter import *
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re
import random
import os
import cv2


window = tk.Tk()
window.state('zoomed')   #Maximized View

w, h = window.winfo_screenwidth(), window.winfo_screenheight()
window.geometry("%dx%d+0+0" % (w, h))
window.title("REGISTRATION FORM")
window.configure(background="grey")
#window.resizable(False,False)

Fullname = tk.StringVar()
address = tk.StringVar()
username = tk.StringVar()
Email = tk.StringVar()
Phoneno = tk.IntVar()
var = tk.IntVar()
age = tk.IntVar()
password = tk.StringVar()
password1 = tk.StringVar()

value = random.randint(1, 1000)
print(value)

# database code
db = sqlite3.connect('evaluation.db')
cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS registration"
               "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT,Gender TEXT,age TEXT , password TEXT)")
db.commit()



def password_check(passwd): 
	
	SpecialSym =['$', '@', '#', '%'] 
	val = True
	
	if len(passwd) < 6: 
		print('length should be at least 6') 
		val = False
		
	if len(passwd) > 20: 
		print('length should be not be greater than 8') 
		val = False
		
	if not any(char.isdigit() for char in passwd): 
		print('Password should have at least one numeral') 
		val = False
		
	if not any(char.isupper() for char in passwd): 
		print('Password should have at least one uppercase letter') 
		val = False
		
	if not any(char.islower() for char in passwd): 
		print('Password should have at least one lowercase letter') 
		val = False
		
	if not any(char in SpecialSym for char in passwd): 
		print('Password should have at least one of the symbols $@#') 
		val = False
	if val: 
		return val 

def insert():
    fname = Fullname.get()
    addr = address.get()
    un = username.get()
    email = Email.get()
    mobile = Phoneno.get()
    gender = var.get()
    time = age.get()
    pwd = password.get()
    cnpwd = password1.get()

    with sqlite3.connect('evaluation.db') as db:
        c = db.cursor()

    # Find Existing username if any take proper action
    find_user = ('SELECT * FROM registration WHERE username = ?')
    c.execute(find_user, [(username.get())])

    # else:
    #   ms.showinfo('Success!', 'Account Created Successfully !')

    # to check mail
    #regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
    regex='^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if (re.search(regex, email)):
        a = True
    else:
        a = False
    # validation
    if (fname.isdigit() or (fname == "")):
        ms.showinfo("Message", "Please Enter Valid Name")
    elif (addr == ""):
        ms.showinfo("Message", "Please Enter Address")
    elif (email == "") or (a == False):
        ms.showinfo("Message", "Please Enter Valid Email")
    elif((len(str(mobile)))<10 or len(str((mobile)))>10):
        ms.showinfo("Message", "Please Enter 10 Digit Mobile Number")
    elif ((time > 100) or (time == 0)):
        ms.showinfo("Message", "Please Enter Valid Age")
    elif (c.fetchall()):
        ms.showerror('Error!', 'Username Taken Try a Diffrent One.')
    elif (pwd == ""):
        ms.showinfo("Message", "Please Enter Valid Password")
    elif (var == False):
        ms.showinfo("Message", "Please Enter Gender")
    elif(pwd=="")or(password_check(pwd))!=True:
        ms.showinfo("Message", "Password must contain atleast 1 Uppercase letter,1 symbol,1 number")
    elif (pwd != cnpwd):
        ms.showinfo("Message", "Password & Confirm Password must be same")
    else:
        conn = sqlite3.connect('evaluation.db')
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registration(Fullname, address, username, Email, Phoneno, Gender, age , password) VALUES(?,?,?,?,?,?,?,?)',
                (fname, addr, un, email, mobile, gender, time, pwd))

            conn.commit()
            db.close()
            ms.showinfo('Success!', 'Account Created Successfully !')
            window.destroy()
            from subprocess import call
            call(["python", "login.py"])
            # window.destroy()
            

#####################################################################################################################################################

#from subprocess import call
#call(["python", "lecture_login.py"])


# assign and define variable
# def login():

# #####For background Image
image2 = Image.open('12.png')
image2 = image2.resize((w, h))

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(window, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)


# image3 = Image.open('l9.png')
# image3 = image3.resize((80, 80))

# background_image1 = ImageTk.PhotoImage(image3)

# background_label1 = tk.Label(window, image=background_image1)

# background_label1.image = background_image1

# background_label1.place(x=350, y=10)

frame_alpr1 = tk.LabelFrame(window, text="Registration", width=600, height=630, bd=5, font=('times', 14, ' bold '),bg="snow")
frame_alpr1.grid(row=0, column=0)
frame_alpr1.place(x=400, y=80)

l1 = tk.Label(window, text="CREATE NEW ACCOUNT", font=("Times new roman", 15, "bold"), fg="black")
l1.place(x=600, y=40)

# that is for label1 registration

l2 = tk.Label(frame_alpr1, text="Full Name :", width=12, font=("Times new roman", 15, "italic"),fg="black")
l2.place(x=130, y=50)
t1 = tk.Entry(frame_alpr1, textvar=Fullname, width=20, font=('', 15))
t1.place(x=330, y=50)
# that is for label 2 (full name)


l3 = tk.Label(frame_alpr1, text="Address :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l3.place(x=130, y=100)
t2 = tk.Entry(frame_alpr1, textvar=address, width=20, font=('', 15))
t2.place(x=330, y=100)
# that is for label 3(address)


# that is for label 4(blood group)

l5 = tk.Label(frame_alpr1, text="E-mail :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l5.place(x=130, y=150)
t4 = tk.Entry(frame_alpr1, textvar=Email, width=20, font=('', 15))
t4.place(x=330, y=150)
# that is for email address

l6 = tk.Label(frame_alpr1, text="Phone number :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l6.place(x=130, y=200)
t5 = tk.Entry(frame_alpr1, textvar=Phoneno, width=20, font=('', 15))
t5.place(x=330, y=200)
# phone number
l7 = tk.Label(frame_alpr1, text="Gender :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l7.place(x=130, y=250)
# gender
tk.Radiobutton(frame_alpr1, text="Male", padx=5, width=5, fg="black", font=("italic", 15), variable=var, value=1).place(x=330,
                                                                                                                y=250)
tk.Radiobutton(frame_alpr1, text="Female", padx=20, width=4, fg="black", font=("italic", 15), variable=var, value=2).place(
    x=440, y=250)

l8 = tk.Label(frame_alpr1, text="Age :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l8.place(x=130, y=300)
t6 = tk.Entry(frame_alpr1, textvar=age, width=20, font=('', 15))
t6.place(x=330, y=300)

l4 = tk.Label(frame_alpr1, text="User Name :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l4.place(x=130, y=350)
t3 = tk.Entry(frame_alpr1, textvar=username, width=20, font=('', 15))
t3.place(x=330, y=350)

l9 = tk.Label(frame_alpr1, text="Password :", width=12, font=("Times new roman", 15, "italic"), fg="black")
l9.place(x=130, y=400)
t9 = tk.Entry(frame_alpr1, textvar=password, width=20, font=('', 15), show="*")
t9.place(x=330, y=400)

l10 = tk.Label(frame_alpr1, text="Confirm Password:", width=13, font=("Times new roman", 15, "italic"), fg="black")
l10.place(x=130, y=450)

t10 = tk.Entry(frame_alpr1, textvar=password1, width=20, font=('', 15), show="*")
t10.place(x=330, y=450)

btn = tk.Button(frame_alpr1, text="Register", bg="blue",font=("",20),fg="white", width=9, height=1, command=insert)
btn.place(x=260, y=500)
# tologin=tk.Button(window , text="Go To Login", bg ="dark green", fg = "white", width=15, height=2, command=login)
# tologin.place(x=330, y=600)
window.mainloop()