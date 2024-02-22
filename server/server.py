from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory,Response, send_file
from werkzeug.utils import secure_filename
import os
from livereload import Server
import time



static = os.path.join('server','static')
img = os.path.join('static','images')
pythons = os.path.join('static','pythons')
txt = os.path.join('static','logs')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = static
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')


def index():
    return render_template('index.html')
    
@app.route('/images')
def images():
    full_filename = []    
    for file in os.listdir(img):
        if "png" or "jpg" in file:
            full_filename.append(os.path.join(img, file))
    return render_template("images.html",images = full_filename)

@app.route('/dynamic_text')
def show_text():
    file = []
    for t in os.listdir(txt):
        file.append(t)
    #print(file)
    return send_from_directory(txt, "logs.txt")
@app.route('/pythons')
def show_pythos():
    file = []
    for t in os.listdir(pythons):
        file.append(os.path.join(pythons, t))
    #
    return render_template("pythons.html", files = file)
 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',use_reloader=True)
    #
    
    
    
    #<img src="{{ image }} alt ="image" />