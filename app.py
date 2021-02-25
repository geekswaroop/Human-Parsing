import os
from flask import Flask, flash, request, redirect, url_for, render_template
from datetime import datetime
import subprocess

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the input image
            filename = "input.png"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Learned ML model to predict segmentation

            #subprocess.call('python3 test.py', shell=True)
            subprocess.call(['python3', 'train.py'])

            # Add dummy values to image path to avoid HTML Image caching

            suffix = "?" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            print("Suffix = ", suffix, type(suffix))
            input_filename = './static/input.png' + suffix
            output_filename = './static/output.png' + suffix
            return render_template('display.html', inputImage = input_filename, outputImage = output_filename)
    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
