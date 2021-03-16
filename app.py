import os
from flask import Flask, flash, request, redirect, url_for, render_template
from datetime import datetime

# Constants
UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
MODEL = 'densenet'

app = Flask(__name__)
app.secret_key = b'_5ef8t202w@$Trz\n\xec]/'
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

            # Input and Output filenames
            input_filename = './static/input.png'


            models = request.form.getlist('model')
            if models == []:
                flash('No ML model selected')
                return redirect(request.url)

            MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet']
            model_dict = {}
            for MODEL in MODELS:
                if MODEL in models:
                    model_dict[MODEL] = True
                    # Run inference.py
                    script_command = f'python3 inference.py -d {input_filename} -o ./static/output_{MODEL}.png -be {MODEL}'
                    print(MODEL)
                    os.system(script_command)
                else:
                    model_dict[MODEL] = False

            # Add dummy values to image path to avoid HTML Image caching
            suffix = "?" + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

            return render_template('display.html', suffix = suffix, model_dict = model_dict)
    return render_template('home.html')
    

if __name__ == "__main__":
    app.run(debug=True)
