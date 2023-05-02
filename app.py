from flask import Flask, render_template, request
from PIL import Image
import cv2
import os
import numpy as np
import pandas as pd
from csv import DictWriter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__, template_folder='templates')

@app.route('/')
def upload_file():
    return render_template('home.html')

@app.route('/text_compare')
def text_compare():
    return render_template('text.html')

@app.route('/image_compare')
def image_compare():
    return render_template('index.html')

@app.route('/text_comparing',methods=['POST','GET'])
def text_comparing():
    var = 0
    if request.method == 'POST':
        text_name = request.form.get("text")
        print(text_name)
        df = pd.read_csv('LogoNameDatabase.csv')
        for x in df['logoName'].values:
            print(x.lower)
            if x.lower() == text_name.lower():
                var = 1
                print("exists")
        if var == 0:
            dict1 = {}
            dict1["logoName"] = text_name
            with open('LogoNameDatabase.csv', 'a') as writer:
                objj = DictWriter(writer, ['logoName'])
                objj.writerow(dict1)
                writer.close()
    llt = []
    if var==1:
        llt.append("This name already Patent : " + text_name + " please choose another name!!!")
    else:
        llt.append("This name is unique : " + text_name + " now saved in our database.")

    return render_template('output_text.html',result=llt)




@app.route('/uploads',methods=['POST','GET'])
def uploads():
    if request.method == 'POST':
        img = request.files['file']
        image = Image.open(img)
        img1 = image.convert('L')
        #img1.save('gray_image.jpg')
        img1 = np.array(img1)
        lt = []
        a=0
        
        
        for i in range(2,11):
            path = "F:\ipr ibm final\static\logo ({a}).png".format(a=i)
            print(path)
            img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            # match_ratio = len(good_matches) / min(len(kp1), len(kp2))
            match_ratio = len(good_matches) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
            threshold = 0.3

            if match_ratio >= threshold:
                print('This logo Image is already Patent, Please choose another logo!!!')
                #lt.append("This logo is already Patent, please choose another logo!!!")
                a=1
                break
            else:
                print('The two images are different.')
    if a==1:
        lt.append("This logo Image is already Patent, Please choose another logo!!!")
    else:
        lt.append("Your logo Image is Unique.")

    return render_template('output.html',result=lt)

if __name__ == '__main__':
    app.run(debug = True, port=5051)