import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory,redirect

app = Flask(__name__)
classes = ["Bathroom Mirrors", "Bathroom Shelves", "Bathroom Sinks", "Ceiling Fans", "Curtain Rods", "Pet Beds", "Rugs", "Toilet Paper Holders", "Towel Bars", "Towel Hooks", "Towel Racks", "Towel Rings"]
transfer_learning_model = None
# home page
@app.route("/")
def home():
  return render_template("test.html")


@app.route('/predict',methods=['POST'])
def predict():
    global transfer_learning_model
    if not transfer_learning_model :
        print("loading model..")
        transfer_learning_model = tf.keras.models.load_model("snaplookupv2.h5")
    else :
        print("model is already loaded")
    file = request.files['imgfile']
    preprocessed_image=tf.image.decode_jpeg(file.read(),channels=3)
    preprocessed_image=tf.image.resize(preprocessed_image,[100,100])
    preprocessed_image/=255.0
    preprocessed_image=tf.reshape(preprocessed_image,(1,100,100,3));
    res = transfer_learning_model.predict_classes(preprocessed_image, 1, verbose=0)[0]
    print("response:",res)
    print("response:",classes[res])
    return "https://www.lowes.com/search?searchTerm="+classes[res], 200

if(__name__=='__main__'):
  app.run(port=8080,debug='true',threaded=False)
