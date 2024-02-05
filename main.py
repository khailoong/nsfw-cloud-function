import os
from flask import Flask, jsonify, request
from transformers import pipeline
import time

app = Flask(__name__)

model_path = "model"

@app.route('/nsfw')
def classify_review():
    start_time = time.time()

    img = request.args.get('img')

    if img is None:
        return jsonify(code=403, message="Image URL is required")
    
    pipe = pipeline("image-classification", model=model_path, tokenizer=model_path)
    result = pipe(img.strip('\"'))

    end_time = time.time()
    execution_time = end_time - start_time

    return jsonify(result=result, execution_time=execution_time)


#for localhost only
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
