import model_cnn as model
#import model_simple as model

from flask import Flask, request, jsonify, logging
from flask_cors import CORS

import numpy as np

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default

logging.getLogger('flask_cors').level = logging.DEBUG

# default route
@app.route('/')
def index():
    return "Index"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500# default route

# API route
@app.route('/api', methods=['POST'])
def api():
    input_data = request.json
    print(input_data)
    output_data = model.predict(input_data)
    print(np.argmax(output_data[0]))
    print(output_data[0][np.argmax(output_data)])
    response = jsonify(output_data.tolist())
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30500, debug=False)
