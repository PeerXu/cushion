#!/usr/bin/env python2.7
"""
@author: Peer Xu
@email: pppeerxu@gmail.com
@TODO: XXX
"""

import os
from flask import Flask
from flask import request
from flask import redirect
from flask import url_for
from flask import render_template 
from flask import jsonify
from flask import Response
import core
from cStringIO import StringIO
import requests

UPLOAD_FOLDER = core.ORIG_IMG_DB
ALLOWED_EXTENSIONS = set(['gif', 'png', 'jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['API_SERVICE'] = 'http://localhost:8000'

# API

@app.route('/v1/original_images', methods=['GET'])
def list_original_images():
    ois = [core.OrigImage.make_from_id(id).serialize() 
           for id in core.OrigImage.ids()]
    response = jsonify(origimages=ois)
    return response

@app.route('/v1/original_images/<orig_img_id>', methods=['GET'])
def show_original_image(orig_img_id):
    ois = [core.OrigImage.make_from_id(orig_img_id).serialize()]
    response = jsonify(origimages=ois)
    return response

@app.route('/v1/original_images/<orig_img_id>/value/<value>', methods=['POST'])
def post_original_image_for_value(orig_img_id, value):
    oi = core.OrigImage.make_from_id(orig_img_id)
    map(
        lambda splt_img_id, val: core \
            .get_perceived_character_cluster_instance(val) \
            .add_perceived_character(
                core.PerceivedCharacter.make_from_seed(splt_img_id)), 
        oi.split_list, value)
    return core.no_content_response()

@app.route('/v1/splited_images', methods=['GET'])
def list_splited_images():
    sis = [core.SplitedImage.make_from_id(id).serialize() 
           for id in core.SplitedImage.ids()]
    response = jsonify(splitedimages=sis)
    return response

@app.route('/v1/splited_images/<splt_img_id>', methods=['GET'])
def show_splited_images(splt_img_id):
    sis = [core.SplitedImage.make_from_id(splt_img_id).serialize()]
    response = jsonify(splitedimages=sis)
    return response

@app.route('/v1/splited_images/<splt_img_id>/character', methods=['GET'])
def show_splited_image_character(splt_img_id):
    response = jsonify(
        splited_image_id=splt_img_id, 
        character = core.SplitedImage.make_from_id(splt_img_id).get_character())
    return response

@app.route('/v1/original_images', methods=['POST'])
def post_original_image():
    file = request.files['original_image']
    if file and allowed_file(file.filename):
        filename = file.filename
        fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(fp)
        oi = core.OrigImage.make_from_src(fp)
        response = jsonify(**oi.serialize())
    else:
        response = jsonify(message="upload original image failed")
    return response

@app.route('/v1/perceived_characters/<ch>/<splt_img_id>', methods=['POST'])
def post_perceived_character(ch, splt_img_id):
    pcc = core.get_perceived_character_cluster_instance(ch)
    pcc.add_perceived_character(
        core.PerceivedCharacter.make_from_seed(splt_img_id))
    return jsonify(perceived_character=pcc.serialize())

@app.route('/v1/perceived_characters', methods=['GET'])
def list_perceived_character():
    chs = core.ALL_CHARACTERS
    return jsonify(perceived_characters=map(
        lambda x: core \
            .get_perceived_character_cluster_instance(x) \
            .serialize(),
        chs))

@app.route('/v1/perceived_characters/<ch>', methods=['GET'])
def show_perceived_character(ch):
    return jsonify(perceived_characters=[
            core.get_perceived_character_cluster_instance(ch) \
                .serialize()])

@app.route('/v1/actions/<action>', methods=['POST'])
def do_action(action):
    fn = 'action_' + action
    if hasattr(core, fn):
        getattr(core, fn)()
    return core.no_content_response()

# END API

@app.route('/original_images/<orig_img_id>', methods=['GET'])
def service_show_original_image(orig_img_id):
    oim = core.OrigImage.make_from_id(orig_img_id)
    return render_template('show_original_image1.html', oim=oim.serialize())

@app.route('/splited_images/<splt_img_id>', methods=['GET'])
def service_show_splited_image(splt_img_id):
    sim = core.SplitedImage.make_from_id(splt_img_id)
    return render_template('show_splited_image.html', sim=sim.serialize())

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/photos/<photo_id>')
def show_photos(photo_id):
    return '''
<!doctype html>
<html>
<body>
<img src="%s"/>
</body>
</html>
''' % url_for("static", filename=photo_id)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['oim']
        if file and allowed_file(file.filename):
            filename = file.filename
            stream = StringIO()
            file.stream.seek(0)
            stream.write(file.stream.read())
            stream.seek(0)
            files = {'original_image': (filename, stream)}
            resp = requests.post(
                app.config['API_SERVICE']+'/v1/original_images', files=files)
    return render_template('index.html', response=resp.json)

@app.route('/debug', methods=['POST'])
def debug():
    import pdb; pdb.set_trace();
    return jsonify()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', response={})

def _main(script, *args):
    app.run(debug=True)

if __name__ == '__main__':
    import sys
    _main(*sys.argv)
