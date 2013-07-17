# coding=utf8
from PIL import Image
import itertools
import os
import commands
import redis
import md5
from functools import partial
from  cStringIO import StringIO
import json
import tempfile
import base64
import uuid
import string
import flask

import em

import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

RGBA = {
    'black': (0, 0, 0, 255),
    'white': (255, 255, 255, 255)}

P = {
    'black': 0,
    'white': 225
}

STEPS = (3, 2)

RC = lambda:redis.Redis()
ORIG_IMG_DB = 'static/db/'
SPLT_IMG_DB = 'static/cache/'
ID_GEN = lambda: str(uuid.uuid4()).replace('-', '')
PERCEIVED_CHARACTER_CLUSTER = {}
ALL_CHARACTERS = string.lowercase + ''.join(map(str, range(0, 10))) + '_'

MAX_DISTANCE = 10000

def load_value(f):
    with open(f) as fr:
        return eval(fr.read())

class MyImage(object):
    S2P_LOG = 's2p.txt'
    def __init__(self, src):
        self.src = src
        self.img = Image.open(src)
        self.pixels = self.img.load()
        self.size = self.img.size
        self.pts = _convert2dat(self.pixels, self.img.size)
        self.ism = ImageSignMaker(self)

    @property
    def S2P(self):
        if not hasattr(MyImage, '_S2P'):
            MyImage._S2P = load_value(MyImage.S2P_LOG)
        return MyImage._S2P

    @property
    def sign(self):
        if not hasattr(self, '__sign'):
            self.__sign = self.ism.sign()
        return self.__sign

    def distance(self, img):
        def _distance(v, w):
            return sum([(a-b)**2 for a, b in zip(v, w) if a or b]) ** 0.5
        return _distance(self.sign, img.sign)

    @property
    def orig_path(self):
        return self.S2P[str(self.sign)]


class IsVisited(object):
    visited = False
class IsNoise(object):
    is_nose = False

class Point(IsNoise, IsVisited):
    def __init__(self, x, y, visited=False, is_noise=False):
        self.x = x
        self.y = y

    def d(self, p):
        return ((p.x - self.x) ** 2 + (p.y - self.y) ** 2) ** 0.5

    def show(self):
        return self.x, self.y

    def __repr__(self):
        return '<Point x:%s y:%s>' % (self.x, self.y)

    __str__ = __repr__

class PointGraph(object):
    def __init__(self, points):
        self.points = points
        self._ordered_xpts = sorted([(p.x, p) for p in self.points])
        self._ordered_ypts = sorted([(p.y, p) for p in self.points])

    def vertices(self):
        return self.points

    def region_query(self, p, r):
        def limt_pts(limts, pts):
            s = set()
            for v, p in pts:
                if v < limts[0]:
                    continue
                if limts[0] <= v <= limts[1]:
                    s.add(p)
                elif limts[1] < v:
                    break
            return s
        
        xlimt = (p.x - r, p.x + r)
        ylimt = (p.y - r, p.y + r)
        xs = limt_pts(xlimt, self._ordered_xpts)
        ys = limt_pts(ylimt, self._ordered_ypts)
        standby_pts = xs & ys

        neighbor_pts = [p]
        for sp in standby_pts:
            if p.d(sp) <= r:
                neighbor_pts.append(sp)

        return neighbor_pts

class ImagePoint(IsVisited, IsNoise, MyImage):
    def __init__(self, src):
        MyImage.__init__(self, src)
        self.memoized = {}

    @classmethod
    def instance_maker(cls, fs):
        return [cls(f) for f in fs if os.path.exists(f)]

    @classmethod
    def instance(cls, f):
        assert os.path.exists(f)
        return cls(f)

class ImageGraph(object):
    def __init__(self, ips):
        self.ips = ips

    def vertices(self):
        return self.ips

    def region_query(self, ip, r):
        return [p for p in self.ips if ip.distance(p) <= r]

class VertexGraph(dict):
    def __init__(self, vs=None, es=None):
        if vs is None:
            vs = []
        if es is None:
            es = []

        for v in vs:
            self.add_vertex(v)
        for e in es:
            self.add_edge(e)

    def add_vertex(self, v):
        self[v] = {}
        self.get("*index*", {})[v.label] = v

    def add_edge(self, e):
        v, w = e[:2]
        self[v][w] = e
        self[w][v] = e

    def get_vertex(self, lbl):
        return self.get("*index*", {}).get(lbl, None)

    def get_edge(self, v, w):
        try:
            return self[v][w]
        except KeyError:
            return None

    def remove_edge(self, e):
        v, w = e[:2]
        del self[v][w]
        del self[w][v]

    def vertices(self):
        return self.keys()

    def edges(self):
        es = set()
        for v, ws in self.iteritems():
            for w in ws:
                es.add(self[v][w])
        return list(es)

    def out_vertices(self, v):
        try:
            self[v].keys()
        except KeyError:
            return []

    def out_edges(self, v):
        try:
            self[v].values()
        except KeyError:
            return []

    def region_query(self, v, r):
        return sorted([(e[2], e[1]) for e in self.out_edges(v) if e[2] <= r])


class Vertex(IsVisited, IsNoise):
    def __init__(self, label=''):
        self.label = label

    def __repr__(self):
        return 'Vertex(%s)' % repr(self.label)

    __str__ = __repr__

class SignVertex(Vertex):
    def __init__(self, label, signed_vector):
        Vertex.__init__(self, label)
        self.svs = signed_vector

class Edge(tuple):
    def __new__(self, *vs):
        assert len(vs) == 3
        return tuple.__new__(self, vs)

    def __repr__(self):
        return 'Edge(%s, %s, %s)' % (repr(self[0]), repr(self[1]), repr(self[2]))

    __str__ = __repr__

class DBSCAN(object):
    def __init__(self, graph, eps, min_pts):
        self.clusters = {}
        self.graph = graph
        self.eps = eps
        self.min_pts = min_pts
        self._cluster_id = -1

    def _next_cluster(self):
        self._cluster_id += 1
        return self._cluster_id

    def DBSCAN(self):
        n = 0
        for v in self.graph.vertices():
            if v.visited:
                continue
            n += 1
            v.visited = True
            neighbor_pts = self.region_query(v, self.eps)
            if len(neighbor_pts) < self.min_pts:
                v.is_noise = True
            else:
                cluster_id = self._next_cluster()
                self.expand_cluster(
                    v, neighbor_pts, cluster_id, self.eps, self.min_pts)
                LOG.debug("times %s: cid %s: %s: neighbors size = %s" % (n, cluster_id, v, len(neighbor_pts)))


    def expand_cluster(self, p, neighbor_pts, cluster_id, eps, min_pts):
        cluster = [p]

        for pp in neighbor_pts:
            if not pp.visited:
                pp.visited = True
                new_neighbor_pts = self.region_query(pp, self.eps)
                if len(new_neighbor_pts) >= self.min_pts:
                    neighbor_pts += new_neighbor_pts
                if not self.is_member_of_cluster(pp):
                    cluster.append(pp)
        
        self.clusters[cluster_id] = cluster
                

    def is_member_of_cluster(self, p):
        for k in self.clusters.keys():
            if p in self.clusters[k]:
                return True
        return False

    def region_query(self, p, eps):
        return self.graph.region_query(p, eps)

def convert(src, dst):
    img = Image.open(src).convert('RGBA')
    pixdata = img.load()

    for y in xrange(img.size[1]):
        for x in xrange(img.size[0]):
            if pixdata[x,y][0] < 90:
                pixdata[x,y] = (0, 0, 0, 255)
            else:
                pixdata[x,y] = (255,255,255, 255)
            if pixdata[x,y][1] < 136:
                pixdata[x,y] = (0, 0, 0, 255)
            else:
                pixdata[x,y] = (255,255,255, 255)
            if pixdata[x,y][2] > 0:
                pixdata[x,y] = (255,255,255,255)
            else:
                pixdata[x,y] = (0,0,0,0)
    img.save(dst, "GIF")

def save_image(file, pts, size):
    img = Image.new("RGBA", size)
    ps = img.load()
    for p in _mk_ps(size):
        if p in pts:
            ps[p] = RGBA['black']
        else:
            ps[p] = RGBA['white']
    img.save(file, "GIF")

def _mk_ps(size):
    return itertools.product(xrange(size[0]),
                             xrange(size[1]))

def _copy_pixels(ps, size):
    pixels = {}
    for p in _mk_ps(size):
        pixels[p] = ps[p]
    return pixels

def __denoise(ps, size):
    pixels = _copy_pixels(ps, size)
    for p in _mk_ps(size):
        if pixels[p][0] < 90 or \
                pixels[p][1] < 136 or \
                pixels[p][2] >= 0:
            pixels[p] = P['black']
        else:
            pixels[p] = P['white']
    return pixels

def _denoise(ps, size):
    pixels = _copy_pixels(ps, size)
    for p in _mk_ps(size):
        if pixels[p] == (0, 0, 153, 255):
            pixels[p] = P['black']
        else:
            pixels[p] = P['white']
    return pixels

def _convert2dat(seq, size):
    pts = []
    for p in _mk_ps(size):
        if seq[p] == 0:
            pts.append(p)
    return pts

def _mk_img(pixels, size):
    img = Image.new('P', size)
    pxls = img.load()
    for p in _mk_ps(size):
        pxls[p] = pixels[p]
    return img

img_2_pnt = _convert2dat

def convert2dat(src, dst):
    img = Image.open(src)
    pixdata = img.load()
    pts = _convert2dat(pixdata, img.size)
    with open(dst, 'w') as fw:
        for p in pts:
            fw.write('%s %s\n' % (p[0], p[1]))

class Character(object):
    def __init__(self, pts, orig):
        self.pts = pts
        self.size = (
            max([p[0] for p in pts]),
            max([p[1] for p in pts]))
        self.orig = orig

    @classmethod
    def make_from_unoriginal_pts(cls, pts):
        orig_pts = move_to_orig(pts)
        orig = find_origin(pts)
        inst = cls(orig_pts, orig)
        return inst
        
def find_origin(ps):
    return (min([p[0] for p in ps]), min([p[1] for p in ps]))

def move_to_orig(ps):
    mx, my = find_origin(ps)
    return [(p[0] - mx + 1, p[1] - my + 1) for p in ps]

def split_graphic_to_character(g):
    dbscan = DBSCAN(g, 2, 4)
    dbscan.DBSCAN()

    pts = []
    for ps in dbscan.clusters.values():
        pts.append([p.show() for p in ps])

    chs = []
    for ps in pts:
        chs.append(Character.make_from_unoriginal_pts(ps))

    return chs

def basename(filename):
    return os.path.basename(filename).rsplit('.')[0]
    
def is_split_success(ch):
    return ch.size[0] <= 24 and ch.size[1] <= 24  and 25 <= len(ch.pts) <= 120

def _sign(seq, size, step):
    signed = []
    grid_pts = [x for x in itertools.product(range(step[0]), range(step[1]))]
    orig_pts = [x for x in itertools.product(
            range(0, size[0], step[0]), range(0, size[1], step[1]))]
    for op in orig_pts:
        c = 0
        for gp in grid_pts:
            x, y = gp[0] + op[0], gp[1] + op[1]
            if seq[x,y] == 0:
                c += 1
        signed.append(c)
    return signed

def sign(src, dst):
    img = Image.open(src)
    pixels = img.load()
    signed = _sign(pixels, img.size, STEPS)
    with open(dst, 'a') as fw:
        fw.write('%s %s\n' % (src,
                              ' '.join(map(str, signed))))

def uniq_sign(signfile):
    uniq_signed = {}
    with open(signfile) as fr:
        for line in fr:
            f, vs = line.split(' ', 1)
            if vs not in uniq_signed:
                uniq_signed[vs] = f
    with open(signfile, 'w') as fw:
        for vs, f in uniq_signed.iteritems():
            fw.write('%s %s' % (f, vs))
    
def delete_repeat(signfile):
    A = set(['target/' + s for s in commands.getoutput('ls target/').split('\n')])
    B = set([s for s in commands.getoutput("awk '{print $1}' signed.txt").split('\n')])
    map(os.unlink, A-B)

def chs_cluster(fs):
    imgs = ImagePoint.instance_maker(fs)
    g = ImageGraph(imgs)
    dbscan = DBSCAN(g, 3, 2)
    dbscan.DBSCAN()

def clean_image(src):
    img = Image.open(src)
    img.save(src)

def copy_pixels(pixels, size):
    cp = {}
    for p in _mk_ps(size):
        cp[p] = pixels[p]
    return cp

def file_md5(f):
    with open(f) as fr:
        return md5.md5(fr.read()).hexdigest()

class SignMaker(object):
    def __init__(self, steps):
        self.steps = steps

    def sign(self, seq, size):
        signed = []
        grid_pts = \
            [x for x in itertools.product(range(self.steps[0]), 
                                          range(self.steps[1]))]
        orig_pts = \
            [x for x in itertools.product(range(0, size[0], self.steps[0]), 
                                          range(0, size[1], self.steps[1]))]
        for op in orig_pts:
            c = 0
            for gp in grid_pts:
                x, y = gp[0] + op[0], gp[1] + op[1]
                if seq[x,y] == 0:
                    c += 1
            signed.append(c)
        return signed

class ImageSignMaker(SignMaker):
    def __init__(self, img):
        steps = ImageSignMaker.calc_out_steps(img)
        SignMaker.__init__(self, steps)
        self.img = img

    def sign(self):
        return SignMaker.sign(self, self.img.pixels, self.img.size)

    @classmethod
    def calc_out_steps(cls, _img):
        return STEPS

class Dumper(object):
    def __init__(self): pass

    def _contains(self, id):
        return RC().keys(id) != []

    def _dump(self, id, its):
        rc = RC()
        f = partial(rc.hset, id)
        map(lambda x: apply(f, x), its)

    def dump(self, id, its):
        rc = RC()
        if self._contains(id):
            return
        self._dump(id, its)

    def force_dump(self, id, its):
        self._dump(id, its)

class Keyer(object):
    def __init__(self):
        pass

    @classmethod
    def key(cls, x=None):
        if x:
            return ':'.join([cls.PREFIX, x])
        return cls.PREFIX

class OrigImage(Dumper, Keyer):
    """
    id: Original Image的id值, 为该图片的md5值
    value: 图片的值
    splited: 可分片值
    split-list: 分片后的SplitedImage Ids
    """

    PREFIX = 'oim'

    def _init_from_img(self, src):
        clean_image(src)
        img = Image.open(src).convert('RGBA')
        self.id = file_md5(src)
        self.path = os.path.join(ORIG_IMG_DB, '-'.join([self.PREFIX, self.id]))
        self.splited = True
        self.split_list = []
        self.ops = [] # origin point list
        self.value = ''
        pixels = _denoise(img.load(), img.size)
        pts = [Point(*p) for p in _convert2dat(pixels, img.size)]
        g = PointGraph(pts)
        chs = split_graphic_to_character(g)
        for ch in chs:
            if is_split_success(ch):
                f = tempfile.mktemp()
                save_image(f, ch.pts, SplitedImage.SIZE)
                si = SplitedImage.make_from_src(f)
                os.unlink(f)
                self.split_list.append(si.id)
                self.ops.append(ch.orig)
            else:
                self.splited = False
        if self.splited == True:
            self.value = '*' * len(chs)
        os.rename(src, self.path)

    def serialize(self):
        return {
            'id': self.id,
            'value': self.value,
            'splited': self.splited,
            'split_list': self.split_list,
            'path': self.path,
            'ops': self.ops}

    def dump(self):
        Dumper.dump(self, self.key(self.id),
                    [('id', self.id),
                     ('path', self.path),
                     ('value', self.value),
                     ('splited', self.splited),
                     ('split_list', self.split_list),
                     ('ops', self.ops)])

    @classmethod
    def ids(cls):
        rc = RC()
        return [id.split(':')[1] for id in rc.keys(cls.key('*'))]

    @classmethod
    def load(cls, id):
        inst = cls()
        rc = RC()
        kv = rc.hgetall(cls.key(id))
        for k, v in kv.items():
            setattr(inst, k, v if k not in ('split_list', 
                                            'ops') else eval(v))
        return inst

    @classmethod
    def make_from_src(cls, src):
        inst = cls()
        inst._init_from_img(src)
        inst.dump()
        return inst

    @classmethod
    def make_from_id(cls, id):
        return cls.load(id)

    def __init__(self):
        Dumper.__init__(self)
        Keyer.__init__(self)

class SplitedImage(Dumper, Keyer):
    """
    id: Splited Image id
    image: bit image
    """
    SIZE = (24, 24)
    PREFIX = 'sim'
    def __init__(self):
        Dumper.__init__(self)
        Keyer.__init__(self)

    def _init_from_img(self, src):
        sio = StringIO()
        img = Image.open(src)
        img.save(sio, format=img.format)
        sio.seek(0)
        self.id = file_md5(src)
        self.bits = sio.read()
        self.pixels = _copy_pixels(img.load(), img.size)
        self.size = img.size
        self.pts = _convert2dat(self.pixels, self.size)
        self.bps = len(self.pts)
        self.ism = ImageSignMaker(self)
        self.path = os.path.join(SPLT_IMG_DB, 'sim-' + self.id)

    def get_character(self):
        pccs = all_perceived_character_clusters()
        closest_pcc = min([(pcc.distance(self), pcc.ch) for pcc in pccs])
        return closest_pcc[1] if closest_pcc[0] < em.NEIGHBOUR_THRESHOLD else '*'

    def ensure_cached(self):
        if os.path.exists(self.path):
            return
        with open(self.path, 'w') as fw:
            fw.write(self.bits)

    def serialize(self):
        self.ensure_cached()
        return {
            'id': self.id,
            'bits': base64.encodestring(self.bits),
            'pixels': {str(k): v for k, v in self.pixels.items()},
            'size': self.size,
            'pts': self.pts,
            'bps': self.bps,
            'sign': self.sign,
            'path': self.path}

    @classmethod
    def ids(cls):
        return [id.split(':')[1] for id in RC().keys(cls.key('*'))]

    @property
    def sign(self):
        if not hasattr(self, '__sign'):
            self.__sign = self.ism.sign()
        return self.__sign

    def distance(self, img):
        def _distance(v, w):
            return sum([(a-b)**2 for a, b in zip(v, w) if a or b]) ** 0.5
        return _distance(self.sign, img.sign)

    @classmethod
    def make_from_src(cls, src):
        inst = cls()
        inst._init_from_img(src)
        inst.dump()
        return inst

    @classmethod
    def make_from_id(cls, id):
        return cls.load(id)

    def dump(self):
        Dumper.dump(self, self.key(self.id),
                    [('id', self.id),
                     ('bits', str(self.bits)),
                     ('pixels', str(self.pixels)),
                     ('size', str(self.size)),
                     ('pts', str(self.pts)),
                     ('bps', self.bps),
                     ('sign', str(self.sign)),
                     ('path', self.path)])

    @classmethod
    def load(cls, id):
        inst = cls()
        rc = RC()
        kv = rc.hgetall(cls.key(id))
        for k, v in kv.items():
            if k in ('id', 'path', 'bits'):
                setattr(inst, k, v)
            elif k == 'sign':
                pass
            else:
                setattr(inst, k, eval(v))
        inst.ism = ImageSignMaker(inst)
        return inst

class PerceivedCharacterCluster(Keyer, Dumper):
    PREFIX = 'pcc'

    def __init__(self):
        Keyer.__init__(self)
        Dumper.__init__(self)

    def _init_from_ch(self, ch):
        rc = RC()
        self.ch = ch
        if rc.exists(self.k):
            pc_ids = rc.smembers(self.k)
            self.cells = map(PerceivedCharacter.make_from_id, pc_ids)
        else:
            self.cells = []

    def add_perceived_character(self, pc):
        rc = RC()
        if rc.sismember(self.k, pc.id):
            return
        rc.sadd(self.k, pc.id)
        self.cells.append(pc)

    def delete_perceived_character(self, pc_id):
        rc = RC()
        if not rc.sismember(self.k, pc_id):
            return
        [self.cells.remove(pc) for pc in self.cells if pc.id == pc_id]
        return

    def serialize(self):
        return {
            "ch": self.ch,
            "cells": [cell.serialize() for cell in self.cells]
            }

    @property
    def k(self):
        return self.key(self.ch)
        
    @classmethod
    def make_from_ch(cls, ch):
        return cls.load(ch)

    @classmethod
    def load(cls, ch):
        inst = cls()
        inst._init_from_ch(ch)
        return inst

    def distance(self, splt_img):
        try:
            return min([cell.distance(splt_img) for cell in self.cells])
        except ValueError:
            return MAX_DISTANCE

    def is_contain_splited_image(self, splt_img):
        for cell in self.cells:
            if cell.is_contain_splited_image(splt_img):
                return True
        return False

class PerceivedCharacter(Keyer, Dumper):
    PREFIX = 'pc'

    def __init__(self):
        Keyer.__init__(self)
        Dumper.__init__(self)

    @classmethod
    def load(cls, id):
        inst = cls()
        rc = RC()
        kv = rc.hgetall(cls.key(id))
        for k, v in kv.items():
            setattr(inst, k, v if k not in ('sign',
                                            'members') else eval(v))
        return inst

    def _init_from_seed(self, id):
        si = SplitedImage.make_from_id(id)
        self.id = ID_GEN()
        self.sign = si.sign
        self.seed = id
        self.members = [id]

    def update(self, members):
        self.members = members
        self._update_sign()
        self.dump(force=True)

    def _update_sign(self):
        sims = map(SplitedImage.make_from_id, self.members)
        self.sign = em.EMPoint.sign_neighbour_points(sims)

    def serialize(self):
        return {
            'id': self.id,
            'sign': self.sign,
            'seed': self.seed,
            'members': self.members
            }

    def dump(self, force=False):
        dump = Dumper.force_dump if force else Dumper.dump
        dump(self, self.key(self.id),
             [('id', self.id),
              ('sign', str(self.sign)),
              ('seed', self.seed),
              ('members', str(self.members))])

    @classmethod
    def make_from_seed(cls, id):
        inst = cls()
        inst._init_from_seed(id)
        inst.dump()
        return inst

    @classmethod
    def make_from_id(cls, id):
        return cls.load(id)

    def distance(self, splt_img):
        return em.distance(self.sign, splt_img.sign)

    def is_contain_splited_image(self, splt_img):
        return em.distance(self.sign, splt_img.sign) < em.NEIGHBOUR_THRESHOLD

def get_perceived_character_cluster_instance(ch):
    global PERCEIVED_CHARACTER_CLUSTER
    if ch not in PERCEIVED_CHARACTER_CLUSTER:
        PERCEIVED_CHARACTER_CLUSTER[ch] = \
            PerceivedCharacterCluster.make_from_ch(ch)
    return PERCEIVED_CHARACTER_CLUSTER[ch]

def all_perceived_character_clusters():
    return map(get_perceived_character_cluster_instance, ALL_CHARACTERS)

def no_content_response():
    response = flask.Response()
    response.status_code = 204
    return response

def action_clean_perceived_character_cluster():
    global PERCEIVED_CHARACTER_CLUSTER
    PERCEIVED_CHARACTER_CLUSTER = {}
    rc = RC()
    ks = rc.keys(PerceivedCharacterCluster.PREFIX + ':*')
    for pcc in ks:
        pcs = rc.smembers(pcc)
        for pc in pcs:
            rc.delete(PerceivedCharacter.PREFIX + ':' + pc)
        rc.delete(pcc)

def action_clean_original_image_value():
    rc = RC()
    ks = rc.keys(OrigImage.PREFIX + ':*')
    [rc.hset(k, 'value', '****') for k in ks if rc.hget(k, 'splited') == 'True']

def action_resign_perceived_character_cluster():
    resign_perceived_character_clusters()

def all_seed_points():
    return map(em.EMPoint.make_from_seed,
               map(SplitedImage.make_from_id,
                   reduce(lambda x, acc: x + acc,
                          [[pc.seed 
                            for pc 
                            in get_perceived_character_cluster_instance(ch) \
                                .cells]
                           for ch in ALL_CHARACTERS], [])))

def all_em_points():
    return map(
        em.EMPoint.make_from_point, 
        map(SplitedImage.make_from_id, 
            SplitedImage.ids()))

def em_splited_points():
    EM = em.EM(all_seed_points(),
               all_em_points())
    EM.em()
    return EM.clusters

def resign_perceived_character_clusters():
    clusters = em_splited_points()
    ncls = dict([(cluster.id, 
                  [pt.id for pt in cluster.neighbours]) 
                 for cluster in clusters])
    opcs = reduce(lambda x, acc: x+acc, 
                  [pcc.cells for pcc in all_perceived_character_clusters()],
                  [])
    map(lambda pc: pc.update(ncls[pc.seed]), opcs)

def get_character_from_splited_id(splt_img_id):
    pccs = all_perceived_character_clusters()
    splt_img = SplitedImage.make_from_id(splt_img_id)
    for pcc in pccs:
        if pcc.is_contain_splited_image(splt_img):
            return pcc.ch
    return '*'
