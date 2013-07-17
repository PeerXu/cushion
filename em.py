#!/usr/bin/env python2.7
"""
@author: Peer Xu
@email: pppeerxu@gmail.com
@TODO: XXX
"""

NEIGHBOUR_THRESHOLD = 5

def distance(v, w):
    return sum([(a-b)**2 for a, b in zip(v, w) if a or b]) ** 0.5

def load_orig_clusters(f):
    with open(f) as fr:
        return dict([line.split() \
                         for line in fr.read().split('\n') if line != ''])

def load_cvs(f, g):
    cvls = load_orig_clusters(f)
    return {k: g[v] for k, v in cvls.iteritems()}

def find_closest_cluster(cvs, v):
    d, c = min([(distance(v, w), k) for k, w in cvs.items()])
    return c if d <= 100 else '_'

def average(vs):
    return [sum(l)*1.0/len(l) for l in apply(zip, vs)]

def expect(vs):
    return average(vs)

def is_convergent(cvs, nvs):
    return all([distance(cvs[k], nvs[k]) < 0.2 for k in cvs.keys()])

def sort_result(nvs, vst):
    def _sort(exp_v, vs):
        return [w for _d, w in sorted([(distance(exp_v, w), w) for w in vs])]
    return {k: _sort(nvs[k], vst[k]) for k in nvs.keys()}

def init_vst(cvs):
    vst = {k: [] for k in cvs.keys()}
    vst.update({'_': []})
    return vst

def em(g):
    cvs = load_cvs('orig-clusters.txt', g)
    count = 0
    while True:
        vst = init_vst(cvs)
        [vst[find_closest_cluster(cvs, v)].append(v) for _k, v in g.items()]
        nvs = {k: expect(v) for k, v in vst.items()}
        if is_convergent(cvs, nvs):
            return sort_result(nvs, vst)
        cvs = {k: v for k, v in nvs.items() if len(v)}
        count += 1
        print count

def make_graph(f):
    graph = {}
    with open(f) as fr:
        for line in fr:
            if line == '':
                continue
            k, sv = (lambda k, *sv: (k, map(int, sv)))(*line.split())
            graph[k] = sv
    return graph

def _main(script, *args):
    g = make_graph('signed.txt')
    s2p = {str(v): k for k, v in g.items()}
    rt = em(g)
    with open('output.txt', 'w') as fw:
        fw.write(str(rt))
    with open('s2p.txt', 'w') as fw:
        fw.write(str(s2p))

import functools

class EMPoint(object):
    def __init__(self):
        pass

    @classmethod
    def average(cls, *vs):
        return sum(vs)*1.0/len(vs)

    def _init_from_point(self, point):
        self.id = str(point.id)
        self.sign = list(point.sign)

    @classmethod
    def make_from_seed(cls, seed_point):
        inst = cls()
        inst._init_from_point(seed_point)
        return inst

    @classmethod
    def make_from_point(cls, point):
        inst = cls()
        inst._init_from_point(point)
        return inst

    @classmethod
    def sign_neighbour_points(cls, neighbour_points):
        return apply(functools.partial(map, cls.average), 
                     [p.sign for p in neighbour_points])

    def _init_from_neighbour(self, seed_point, neighbour_points):
        self.id = str(seed_point.id)
        self.sign = list(self.sign_neighbour_points(neighbour_points))

    @classmethod
    def make_from_neighbour(cls, seed_point, neighbour_points):
        inst = cls()
        inst._init_from_neighbour(seed_point, neighbour_points)
        return inst

    def distance(self, point):
        return distance(self.sign, point.sign)

class EMCluster(object):
    @classmethod
    def mid_point(cls, cluster):
        return EMPoint.make_from_neighbour(cluster.seed, cluster.neighbours)
    
    def __init__(self, seed, neighbours):
        self.seed = seed
        self.neighbours = neighbours
        self.mid = self.mid_point(self)

    @property
    def id(self):
        return self.seed.id

    @property
    def sign(self):
        return self.mid.sign

    def distance(self, cluster):
        return self.mid.distance(cluster.mid)

class EM(object):
    def __init__(self, seeds, points):
        self.seeds = seeds
        self.points = points
        self.clusters = map(
            EMCluster, 
            self.seeds, 
            [[seed] for seed in self.seeds])

    def point_in_cluster(self, point):
        try:
            return min(filter(lambda pair: pair[0] < NEIGHBOUR_THRESHOLD, 
                              [(distance(
                                cluster.sign, point.sign), 
                                cluster) for cluster in self.clusters]))[1]
        except Exception, ex:
            return None

    def iterable(self, clusters):
        cluster_dict = {}
        for cluster in self.clusters:
            cluster_dict[cluster.id] = cluster

        for cluster in clusters:
            if cluster.distance(cluster_dict[cluster.id]) > 1:
                return True
        return False

    def em(self):
        while True:
            point_set = {}
            for point in self.points:
                cluster = self.point_in_cluster(point)
                if cluster != None:
                    if cluster not in point_set:
                        point_set[cluster] = []
                    point_set[cluster].append(point)
            new_clusters = []
            for cluster, neighbours in point_set.items():
                new_clusters.append(EMCluster(cluster.seed, neighbours))
            if not self.iterable(new_clusters):
                break
            self.clusters = new_clusters

if __name__ == '__main__':
    import sys
    _main(*sys.argv)
