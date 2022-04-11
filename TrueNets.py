from rasterio.io import DatasetReader
from viewshed import Viewshed
import building_interface
import time
import csv
import argparse
import rasterio as rio
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
from rasterio.mask import mask


class TrueNets():
    def __init__(self, base_dir: str, raster_dir: str, comune: str, poi_elev: int, tgt_elev: int, dataset: str):
        self.vs = Viewshed()
        self.DSN = os.environ['DSN']
        self.srid = 3003  # TODO: check if it is metric
        self.crs = "EPSG:%4d" % (self.srid)
        self.base_dir = base_dir
        self.raster_dir = raster_dir
        self.poi_elev = poi_elev
        self.tgt_elev = tgt_elev
        self.comune = comune
        self.dataset_type = dataset

    def convert_matrix_to_simple_edgelist(self, matrix: np.ndarray, filename: str):
        with open("%s/%s.edgelist" % (self.base_dir, filename), 'w') as fw:
            for i, np_line in enumerate(tqdm(matrix)):
                fw.write("\n")
                idx = np.nonzero(np_line)[0]
                def fx(x): return "%d %d" % (self.ordered_coordinates[i][0],
                                             self.ordered_coordinates[x][0])
                neighs = list(map(fx, idx))
                fw.write('\n'.join(neighs))

    def convert_matrix_to_adj(self, matrix, filename):
        with open("%s/%s.adj" % (self.base_dir, filename), 'w') as fw:
            csv_w = csv.writer(fw)
            for i, np_line in enumerate(tqdm(matrix)):
                idx = np.nonzero(np_line)[0]
                neighs = [self.ordered_coordinates[i][0]]
                def fx(x): return self.ordered_coordinates[x][0]
                neighs.extend(map(fx, idx))
                csv_w.writerow(neighs)

    def convert_matrix_to_jpg(self, matrix: np.ndarray, filename: str, mult: int = 255):
        im = Image.fromarray(np.uint8(matrix*mult))
        im.save(self.base_dir+'/'+filename+".jpg")

    def read_raster(self, path: str) -> DatasetReader:
        self.dataset = rio.open(path, crs=self.crs)
        return self.dataset.read(1)

    def save_raster(self, data: np.ndarray, filename: str):
        trans = rio.transform.from_origin(self.dataset.bounds.left, self.dataset.bounds.top, self.dataset.res[0], self.dataset.res[1])
        new_dataset = rio.open('%s' % (filename), 'w', driver='GTiff',
                               height=data.shape[0],
                               width=data.shape[1],
                               count=1, dtype=str(data.dtype),
                               crs=self.dataset.crs,
                               transform=trans
                               )
        new_dataset.write(data, 1)
        new_dataset.close()

    def convert_coordinates(self, coordinates: 'tuple[float, float]') -> np.ndarray:
        mapped_coord = self.dataset.index(coordinates[0], coordinates[1])
        coordinates_np = np.array([mapped_coord[0], mapped_coord[1]], dtype=np.int32)
        return coordinates_np

    def get_comune(self, name: str):
        print(name)
        if name == "Barberino":
            name = "Barberino di Mugello"
        if(self.dataset_type == 'osm'):
            self.BI = building_interface.OSMInterface(self.DSN, srid=self.srid)
        elif(self.dataset_type == 'ctr'):
            self.BI = building_interface.CTRInterface(self.DSN, srid=self.srid)
        else:
            print("Invalid dataset type chose osm or CTR")
            exit(1)
        self.area = self.BI.get_area(name)
        self.buildings = self.BI.get_buildings(shape=self.area)
        if(len(self.buildings) == 0):
            print("Warning: 0 buildings in the area")
        else:
            print("%d buildings" % (len(self.buildings)))

    def single_viewshed(self, raster_file: str, poi_coord: 'tuple[float, float]', output: str):
        raster = self.read_raster(raster_file)
        poi = self.convert_coordinates(poi_coord)
        viewshed = self.vs.single_viewshed(raster, np.array(poi[0], poi[1]), self.poi_elev, self.tgt_elev)
        self.save_raster(viewshed, output)

    def cumulative_viewshed_highest_p(self):
        self.get_comune(self.comune)
        raster = self.read_raster("%s/%s.tif" % (self.raster_dir, self.comune.lower()))
        high_p = np.loadtxt("%s/high_p.csv" % (self.base_dir), delimiter=',', skiprows=1)
        highest = [(p[1], p[2]) for p in high_p]
        highest_coords = [self.convert_coordinates(c) for c in highest]
        self.vs.prepare_cumulative_viewshed(raster)
        result = self.vs.cumulative_viewsheds(highest_coords, self.poi_elev, self.tgt_elev)
        self.save_raster(result, "%s/cumulative_hp.tif" % (self.base_dir))

    def cumulative_viewshed(self):
        self.get_comune(self.comune)
        raster = self.read_raster("%s/%s.tif" % (self.raster_dir, self.comune.lower()))
        self.vs.prepare_cumulative_viewshed(raster)
        centroids = [b.xy() for b in self.buildings]
        coords = [self.convert_coordinates(c) for c in centroids]
        result = self.vs.cumulative_viewsheds(coords, self.poi_elev, self.tgt_elev)
        self.save_raster(result, "%s/cumulative.tif" % (self.base_dir))

    def find_best_point(self, highest=False):
        self.get_comune(self.comune)
        if highest:
            in_filename = 'cumulative_hp.tif'
            out_filename = 'best_p_hp.csv'
        else:
            in_filename = 'cumulative.tif'
            out_filename = 'best_p.csv'
        raster = self.read_raster(f"{self.base_dir}/{in_filename}")  # must be a metric reference system
        results = np.zeros(shape=(len(self.buildings), 3))
        for idx, b in enumerate(tqdm(self.buildings)):
            roof = b.shape()
            relative_visibility = mask(self.dataset, roof, crop=True)
            relative_position = np.unravel_index(relative_visibility[0].argmax(), relative_visibility[0].shape)
            abs_pos = rio.transform.xy(relative_visibility[1], relative_position[1], relative_position[2], offset='ul')
            results[idx] = [b.id(), abs_pos[0], abs_pos[1]]
        np.savetxt(f"{self.base_dir}/{out_filename}", results, fmt='%d', delimiter=',', header="id,x,y")

    def find_highest_point(self):
        self.get_comune(self.comune)
        raster = self.read_raster("%s/%s.tif" % (self.raster_dir, self.comune.lower()))
        results = np.zeros(shape=(len(self.buildings), 3))
        for idx, b in enumerate(tqdm(self.buildings)):
            roof = b.shape()
            relative_visibility = mask(self.dataset, roof, crop=True)
            max = relative_visibility[0].argmax()
            relative_position = np.unravel_index(relative_visibility[0].argmax(), relative_visibility[0].shape)
            abs_pos = rio.transform.xy(relative_visibility[1], relative_position[1], relative_position[2], offset='ul')
            results[idx] = [b.id(), abs_pos[0], abs_pos[1]]
        np.savetxt("%s/high_p.csv" % (self.base_dir), results, fmt='%d', delimiter=',', header="id,x,y")

    def find_centroids(self):
        self.get_comune(self.comune)
        # raster = self.read_raster("%s/cumulative.tif"%(self.base_dir)) #must be a metric reference system
        results = np.zeros(shape=(len(self.buildings), 3))
        for idx, b in enumerate(tqdm(self.buildings)):
            roof = b.shape()
            c = roof.representative_point()
            results[idx] = [b.id(), c.x, c.y]
        np.savetxt("%s/centroids.csv" % (self.base_dir), results, fmt='%d', delimiter=',', header="id,x,y")

    def generate_intervisibility_fast(self, filename='best_p'):
        raster = self.read_raster("%s/%s.tif" % (self.raster_dir, self.comune.lower()))
        point_list = np.genfromtxt(f"{self.base_dir}/{filename}.csv", delimiter=',')  # array of (id, x, y)
        point_list = point_list[:]
        building_n = len(point_list)
        mapped_coordinates = np.zeros(shape=(building_n, 2), dtype=np.int32)
        self.ordered_coordinates = np.array(sorted(point_list, key=lambda x: x[0]), dtype=np.int32)
        for idx, c in enumerate(self.ordered_coordinates):
            mc = self.dataset.index(c[1], c[2])
            mapped_coordinates[idx] = (mc[0], mc[1])
        print(time.time())
        result = self.vs.generate_intervisibility_fast(raster, mapped_coordinates, self.poi_elev, self.tgt_elev)
        print(time.time())
        print(building_n)
        #np.savetxt(f"{self.base_dir}/{filename}ivg.csv", result, delimiter=',', fmt='%d')
        self.convert_matrix_to_simple_edgelist(result, f"{filename}_intervisibility")
        self.convert_matrix_to_adj(result, f"{filename}_intervisibility")

    def distance(self, filename='best_p_intervisibility.adj'):
        adj_list = []
        with open(f"{self.base_dir}/{filename}", 'r') as csv_f:
            csv_r = csv.reader(csv_f, delimiter=',')
            for l in csv_r:
                if(l):
                    line = []
                    for c in l:
                        line.append(int(c))
                    adj_list.append(line)
        raster = self.read_raster("%s/%s.tif" % (self.raster_dir, self.comune.lower()))
        point_list = np.genfromtxt("%s/best_p.csv" % (self.base_dir), delimiter=',')
        self.ordered_coordinates = np.array(sorted(point_list, key=lambda x: x[0]), dtype=np.uint32)
        # Build a dictionary with the mapped coordinates of the buildings and the index
        self.coordinates_dict = {}
        for idx, c in enumerate(self.ordered_coordinates):
            mc = self.dataset.index(c[1], c[2])
            self.coordinates_dict[c[0]] = [mc[0], mc[1], idx]
        # call the function
        result = self.vs.calculate_distance(adj_list,
                                            self.coordinates_dict,
                                            self.ordered_coordinates.shape[0])
        #np.savetxt("%s/distance.csv" % (self.base_dir), result, delimiter=',', fmt='%d')
        with open("%s/%s.edgelist" % (self.base_dir, "distance"), 'w') as fw:
            for i in tqdm(range(result.shape[0])):
                distance_line = result[i]
                fw.write("\n")
                idx = np.nonzero(distance_line)[0]

                def fx(x): return "%d %d %.2f " % (self.ordered_coordinates[i][0],
                                                   self.ordered_coordinates[x][0],
                                                   distance_line[x])  # src dst distance
                neighs = list(map(fx, idx))
                fw.write('\n'.join(neighs))

    def get_building_height(self):
        raster = self.read_raster("%s/%s.tif" % (self.raster_dir, self.comune.lower()))
        point_list = np.genfromtxt("%s/best_p.csv" % (self.base_dir), delimiter=',')  # array of (id, x, y)
        building_n = len(point_list)
        mapped_coordinates = np.zeros(shape=(building_n, 2), dtype=np.uint32)
        self.ordered_coordinates = np.array(sorted(point_list, key=lambda x: x[0]), dtype=np.uint32)
        #id_h = np.zeros(shape=(building_n), dtype=np.dtype(['int', 'float']))
        with open(f"{self.base_dir}/heights.csv", 'w') as fw:
            writer = csv.writer(fw, delimiter=',')
            for idx, c in enumerate(self.ordered_coordinates):
                mc = self.dataset.index(c[1], c[2])
                h = raster[mc[0], mc[1]]
                if h != -9999:
                    writer.writerow([f'{c[0]}', f'{h:.2f}'])
        #np.savetxt("%s/heights.csv"%(self.base_dir), id_h, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Truenets utility for offline intervisibility')
    parser.add_argument("-c", "--comune",
                        help="Nome del comune da analizzare",
                        required=True)
    parser.add_argument("-d", "--dataset", help="raster dataset to use (ctr or osm)", default="osm")
    parser.add_argument("-r", "--raster_dir",
                        help="Percorso della cartella contenente i rasters",
                        required=True)
    parser.add_argument("-o", "--output", help="dir for output files",
                        required=True)
    parser.add_argument("-pe", "--poi_elev", help="height of both poles",
                        type=int,
                        required=True)
    parser.add_argument("-te", "--tgt_elev", help="height of both poles",
                        type=int,
                        required=False)

    parser.add_argument("-cv", help="calculate the cumulative viewshed using the centroid",
                        action='store_true')
    parser.add_argument("-cvh", help="calculate the cumulative viewshed using the highest point",
                        action='store_true')
    parser.add_argument("-fb", help="find best point for each building",
                        action='store_true')
    parser.add_argument("-fbh", help="find best point for each building using the highest point",
                        action='store_true')
    parser.add_argument("-fh", help="find highest point for each building",
                        action='store_true')

    parser.add_argument("-fc", help="find centroids for each building",
                        action='store_true')
    parser.add_argument("-gi", help="generate intervisibility matrix",
                        action='store_true')
    parser.add_argument("-gif", help="generate intervisibility fast matrix",
                        action='store_true')
    parser.add_argument("-ke", help="calculate knife_edge",
                        action='store_true')
    parser.add_argument("-bh", help="calculate building heights",
                        action='store_true')
    parser.add_argument("-dist", help="calculate distance",
                        action='store_true')
    parser.add_argument("-all", type=int)
    parser.add_argument("-debug", help="debug on a smaller area",
                        action='store_true')

    args, unknown = parser.parse_known_args()
    if not args.tgt_elev:
        args.tgt_elev = args.poi_elev

    tn = TrueNets(args.output, args.raster_dir, args.comune, args.poi_elev, args.tgt_elev, args.dataset)
    if(args.cv):
        tn.cumulative_viewshed()
    if(args.cvh):
        tn.cumulative_viewshed_highest_p()
    if(args.fb):
        tn.find_best_point(highest=False)
    if(args.fbh):
        tn.find_best_point(highest=True)
    if(args.fh):
        tn.find_highest_point()
    if(args.fc):
        tn.find_centroids()
    if(args.gif):
        tn.generate_intervisibility_fast()
    if(args.dist):
        tn.distance()
    if(args.bh):
        tn.get_building_height()

    if(args.all == 0):
        tn.find_highest_point()
        tn.cumulative_viewshed_highest_p()
        tn.find_best_point(highest=True)
        tn.generate_intervisibility_fast(filename='best_p_hp')

    elif(args.all == 1):
        tn.find_highest_point()
        tn.generate_intervisibility_fast(filename='high_p')

    elif(args.all == 2):
        tn.cumulative_viewshed()
        tn.find_best_point()
        tn.generate_intervisibility_fast(filename='best_p')

    elif(args.all == 3):
        tn.find_centroids()
        tn.generate_intervisibility_fast(filename='centroids')
