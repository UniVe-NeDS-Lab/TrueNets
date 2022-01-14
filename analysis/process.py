import networkx as nx
import argparse
import collections
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import geopandas as gpd
import numpy as np
import csv
import math
from scipy.optimize import minimize
from tqdm import tqdm

def los_nlos_binned(path, datasets, pe, te):
    for d in datasets:
        data = np.loadtxt("%s/%s_%d_%d/link_disthist.csv"%(path,d,pe,te), delimiter=' ').T
        width = 0.7 * (data[0][1] - data[0][0])
        center = (data[0][:-1] + data[0][1:]) / 2
        print(data[1])
        plt.bar(center, data[1][:-1], align='center', width=width)
        plt.title(d)
        plt.show()
        plt.clf()


def degree_distribution(path, datasets,pe,te):
    for d in datasets:
        conn_comp = []
        with open("%s/%s_%d_%d/biggest_component.csv"%(path,d,pe,te)) as fr:
            for n in fr.readlines():
                conn_comp.append(n[:-1])
        G = nx.read_edgelist("%s/%s_%d_%d/loss_graph.edgelist"%(path,d,pe,te), delimiter = " ", data=(('weight', float),('src_orient', float), ('dst_orient', float)))#.to_undirected()
        con_G = G.subgraph(conn_comp)
        print("size of graph %d, number of nodes in bcc %d, number of node in subgraph %d"%(len(G), len(conn_comp), len(con_G)))
        degree_sequence = sorted([d for n, d in con_G.degree()], reverse=False)  # degree sequence
        degrees = np.array(degree_sequence)
        hist = np.histogram(degrees, bins=int((degrees.max()-degrees.min())))
        h = np.zeros(shape=(len(hist[0]), 2))
        h[:, 1] = hist[0]
        h[:, 0] = hist[1][:-1]
        np.savetxt("%s/graphs/degree_dist_1bin/%s.csv" % (path, d), h)


def loss_on_dist(path, datasets,pe,te):
    for d in datasets:
        try:
            G_d = pd.read_csv("%s/%s_%d_%d/distance.edgelist"%(path,d,se,te), sep=' ', usecols=[0,1,2], header=None, names=['src', 'dst', 'distance'])
            G_l = pd.read_csv("%s/%s_%d_%d/loss_graph.edgelist"%(path,d,se,te), sep=' ', usecols=[0,1,2], header=None, names=['src', 'dst', 'loss'])
            G = pd.merge(G_d, G_l, on=['src', 'dst'])
            data = G[['distance', 'loss']].to_numpy()
            np.savetxt("%s/graphs/dist_loss/%s.csv" % (path, d), data)
        except FileNotFoundError:
            continue

def distance_histogram_merge(path, datasets,pe,te):
    for d in datasets:
        try:
            G = pd.read_csv("%s/%s_%d_%d/distance.edgelist"%(path,d,pe,te), sep=' ', usecols=[2], header=None, names=['distance'])
            distances = G.to_numpy()
            hist = np.histogram(distances, bins=range(0,int(np.amax(distances)), 100), density=True)
            h = np.zeros(shape=(len(hist[0]), 2))
            h[:, 1] = hist[0]
            h[:, 0] = hist[1][:-1]
            np.savetxt("%s/graphs/dist_hist/%s.csv" % (path, d), h)
            # ax = G.plot.hist(bins=int((G.distance.max()-G.distance.min())/100))
            # plt.savefig("%s/graphs/dist_hist/%s.pdf" % (path, d), format="pdf")
        except FileNotFoundError:
            continue

def relative_distance_histogram(path, datasets,pe,te):
    # for d in datasets:
    #     try:
    #         G = pd.read_csv("%s/%s_%d_%d/distance.edgelist"%(path,d,pe,te), sep=' ', usecols=[2], header=None, names=['distance'])
    #         distances = G.to_numpy()
    #         hist = np.histogram(distances, bins=range(0,int(np.amax(distances)), 100))
    #         h = np.zeros(shape=(len(hist[0]), 2))
    #         h[:, 1] = hist[0]
    #         h[:, 0] = hist[1][:-1]
    #         np.savetxt("%s/graphs/reldist_hist/los_%s.csv" % (path, d), h)
    #         # ax = G.plot.hist(bins=int((G.distance.max()-G.distance.min())/100))
    #         # plt.savefig("%s/graphs/dist_hist/%s.pdf" % (path, d), format="pdf")
    #     except FileNotFoundError:
    #         continue

    for d in datasets:
        try:
            alllink = np.loadtxt("%s/%s_%d_%d/link_disthist.csv"%(path,d,pe,te), delimiter=' ')
            loslink = np.loadtxt("%s/graphs/reldist_hist/los_%s.csv" % (path, d), delimiter=' ')
            rel_link = np.zeros(shape=loslink.shape)
            for idx, l in enumerate(loslink):
                rel_link[idx][0] = l[0]
                rel_link[idx][1] = l[1]/alllink[idx][1]
            np.savetxt("%s/graphs/reldist_hist/%s.csv" % (path, d), rel_link)
        except FileNotFoundError:
            continue

def calc_ecdf(data):
    ecdf = np.zeros(shape=(len(data), 2))
    x = np.sort(data[:,0], axis=0)
    y = np.arange(1,len(data)+1)/len(data)
    ecdf[:, 0] = x
    ecdf[:, 1] = y

    return ecdf

def distance_ecdf(path, datasets,pe,te):
    for d in datasets:
        try:
            G = pd.read_csv("%s/%s_%d_%d/distance.edgelist"%(path,d,pe,te), sep=' ', usecols=[2], header=None, names=['distance'])
            distances = G.to_numpy()
            ecdf = calc_ecdf(distances)
            #plt.plot(ecdf[0],ecdf[1], linestyle='none', marker='.')
            np.savetxt("%s/graphs/dist_ecdf/%s.csv" % (path, d), ecdf)
            # ax = G.plot.hist(bins=int((G.distance.max()-G.distance.min())/100))
            #plt.savefig("%s/graphs/dist_ecdf/%s.png" % (path, d), format="png")
            #plt.clf()
        except FileNotFoundError:
            continue

def find_best_plexp(X, Y):
    def model(params, X):
        exp = params[0]
        return exp*10*np.log10(5000000000*X*4*math.pi/299792458)

    def sum_of_squares(params, X, Y):
        y_pred = model(params, X)
        obj = np.sqrt((y_pred-Y)**2).sum()
        return obj

    #import pdb; pdb.set_trace()
    res = minimize(sum_of_squares, [2.0, ], args=(X,Y), tol=1e-3, method='Powell')
    return(res)


def distance_histogram_process(path, datasets,pe,te):
    fig = plt.figure(figsize=(15, 15))
    binsize = 50
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    for idx, d in enumerate(tqdm(datasets)):
        try:
            G = pd.read_csv("%s/graphs/dist_loss/%s.csv" % (path, d), sep=' ', names=['distance','loss'])
            G = G.sort_values('distance')
            np_G = G.to_numpy()
            n_bins = int((G.distance.max()-G.distance.min())/binsize) #bin each 500m
            bins = np.arange(0, G.distance.max(), binsize)
            binplace = np.digitize(np_G[:,0], bins)
            data = np.zeros(shape=(n_bins, 6))
            data[:,:] = np.nan
            data[:,0] = bins[:-1]
            sets = []
            for i in range(0,n_bins):
                binvals = np.take(np_G[:,1], indices=np.where(binplace == i+1)[0])
                if(len(binvals)):
                    #empty bins are NaN and will be removed later
                    data[i,1] = np.mean(binvals)
                    data[i,2] = np.median(binvals)
                    data[i,3] = np.std(binvals)
                    data[i,4] = np.min(binvals)
                    data[i,5] = np.max(binvals)
                sets.append(binvals)
            data = data[~np.isnan(data).any(axis=1)]

            f = 5000000000
            c = 299792458
            def fspl(d,e):
                return e*10*np.log10(f*d*4*math.pi/c)
            x = np.linspace(G.distance.min(), G.distance.max(), num=500)
            ax = plt.subplot(3,3,idx+1)

            #plt.scatter(np_G[:,0], np_G[:,1])
            #plt.errorbar(bins[:-1],means,stds, c='red')
            y1 = (data[:,1]-data[:,3]).squeeze()
            y2 = (data[:,1]+data[:,3]).squeeze()
            m1, = ax.plot(data[:,0]/1000, data[:,1], label="")
            s1 = ax.fill_between(data[:,0]/1000, y1, y2, alpha=0.5)
            s2 = ax.fill_between(data[:,0]/1000, data[:,4].squeeze(), data[:,5].squeeze(), alpha=0.2)
            #ax.boxplot(sets, positions=list(map(int,bins[:-1]+binsize/2)), widths=binsize/2)
            ax.set(xlabel="Link length (km)")
            ax.set(ylabel="path loss (dB)")
            if d=='trento':
                title = 'Urban 1'
            elif d=='firenze':
                title = 'Urban 2'
            elif d=='napoli':
                title = 'Urban 3'
            elif d=='mezzolombardo':
                title = 'Suburban 1'
            elif d=='barberino':
                title = 'Suburban 2'
            elif d=='sorrento':
                title = 'Suburban 3'
            elif d=='predaia':
                title = 'Rural 1'
            elif d=='pontremoli':
                title = 'Rural 2'
            elif d=='visciano':
                title = 'Rural 3'
            else:
                title = d

            ax.set_title(title)
            #best_e_all = find_best_plexp(np_G[:, 0], np_G[:, 1])
            best_e_mean = find_best_plexp(data[:, 0]+binsize/2, data[:, 1])
            best_e_median = find_best_plexp(data[:, 0]+binsize/2, data[:, 2])
            f1, = ax.plot(x/1000,fspl(x,2), c='red', label="")
            #ax.plot(x,fspl(x,best_e_all.x), c='green', label="FSPL with best exp %.3f - all points"%(best_e_all.x))
            #ax.plot(x,fspl(x,best_e_mean.x), c='orange', label="FSPL with best exp %.3f - mean"%(best_e_mean.x))
            f2, = ax.plot(x/1000,fspl(x,best_e_median.x), c='yellow', label="exp %.3f"%(best_e_median.x))
            ax.legend()
        except FileNotFoundError:
            continue
    # plt.figlegend((m1, s1, s2, f1, f2),
    #               ("mean of binned loss",
    #                "std of binned loss",
    #                "min/max of binned loss",
    #                "FSPL with exp 2",
    #                "FSPL with best exp fitted on median"),
    #               ncol=5,
    #               labelspacing=0.,
    #               framealpha=1 )
    plt.tight_layout()
    plt.savefig("%s/graphs/dist_loss/total.png" % (path))
    #plt.show()
    plt.close()

def loss_histogram(path, datasets,pe,te):
    for d in datasets:
        G = pd.read_csv("%s/%s_%d_%d/loss_graph.edgelist"%(path,d,pe,te), sep=' ', usecols=[2], header=None, names=['loss'])
        loss = G.to_numpy()
        min = G.loss.min()
        max = G.loss.max()
        hist = np.histogram(loss, bins=int((max-min)), density=True)
        h = np.zeros(shape=(len(hist[0]), 2))
        h[:, 1] = hist[0]
        h[:, 0] = hist[1][:-1]
        np.savetxt("%s/graphs/loss_hist/%s.csv" % (path, d), h)
        # ax = G.plot.hist(bins=int((max-min)))
        # plt.savefig("%s/graphs/loss_hist/%s.pdf" % (path, d), format="pdf")

def connected_comp(path, datasets, pe, te):
    for d in datasets:
        G = nx.read_edgelist("%s/%s_%d_%d/loss_graph.edgelist"%(path,d,pe,te),
                             data=(("loss", int),
                                   ("yaw", float),
                                   ("tilt", float)))
        cc = sorted(nx.connected_components(G), key = len, reverse=True)
        with open("%s/%s_%d_%d/biggest_component.csv"%(path,d,pe,te), 'w') as fw:
            for n in cc[0]:
                print(n, file=fw)
        print("%s has %d connected components and the largest one has %d of %d nodes"%(d,len(cc), len(cc[0]), len(G)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyize graphs')
    parser.add_argument("-P", "--processes", help="number of parallel processes",
                             default=1, type=int)
    parser.add_argument('-b', "--base_path", help="path of the graph",
                             type=str, required=True)
    parser.add_argument('-pe', "--pole_elev", help="heigh of the pole in meters (2,4,10)", type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    processes = args.processes
    pe = te = args.pole_elev
    datasets = ['trento', 'firenze', 'napoli',  'mezzolombardo', 'barberino', 'sorrento', 'predaia', 'pontremoli', 'visciano',  ]
    #datasets = ['barberino',  'mezzolombardo', 'predaia', 'pontremoli', 'visciano', 'predaia', 'pontremoli', 'visciano', 'barberino']
    #loss_histogram(base_path, datasets,pe,te)
    distance_histogram_process(base_path, datasets,pe,te)
    #degree_distribution(base_path, datasets,pe,te)
    #relative_distance_histogram(base_path, datasets,pe,te)
    #los_nlos_binned(base_path,datasets,pe,te)
    #connected_comp(base_path,datasets,pe,te)
