import subprocess
datasets = ["visciano", "sorrento", "pontremoli", "barberino", "mezzolombardo", "predaia", "trento", "firenze", "napoli"]


def perc_diff(a, b):
    return (a-b)/a


def diff(a, b):
    return (a-b)


print("area centroids high best_hp")
for d in datasets:
    best_p = int(subprocess.getoutput(f"wc -l ../results/{d}/best_p_intervisibility.edgelist | cut -d' ' -f1"))
    centroids = int(subprocess.getoutput(f"wc -l ../results/{d}/centroids_intervisibility.edgelist | cut -d' ' -f1"))
    high_p = int(subprocess.getoutput(f"wc -l ../results/{d}/high_p_intervisibility.edgelist | cut -d' ' -f1"))
    best_p_hp = int(subprocess.getoutput(f"wc -l ../results/{d}/best_p_hp_intervisibility.edgelist | cut -d' ' -f1"))

    print(f"{d} {perc_diff(best_p, centroids):.3f} {perc_diff(best_p, high_p):.3f} {perc_diff(best_p, best_p_hp):.3f}  {diff(best_p, centroids)} {diff(best_p, high_p)} {diff(best_p, best_p_hp)}")

exit(0)
# Cartesian distance between points
distances = []
for idx, d in enumerate(datasets):
    best_p = pd.read_csv(f'results/{d}_2_2/best_p.csv', header=0, names=['id', 'x', 'y']).set_index('id')
    centroids = pd.read_csv(f'~/results_high/{d}_2_2/high_p.csv', header=0, names=['id', 'x', 'y']).set_index('id')
    data = best_p.join(centroids, how='left', lsuffix='_best_p', rsuffix='_centr')
    x_diff = (data['x_best_p'] - data['x_centr'])**2
    y_diff = (data['y_best_p'] - data['y_centr'])**2
    data['dist'] = np.sqrt(x_diff + y_diff)
    ax = plt.subplot(3, 3, idx+1)
    ax.set_title(d)
    ax.set_xlabel('cartesian distance (m)')
    ax.hist(data['dist'].values, bins=50, range=(0, 50))
    ax.set_xlim(0, 50)
    print(d, np.mean(data['dist'].values), np.std(data['dist'].values, ddof=1), np.max(data['dist'].values))
plt.show()
