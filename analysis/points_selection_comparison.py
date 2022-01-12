import subprocess
datasets = ["visciano", "sorrento", "pontremoli", "barberino", "mezzolombardo", "predaia", "trento", "firenze", "napoli"]

def perc_diff(a,b):
    return (a-b)/a


print("area centroids high best_hp")
for d in datasets:
    best_p = int(subprocess.getoutput(f"wc -l results/{d}/best_p_intervisibility.edgelist | cut -d' ' -f1"))
    centroids = int(subprocess.getoutput(f"wc -l results/{d}/centroids_intervisibility.edgelist | cut -d' ' -f1"))
    high_p = int(subprocess.getoutput(f"wc -l results/{d}/high_p_intervisibility.edgelist | cut -d' ' -f1"))
    best_p_hp = int(subprocess.getoutput(f"wc -l results/{d}/best_p_hp_intervisibility.edgelist | cut -d' ' -f1"))
    
    print(f"{d} {perc_diff(best_p, centroids):.3f} {perc_diff(best_p, high_p):.3f} {perc_diff(best_p, best_p_hp):.3f}")
    
