import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba import cuda, types
import math
import numpy as np

threads_n = 256
EARTH_RADIUS = 6371
EARTH_RADIUS_SQUARED = 40589641
CORNER_OVERLAP = 16

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def take_values(viewshed, positions, intervisibility_mat, index):
    i = cuda.grid(1)
    if i < positions.shape[0] and i != index:
        x, y = positions[i]
        intervisibility_mat[index, i] = viewshed[x, y]


@cuda.jit
def memset_k(array, val):
    if len(array.shape) >= 2:
        # 2d
        i, j = cuda.grid(2)
        if i < array.shape[0] and j < array.shape[1]:
            array[i, j] = val
    else:
        # 1d
        i = cuda.grid(1)
        if i < array.shape[0]:
            array[i] = val


@cuda.jit
def sum_k(viewshed, cumulative):
    i, j = cuda.grid(2)
    if i < cumulative.shape[0] and j < cumulative.shape[1]:
        cumulative[i, j] += viewshed[i, j]
        viewshed[i, j] = 0


@cuda.jit
def viewshed_k(dsm, out, poi, max_dist,  width_resol, height_resol, poi_elev, tgt_elev, poi_elev_type):
    if max_dist == 0:
        max_dist = 65536
    p_offset = cuda.shared.array(shape=(threads_n), dtype=types.float32)  # TODO: fix data type
    p_id = cuda.shared.array(shape=(threads_n, 2), dtype=types.int32)
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x - cuda.blockIdx.x * 2
    p_offset[cuda.threadIdx.x] = 1
    p_id[cuda.threadIdx.x] = (-1, -1)

    x_step = np.float32(1)
    y_step = np.float32(1)
    # INITIALIZE DATA WRT to DIRECTION OF TRACING
    if(cuda.blockIdx.y == 0):
        # NORTH
        if(tid >= dsm.shape[0]+CORNER_OVERLAP):
            return
        x_end = tid - CORNER_OVERLAP/2.0
        y_end = np.float32(0)
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        x_step = abs(xs/ys)
        if x_step > 1.0:
            x_step = np.float32(1)
            y_step = ys/xs
        north_south = True
    elif(cuda.blockIdx.y == 1):
        # SOUTH
        if(tid >= dsm.shape[0]+CORNER_OVERLAP):
            return
        x_end = tid - CORNER_OVERLAP/2.0
        y_end = dsm.shape[1]

        xs = x_end - poi[0]
        ys = y_end - poi[1]
        x_step = abs(xs/ys)
        if x_step > 1.0:
            x_step = np.float32(1)
            y_step = ys/xs
        north_south = True
    elif(cuda.blockIdx.y == 2):
        # WEST
        if(tid >= dsm.shape[1]+CORNER_OVERLAP):
            return
        x_end = np.float32(0)
        y_end = tid - CORNER_OVERLAP/2.0
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        y_step = abs(ys/xs)
        if y_step > 1.0:
            y_step = np.float32(1)
            x_step = xs/ys
        north_south = False
    elif(cuda.blockIdx.y == 3):
        # EAST
        if(tid >= dsm.shape[1]+CORNER_OVERLAP):
            return
        x_end = dsm.shape[0]
        y_end = tid - CORNER_OVERLAP/2.0
        xs = x_end - poi[0]
        ys = y_end - poi[1]
        y_step = abs(ys/xs)
        if y_step > 1.0:
            y_step = np.float32(1)
            x_step = xs/ys
        north_south = False
    max_tilt = np.float32(np.inf)
    if(x_step < 0.0):
        x_step = -x_step
    if(xs < 0.0):
        x_step = -x_step

    if(y_step < 0.0):
        y_step = -y_step
    if(ys < 0.0):
        y_step = -y_step

    x = np.float32(0)
    y = np.float32(0)
    colsd = dsm.shape[0]
    rowsd = dsm.shape[1]
    if poi_elev_type == 0:
        # normal old type (origin point + elev)
        h0 = dsm[poi[0], poi[1]] + poi_elev
    elif poi_elev_type == 1:
        # use this to specify the height as third column in coords
        h0 = poi[2] + poi_elev
    step = -1
    while(True):
        step += 1
        x = x_step * step + poi[0] + 0.5
        y = y_step * step + poi[1] + 0.5
        # Exit conditions
        if(x < 0.5):
            break
        if((x+1.0) >= colsd):
            break
        if(y < 0.5):
            break
        if((y+1.0) >= rowsd):
            break

        # distance from poi
        xd = x-poi[0]
        yd = y-poi[1]

        #xd *= width_resolution
        # yd *= height_resolution #useless mult by 1

        # 2-d distance
        distance = math.sqrt(xd*xd + yd*yd)

        # take neighbor point for closest trace interpolation (see section 4.5 of paper)
        p1 = (int(x), int(y))
        if distance < 0.01:
            out[p1] = 1

            continue  # too short distance
        if distance > max_dist:
            return

        if(north_south):
            offset = x-int(x)
        else:
            offset = y-int(y)

        p_offset[cuda.threadIdx.x] = offset
        p_id[cuda.threadIdx.x] = p1
        cuda.syncthreads()
        point_consider = 1
        # If the left or right sweep is more precise do not consider this one
        if (((cuda.threadIdx.x < (cuda.blockDim.x-1)) and
             (cuda.threadIdx.x != 0)) or tid == 0):
            if(cuda.threadIdx.x != cuda.blockDim.x):
                if((p_id[cuda.threadIdx.x+1, 0] == p1[0]) and
                   (p_id[cuda.threadIdx.x+1, 1] == p1[1]) and
                   (p_offset[cuda.threadIdx.x+1] <= offset)):
                    point_consider = 0
            if(cuda.threadIdx.x != 0):
                if((p_id[cuda.threadIdx.x-1, 0] == p1[0]) and
                   (p_id[cuda.threadIdx.x-1, 1] == p1[1]) and
                   (p_offset[cuda.threadIdx.x-1] < offset)):
                    point_consider = 0

        small_dist = distance/1000
        h_corr = (math.sqrt(small_dist*small_dist + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000

        h1 = dsm[p1] - h_corr
        h_diff = h0 - h1
        tilt_land = (h_diff) / distance
        tilt_ant = (h_diff - tgt_elev) / distance
        if(tilt_ant <= max_tilt):

            if point_consider:
                out[p1] = 1
        if(tilt_land < max_tilt):
            max_tilt = tilt_land


@cuda.jit
def los_k(dsm, ordered_coordinates, out, width_resol, height_resol, poi_elev, tgt_elev):
    tid_x, tid_y = cuda.grid(2)
    if tid_x >= ordered_coordinates.shape[0] or \
       tid_y >= ordered_coordinates.shape[0] or \
       tid_x == tid_y:
        return
    x_step = types.float32(1.0)
    y_step = types.float32(1.0)
    poi = ordered_coordinates[tid_y]
    tgt = ordered_coordinates[tid_x]

    xs = tgt[0] - poi[0]
    ys = tgt[1] - poi[1]
    xs_a = abs(xs)
    ys_a = abs(ys)
    if(xs_a >= ys_a):
        # x dominant axis
        x_step = 1*math.copysign(1, xs)
        y_step = ys/xs_a
        dist = xs_a
    else:
        # y dominant axis
        y_step = 1*math.copysign(1, ys)
        x_step = xs/ys_a
        dist = ys_a

    h_poi = dsm[poi[0], poi[1]]
    h_tgt = dsm[tgt[0], tgt[1]]
    distance = math.sqrt(float(xs**2 + ys**2))
    h_ant_corr = (math.sqrt((distance/1000)**2 + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000
    tilt_ant = (-h_poi - poi_elev + h_tgt + tgt_elev) / distance
    for j in range(1, dist):
        x = x_step * j + poi[0]
        y = y_step * j + poi[1]
        xd = x-poi[0]
        yd = y-poi[1]

        # 2-d distance from poi
        d_poi = math.sqrt(xd*xd + yd*yd)
        # take point elevation
        p1 = (int(x), int(y))
        # calc earth radius correction
        small_dist = d_poi/1000
        h_corr = (math.sqrt(small_dist**2 + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000
        # height of LOS
        h_los = tilt_ant * d_poi + h_poi + poi_elev
        if dsm[p1] > h_los:
            return
    out[tid_y, tid_x] = 1


@cuda.jit
def dist_k(tgt_list, out, width_resol, height_resol):
    # threads are mapped to tgt_list which is an array of x, y, id
    poi = tgt_list[0]
    tid = cuda.grid(1)
    if tid >= tgt_list.shape[0] - 1:  # because in the list we also have poi in position 0
        return
    tgt = tgt_list[tid + 1]
    if poi[2] == tgt[2]:
        out[poi[2], tgt[2]] = 0
        return  # not testing self loop

    xs = tgt[0] - poi[0]
    ys = tgt[1] - poi[1]
    distance = math.sqrt(float(xs**2 + ys**2))
    out[poi[2], tgt[2]] = distance


@cuda.jit
def knife_k(dsm, tgt_list, out, angles_mat, width_resol, height_resol, poi_elev, tgt_elev, ple, lmb):
    # threads are mapped to tgt_list which is an array of x, y, id
    poi = tgt_list[0]
    tid = cuda.grid(1)
    if tid >= tgt_list.shape[0] - 1:  # because in the list we also have poi in position 0
        return
    tgt = tgt_list[tid + 1]
    if poi[2] == tgt[2]:
        out[poi[2], tgt[2]] = 0
        return  # not testing self loop

    x_step = types.float32(1.0)
    y_step = types.float32(1.0)
    xs = tgt[0] - poi[0]
    xs_a = abs(xs)
    ys = tgt[1] - poi[1]
    ys_a = abs(ys)
    if(xs_a >= ys_a):
        # x dominant axis
        x_step = 1*math.copysign(1, xs)
        y_step = ys/xs_a
        dist = xs_a
    else:
        # y dominant axis
        y_step = 1*math.copysign(1, ys)
        x_step = xs/ys_a
        dist = ys_a

    x = poi[0]
    y = poi[1]
    h_poi = dsm[poi[0], poi[1]]
    h_tgt = dsm[tgt[0], tgt[1]]
    distance = math.sqrt(float(xs**2 + ys**2))
    h_ant_corr = (math.sqrt((distance/1000)**2 + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000
    tilt_ant = -(h_poi + poi_elev - h_tgt - tgt_elev) / distance
    knife_max1 = 0
    d_poi_max1 = 0
    for i in range(1, dist):
        x = x_step * i + poi[0]
        y = y_step * i + poi[1]
        xd = x-poi[0]
        yd = y-poi[1]
        # 2-d distance from poi
        d_poi = math.sqrt(xd*xd + yd*yd)
        # take point elevation
        p1 = (int(x), int(y))
        # calc earth radius correction
        small_dist = d_poi/1000
        h_corr = (math.sqrt(small_dist**2 + EARTH_RADIUS_SQUARED) - EARTH_RADIUS)*1000
        # height of LOS
        h_los = tilt_ant * d_poi + h_poi + poi_elev - h_ant_corr
        # radius of fresnel zone
        r_fresnel = math.sqrt(d_poi * (distance - d_poi) / distance * lmb) * math.cos(tilt_ant)
        # calc height of knife inside fresnel zone
        knife = dsm[p1] - h_corr - h_los + r_fresnel
        if knife > knife_max1:
            knife_max1 = knife
            d_poi_max1 = d_poi

    knife1_loss = 0
    knife2_loss = 0
    if knife_max1 > 0.0:
        v = knife_max1 * math.sqrt(2 / lmb * (1 / d_poi_max1 + 1 / (distance - d_poi_max1)))
        knife1_loss = 6.9 + 10 * ple * math.log10(math.sqrt((v - 0.1)**2 + 1) + v - 0.1)

    fspl = 10 * ple * math.log10(4*math.pi * distance / lmb)
    val = int(fspl + knife1_loss + knife2_loss)
    out[poi[2], tgt[2]] = val
    angles_mat[poi[2], tgt[2], 0] = int((math.degrees(math.atan2(float(ys), float(xs))) + 180)*100)  # yaw/az
    angles_mat[poi[2], tgt[2], 1] = int((math.degrees(math.atan(tilt_ant)) + 180)*100)  # pitch/elev #multiply by 100 and use int16 to save memory
