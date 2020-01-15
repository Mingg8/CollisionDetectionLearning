
def distBtwPntTri(pnt, tri):
    P = pnt
    B = tri[0]
    E0 = tri[1] - tri[0]
    E1 = tri[2] - tri[0]

    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, B-P)
    e = np.dot(E1, B-P)
    f = np.dot(B-P, B-P)

    s = b*e - c*d
    t = b*d - a*e
    det = a*c - b*b
    if (s+t <= det):
        if (s < 0):
            if (t < 0):
                # region 4
                if (d < 0):
                    t = 0
                    if (-d >= a):
                        s = 1
                    else:
                        s = -d/a
                else:
                    s = 0
                    if ( e >= 0):
                        t = 0
                    elif (-e >= c):
                        t = 1
                    else:
                        t = -e / c
            else:
                # region 3
                s = 0
                if (e >= 0):
                    t = 0
                elif (-e >= c):
                    t = 1
                else:
                    t = -e / c
        elif (t < 0):
            # region 5
            t = 0
            if (d >= 0):
                s = 0
            elif (-d >= a):
                s = 1
            else:
                s = -d/a
        else:
            # region 0
            s /= det
            t /= det
    else:
        if (s < 0):
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if (tmp1 > tmp0):
                numer = tmp1 - tmp0
                denom = a - 2*b + c
                if (numer > denom):
                    s = 1
                else:
                    s = numer / denom
                t = 1 - s
            else:
                s = 0
                if (tmp1 < 0):
                    t = 1
                elif (e >= 0):
                    t = 0
                else:
                    t = -e / c
        elif ( t < 0):
            # region 6
            tmp0 = b + e
            tmp1 = a + d
            if (tmp1 > tmp0):
                numer = tmp1 - tmp0
                denom = a - 2*b + c
                if (numer >= denom):
                    t = 1
                else:
                    t = numer / denom
                s = 1 - t
            else:
                t = 0
                if (tmp1 <= 0):
                    s = 1
                elif (d >= 0):
                    s = 0
                else:
                    s = -d/a
        else:
            # region 1
            numer = (c+e) - (b+d)
            if (numer <= 0):
                s = 0
            else:
                denom = a - 2*b + c
                if (numer >= denom):
                    s = 1
                else:
                    s = numer / denom
            t = 1 - s


def rayIntersectsTriangle(ray_origin, ray_vec, tri):
    EPS = 0.000000001
    tri = np.array(tri).astype(float)
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    h = np.cross(ray_vec, e2)
    a = np.dot(e1, h)
    if (a > -EPS and a < EPS):
        return 0
    f = 1.0 / a
    s = ray_origin - tri[0]
    u = f * np.dot(s, h)
    if (u < 0.0 or u > 1.0):
        return 0
    q = np.cross(s, e1)
    v = f * np.dot(ray_vec, q)
    if (v < 0.0 or u+v>1.0):
        return 0
    t = f * np.dot(e2, q)
    if (t > EPS and t < (1-EPS)):
        pnt = ray_origin + ray_vec * t
        return pnt
    else:
        return 0

def isPointInMesh(pnt, obj, vec):
    count = 0
    EPS = 0.00001
    prev_is_intersect = np.array([0, 0, 0])
    for l in range(len(obj.faces)):
        a = [obj.vertices[ll] for ll in obj.faces[l]]
        is_intersect = rayIntersectsTriangle(pnt, vec, a)
        if (is_intersect is not 0) and \
            (np.linalg.norm(prev_is_intersect - is_intersect) > EPS):
            # the ray intersects with triangle
            prev_is_intersect = is_intersect
            print("isintersect: {}".format(is_intersect))
            count += 1
        # if count == 2:
        #     print("count: {}".format(count))
        #     return 1
    print("origin: {}\n".format(pnt))
    # print("count: {}".format(count))
    if count == 1:
        penet = -np.linalg.norm(is_intersect - pnt)
        return penet
    return 1