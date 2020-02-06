import cv2
# data scheme: (x_start, y_start, x_stop, y_stop)
OFFSETS = [[ (   0,   0,   1,   1), (-0.5,   0, 0.5,   1), (  -1,   0,   0,   1) ],
           [ (   0,-0.5,   1, 0.5), (-0.5,-0.5, 0.5, 0.5), (  -1,-0.5,   0, 0.5) ],
           [ (   0,  -1,   1,   0), (-0.5,  -1, 0.5,   0), (  -1,  -1,   0,   0) ]]

def get_offset(x, y, x_cnt, y_cnt):
    # corners
    if x == 0 and y == 0:
        return OFFSETS[0][0]
    if x == x_cnt-1 and y == 0:
        return OFFSETS[0][-1]
    if x == 0 and y == y_cnt-1:
        return OFFSETS[-1][0]
    if x == x_cnt-1 and y == y_cnt-1:
        return OFFSETS[-1][-1]

    # corner and into range
    if 0 < x < x_cnt-1 and y == 0:
        return OFFSETS[0][1]
    if 0 < x < x_cnt-1 and y == y_cnt-1:
        return OFFSETS[-1][1]
    if x == 0 and 0 < y < y_cnt-1:
        return OFFSETS[1][0]
    if x == x_cnt-1 and 0 < y < y_cnt-1:
        return OFFSETS[1][-1]

    # center
    else:
        return OFFSETS[1][1]

def split_img(img, x_cnt, y_cnt, offset_size=100):
    cp_img = img.copy()
    sp_imgs = []
    points = []
    org_h, org_w, c = img.shape
    h_step = org_h // y_cnt
    w_step = org_w // x_cnt
    for y in range(y_cnt):
        for x in range(x_cnt):
            offsets = get_offset(x, y, x_cnt, y_cnt)
            start_x = int((x*w_step)     + (offsets[0] * offset_size))
            start_y = int((y*h_step)     + (offsets[1] * offset_size))
            stop_x  = int(((x+1)*w_step) + (offsets[2] * offset_size))
            stop_y  = int(((y+1)*h_step) + (offsets[3] * offset_size))
            sp_img = cp_img[start_y:stop_y, start_x:stop_x, :]
            sp_imgs.append(sp_img)
            points.append([start_x, start_y, stop_x, stop_y])

    return sp_imgs, points

def concat_img(sp_imgs, x_cnt, y_cnt):
	h_imgs = []
	for y in range(y_cnt):
	    idxs = [ i + y*y_cnt for i in range(x_cnt) ]
	    imgs = [dst_imgs[i] for i in idxs]
	    h_imgs.append(cv2.hconcat(imgs))
	concat_img = cv2.vconcat(h_imgs)
	return concat_img


if __name__ == '__main__':
    img = cv2.imread('/home/yamauchi/Downloads/titanfall2.jpg')
    x_cnt = 2
    y_cnt = 4
    print('original shape', img.shape)
    sp_imgs = split_img(img, x_cnt, y_cnt, offset_size=20)
    for idx, s in enumerate(sp_imgs):
        print('img_{} shape : '.format(idx), s.shape)
