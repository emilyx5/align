import numpy as np
from pathlib import Path
import imageio.v3 as iio
import skimage.color
from skimage.draw import line, line_nd ,disk, set_color
import cv2
import skimage.filters
import matplotlib.pyplot as plt

def process(img, min, max):

    gray = skimage.color.rgb2gray(img)
    bg = skimage.filters.threshold_otsu(gray)
    mask = gray < bg

    new_img , count = skimage.measure.label(mask, 1, return_num=True)

    obj_features =  skimage.measure.regionprops(new_img)
    for object_id, objf in enumerate(obj_features, start=1):
      if objf["area"] > max or objf["area"] < min :
       new_img[new_img == objf["label"]] = 0
            

    color_new_img = skimage.color.label2rgb(new_img, bg_label=0)    
    return color_new_img

def slice(img):
    HEIGHT, WIDTH, CHANNELS = img.shape
    TILE_H = int(HEIGHT / 3) 
    TILE_W = int(WIDTH / 2)
    min = 4000
    max = 2000*30

    sliced = img.reshape(HEIGHT // TILE_H, TILE_H, WIDTH // TILE_W, TILE_W, CHANNELS )
    sliced = sliced.swapaxes(1,2)
    
    #retrieve tiles with alignment marks and clear any residual small bits by processing again
    topL = sliced[0,0, :TILE_H, :TILE_W-800, :]
    topL = process(topL, min, max)
   
    topR = sliced[0,1, :TILE_H, :TILE_W-500, :]
    topR = process(topR, min, max)

    botL = sliced[2,0, :TILE_H, :TILE_W-500, :]
    botL = process(botL, min, max)
    
    botR = sliced[2,1, :TILE_H, :TILE_W-500, :]
    botR = process(botR, min, max)

    tL_coords = np.ndarray.nonzero(topL)
    tR_coords = np.ndarray.nonzero(topR)
    bL_coords = np.ndarray.nonzero(botL)
    bR_coords = np.ndarray.nonzero(botR)
    
    coords = list()
    coords.append(tL_coords)
    coords.append(tR_coords)
    coords.append(bL_coords) 
    coords.append(bR_coords)

    tiles = list()
    tiles.append(topL)
    tiles.append(topR)
    tiles.append(botL)
    tiles.append(botR)

    bad = -1
    # find quadrant of any 'bad' alignment mark
    q = 1
    for c in coords:
        if c[0].size == 0:
            bad = q
        q+=1

    return coords, bad, tiles

def findPoint(section):
    coords = np.ndarray.nonzero(section)
    xCoords = coords[1]
    yCoords = coords[0]
    w = 20
    count = 0
    avg = 0
    xPts = list()
    yPts = list()
    lastPt = xCoords[0]


    # sum up each row of pixels

    for pt in xCoords:
        if count == 0:
            avg = pt
        count+= 1
        if pt < lastPt:
            avg = lastPt + avg
            avg /= 2
            xPts.append(avg)
            count = 0
            avg = 0
        lastPt = pt
    
    # sum up the avgs into a single x coordinate

    avg = 0
    count = 0
    for pt in xPts:
        avg += pt
        count +=1

    if count != 0:        
        x = avg / count

    else:
        x = w/2

    # reset vals and repeat for y
    avg = 0
    count = 0
    lastPt = yCoords[0]

    for pt in yCoords:
        if count == 0:
            avg = pt
        count+= 1
        if pt < lastPt:
            avg = lastPt + avg
            avg /= 2
            yPts.append(avg)
            count = 0
            avg = 0
    
    
    avg = 0
    count = 0
    for pt in yPts:
            avg += pt
            count +=1

    if count != 0:    
        y = avg / count
    else:
        y = w / 2

    return int(x), int(y)

def splitCoords(axisCoords, quadrant):
    
    xCoords = axisCoords[1]
    yCoords = axisCoords[0]

    xMax = xCoords.max() 
    xMin = xCoords.min()
    yMax = yCoords.max()
    yMin = yCoords.min()

    l = 150
    w = 20

    # split the quadrant into 4 parts, each containing a subsection of the cross to be used for analysis

    top = quadrant[yMin + 40: yMin + 40 + w  + 1, xMin + l: xMax - l  + 1,  :]
    bottom = quadrant[ yMax - 40 - w: yMax - 40 + 1 , xMin + l: xMax - l + 1,:]
    left = quadrant[ yMin + l: yMax - l + 1, xMin + 40: xMin + 40 + w + 1 , :]
    right = quadrant[ yMin + l: yMax - l + 1,  xMax - 40 - w: xMax - 40 + 1 , :]

    return top, bottom, left, right

def slope(x1, y1, x2, y2):
    vert = False
    if (x2 - x1) == 0 :
        vert  = True
        return 1 , x1, vert 
    m = (y2 - y1)/ (x2 - x1)
    c = y1 - m * x1
    return m,c, vert

def intersection(m1, c1, m2, c2, vert):
    if vert:
        x = c1
        y = m2 * x + c2
    else:
        x = (c2 - c1)/ (m1 - m2)
        y = m1 * x + c1
    return int(x) , int(y)
    

def draw(coords, name, bad, tiles):    
    img = iio.imread(uri = name)

    HEIGHT, WIDTH, CHANNELS = img.shape
    TILE_H = int(HEIGHT / 3) 
    TILE_W = int(WIDTH / 2)

    x2 = 0
    x3 = 0

    # define mirror lines for each 'bad' quadrant case
    if bad == 1:
        tiles.pop(0)
        x2 = 2
        x3 = 3
    elif bad == 2:
        tiles.pop(1)
        x2 = 3
        x3 = 4
    elif bad == 3:
        tiles.pop(2)
        x2 = 1
        x3 = 4
    elif bad == 4:
        tiles.pop(3)
        x2 = 2
        x3 = 3
        
    quad = 1
    for c in coords:

        adj_x = 0
        adj_y = 0 

        if quad != bad:
            # adjust position from (0,0) for diff quadrants
            if quad == 2:
                adj_x = 1*TILE_W
            
            elif quad == 3:
                adj_y = 2*TILE_H
            
            elif quad == 4:
                adj_x = 1*TILE_W
                adj_y = 2*TILE_H

            tile = tiles.pop(0)
            t,b,l,r = splitCoords(c, tile)
            
            tX, tY = findPoint(t)
            bX, bY = findPoint(b)
            lX, lY = findPoint(l)
            rX, rY = findPoint(r)

            # adjustments
            xCoords = c[1]
            yCoords = c[0]

            xMax = xCoords.max() 
            xMin = xCoords.min()
            yMax = yCoords.max()
            yMin = yCoords.min()

            tX += xMin + 150  + adj_x
            tY += yMin + 40 + adj_y
            bX += xMin + 150  + adj_x
            bY += yMax - 60 + adj_y
            lX += xMin + 40  + adj_x
            lY += yMin + 150  + adj_y
            rX += xMax - 60  + adj_x
            rY += yMin + 150  + adj_y

            mark1,mark2 = line(tY, tX, bY, bX)
            set_color(img, (mark1 , mark2), [255,0,0])
            
            mark3,mark4 = line(lY, lX, rY, rX)
            set_color(img, (mark3 , mark4), [255,0,0])     
           

            
            m1, c1,vert = slope(tX, tY , bX , bY )
            m2, c2,none = slope(lX, lY, rX, rY)

            xf,yf = intersection(m1, c1, m2, c2, vert)


            if quad == x2:
                x2 = xf
                y2 = yf

            elif quad == x3:
                x3 = xf
                y3 = yf

            else:
                x1 = xf
                y1 = yf

            mark1,mark2 = disk((yf ,xf), 0.5, shape= None)
            set_color(img, (mark1 , mark2), [255,0,0])
        
        quad+=1
    
    # mirror point if there is 'bad' quadrant
    if bad != -1:
        m = (y3 - y2)/(x3 - x2)
        c = (x3*y2 - x2*y3)/(x3-x2)
        d = (x1 + (y1 - c)*m)/(1 + m**2)

        x4 = 2*d - x1
        y4 = 2*d*m - y1 + 2*c 

        mark1,mark2 = disk((y4,x4), 0.5, shape= None)
        set_color(img, (mark1 , mark2), [255,0,0])
    
    return img


def run(name):
    img = iio.imread(uri = name)
    img_R = img * [1,0,0]
    img_R[img_R > 128] = 255


    coordinates, bad, tiles = slice(img_R)
    new_img = draw(coordinates, name, bad, tiles)

    new_name = list(name)
    new_name.insert(len(new_name) - 4, '_marked')
    new_name = "".join(new_name)

    iio.imwrite(uri= new_name, image=new_img)
    plt.imshow(new_img)
    plt.show()  

