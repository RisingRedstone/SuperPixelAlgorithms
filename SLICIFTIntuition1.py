# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:13:32 2024

@author: prath
"""

import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import json

img = cv2.cvtColor(cv2.imread('Example Photos/portrait-young-woman-with-natural-make-up_23-2149084942.jpg', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#img = cv2.imread('Example Photos/099b331721c6a2a6ed8859d51a5ff091.jpg', cv2.IMREAD_GRAYSCALE)

imgS = np.zeros(img.shape, dtype = np.uint8)
imgUpdated = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
imgMask = np.ones((img.shape[0], img.shape[1]), dtype = np.float32) * 1e10
imgIDs = np.ones((img.shape[0], img.shape[1]), dtype = np.int16) * -1
imgDiffs = np.zeros((img.shape[0], img.shape[1], 2), dtype = np.float32)

def ColourDistance(c1, c2):
    c1 = c1.astype(np.float16)
    c2 = c2.astype(np.float16)
    dR = (c1[0] - c2[0]) * (c1[0] - c2[0])
    dG = (c1[1] - c2[1]) * (c1[1] - c2[1])
    dB = (c1[2] - c2[2]) * (c1[2] - c2[2])
    hr = (c1[0] + c2[0]) / (2 * 256)
    return ((2 + hr) * dR) + (4 * dG) + ((2 + (255/256) - hr) * dB)

def PositionDistance(P1, P2):
    return (P1[0] - P2[0]) * (P1[0] - P2[0]) + (P1[1] - P2[1]) * (P1[1] - P2[1])

def AddtoAvgCol(avgC, C1, N):
    C1 = C1.astype(np.float64)
    return ((avgC * N)/(N+1)) + (C1  / (N+1))

def AddtoAvgPos(avgP, P1, N):
    return (((avgP[0] * N) + P1[0] ) / (N+1), ((avgP[1] * N) + P1[1] ) / (N+1))

def RemFromAvgCol(avgC, C1, N):
    return (avgC - (C1 / N)) * (N/(N-1))
    
def RemFromAvgPos(avgP, P1, N):
    return (((avgP[0] - (P1[0] / N)) * (N/(N-1))), ((avgP[1] - (P1[1] / N)) * (N/(N-1))))

def AddBoundaryPoints(pointAdded, IUpdated, IIDs, bound, RegionID):
    lis = []
    assert(IUpdated[pointAdded] != 0)
    temp0 = (pointAdded[0]-1, pointAdded[1])
    ID = RegionID
    if(temp0[0] >= bound[0][0]):
        if(IUpdated[temp0] < 2):
            lis.append(temp0)
            IUpdated[temp0] += 2
    temp0 = (pointAdded[0]+1, pointAdded[1])
    if(temp0[0] < bound[0][1]):
        if(IUpdated[temp0] < 2):
            lis.append(temp0)
            IUpdated[temp0] += 2
    temp0 = (pointAdded[0], pointAdded[1]-1)
    if(temp0[1] >= bound[1][0]):
        if(IUpdated[temp0] < 2):
            lis.append(temp0)
            IUpdated[temp0] += 2
    temp0 = (pointAdded[0], pointAdded[1]+1)
    if(temp0[1] < bound[1][1]):
        if(IUpdated[temp0] < 2):
            lis.append(temp0)
            IUpdated[temp0] += 2
    return lis

def NearbyShortestDistance(point, IMask, IIDs, IUpdated, ICol, IDiffs, RegionID):
    lowP = (-1, -1)
    lowest = 1e10
    ID = RegionID
    temp0 = (point[0]-1, point[1])
    if(temp0[0] >= 0 and IIDs[temp0] == ID and IUpdated[temp0] == 3):
        dis = IMask[temp0] + IDiffs[temp0][0]
        if(dis < lowest):
            lowest = dis
            lowP = temp0
    temp0 = (point[0]+1, point[1])
    if(temp0[0] < IMask.shape[0] and IIDs[temp0] == ID and IUpdated[temp0] == 3):
        dis = IMask[temp0] + IDiffs[point][0]
        if(dis < lowest):
            lowest = dis
            lowP = temp0
    temp0 = (point[0], point[1]-1)
    if(temp0[1] >= 0 and IIDs[temp0] == ID and IUpdated[temp0] == 3):
        dis = IMask[temp0] + IDiffs[temp0][1]
        if(dis < lowest):
            lowest = dis
            lowP = temp0
    temp0 = (point[0], point[1]+1)
    if(temp0[1] < IMask.shape[1] and IIDs[temp0] == ID and IUpdated[temp0] == 3):
        dis = IMask[temp0] + IDiffs[point][1]
        if(dis < lowest):
            lowest = dis
            lowP = temp0
    return (lowP, lowest)

def PotFunction(clusters, IMask):
    SampleSize = 500
    IMGMASKPOTVAL = 1 #Set value that attracts clusters to places with high img mask values
    #CLUSTERREPELVAL = #Set value that repels clusters away from each other

    #I feel like the big size is already compensated for by the 
    ILogMask = np.log(np.maximum(1e-2, IMask))
    for k in range(len(clusters)):
        #HIGH IMG Mask Values Potential
        #approximate with nearby samples
        C = clusters[k]
        clusterLoc = np.resize(clusters[k][0], (1, 2))
        Bounds = [[int(max(0, C[0][0]-(2 * Sx))), int(min(img.shape[0], C[0][0]+(2 * Sx)))], 
                 [int(max(0, C[0][1]-(2 * Sy))), int(min(img.shape[1], C[0][1]+(2 * Sy)))]]
        SampX = np.random.randint(Bounds[0][0], Bounds[0][1], (SampleSize))
        SampY = np.random.randint(Bounds[1][0], Bounds[1][1], (SampleSize))
        SampPoints = [(SampX[k], SampY[k]) for k in range(SampleSize)]
        RelSampPoints =(SampPoints - clusterLoc)
        SampPointVals = [ np.maximum(1e-10, ILogMask[S]) for S in SampPoints]
        SampPointValsSum = np.sum(np.abs(SampPointVals))
        SampPointVals = np.matmul(np.resize(SampPointVals, (SampleSize, 1)), np.ones((1, 2)))
        SampPointWeighted = IMGMASKPOTVAL * np.sum(RelSampPoints * SampPointVals, axis = 0) / SampPointValsSum
        #print(k, clusters[k][0], SampPointWeighted, np.linalg.norm(SampPointWeighted))
        NewPoint = (int(clusters[k][0][0] + SampPointWeighted[0]), int(clusters[k][0][1] + SampPointWeighted[1]))
        NewPoint = (np.clip(NewPoint[0], Bounds[0][0], Bounds[0][1]-1), np.clip(NewPoint[1], Bounds[1][0], Bounds[1][1] - 1))
        clusters[k] = (NewPoint, clusters[k][1], clusters[k][2])
        
    return clusters

def MarkCircsGrayScale(Image, Points):
    for P in Points:
        point = (int(P[0]), int(P[1]))
        cv2.circle(Image, (point[1], point[0]), 2, [255, 0, 0], 1)
    return Image
        


#Here you fill the ImageDiffs array to have consequentcolour difference of pixels

#fill x diff and y diff
for x in range(img.shape[0] - 1):
    for y in range(img.shape[1] -1):
        imgDiffs[x, y, 0] = ColourDistance(img[x, y], img[x + 1, y]) / 760
        imgDiffs[x, y, 1] = ColourDistance(img[x, y], img[x, y + 1]) / 760

for x in range(img.shape[0] - 1):
    imgDiffs[x,  img.shape[1]-1, 0] = imgDiffs[x,  img.shape[1]-2, 0]
    imgDiffs[x,  img.shape[1]-1, 1] = imgDiffs[x,  img.shape[1]-2, 1]

for y in range(img.shape[1] - 1):
    imgDiffs[img.shape[0] - 1, y, 0] = imgDiffs[img.shape[0] - 2, y, 0]
    imgDiffs[img.shape[0] - 1, y, 1] = imgDiffs[img.shape[0] - 2, y, 1]

imgDiffs[img.shape[0] - 1, img.shape[1] - 1, 0] = imgDiffs[img.shape[0] - 2, img.shape[1] - 2, 0]
imgDiffs[img.shape[0] - 1, img.shape[1] - 1, 1] = imgDiffs[img.shape[0] - 2, img.shape[1] - 2, 1]

Kx = 30
Ky = 20
#THRESHOLD = (Set threshold to a low value, for now i will just repeat steps until a low value is reached)

Sx = (img.shape[0] / (Kx * 2))
Sy = (img.shape[1] / (Ky * 2))

#Make clusters
Clusts = []
pos = (0, 0)
for x in range(Kx):
    pos  = (pos[0] + Sx, 0)
    for y in range(Ky):
        pos  = (pos[0], pos[1] + Sy)
        Approxpos = (int(pos[0]), int(pos[1]))
        Clusts.append((Approxpos, img[Approxpos], 1))
        pos  = (pos[0], pos[1] + Sy)
    pos  = (pos[0] + Sx, 0)
    
    
ITERS = 6
for iterer in range(ITERS):

    for k in range(len(Clusts)):
        C = Clusts[k]
        imgIDs[C[0]] = k
        imgUpdated[C[0]] = 3
        imgMask[C[0]] = 0 * ColourDistance(C[1], img[C[0]])
        Bounds = [[int(max(0, C[0][0]-(2 * Sx))), int(min(img.shape[0], C[0][0]+(2 * Sx)))], 
                 [int(max(0, C[0][1]-(2 * Sy))), int(min(img.shape[1], C[0][1]+(2 * Sy)))]]
        Boundary = AddBoundaryPoints(C[0], imgUpdated, imgIDs, Bounds, k)
        deBound = Boundary
        
        BSizes = 0
        while(len(Boundary) > 0):
            BSizes +=1
            bPoint = Boundary[0]
            #print(bPoint)
            checker, dis = NearbyShortestDistance(bPoint, imgMask, imgIDs, imgUpdated, img, imgDiffs, k)
            assert(checker[0] != -1)
            
            
            if((imgUpdated[bPoint] == 3 and dis < imgMask[bPoint]) or (imgUpdated[bPoint] == 2)):
                imgIDs[bPoint] = k
                imgMask[bPoint] = dis
                imgUpdated[bPoint] = 3
                addBounds = AddBoundaryPoints(bPoint, imgUpdated, imgIDs, Bounds, k)
                Boundary += addBounds
                deBound += addBounds
            
            #print(bPoint, imgIDs[bPoint], imgUpdated[bPoint], k)
            Boundary = Boundary[1:]
        
        for P in deBound:
            if(imgUpdated[P] > 2):
                imgUpdated[P] -= 2
        
        print(Clusts[k], len(deBound))
        Clusts[k] = (Clusts[k][0], Clusts[k][1], 0)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            pos = (x, y)
            imgUpdated[pos] = 0
            KVal = imgIDs[pos]
            if(KVal == -1):
                continue
            Clusts[KVal] = (AddtoAvgPos(Clusts[KVal][0], pos, Clusts[KVal][2]),
                            AddtoAvgCol(Clusts[KVal][1], img[pos], Clusts[KVal][2]),
                            Clusts[KVal][2] + 1)
    
    cv2.imshow("FrameMask", MarkCircsGrayScale(np.log(np.maximum(1e-2, imgMask)), [kiter[0] for kiter in Clusts]))
    cv2.imshow("FrameUpdated", MarkCircsGrayScale(255*imgUpdated, [kiter[0] for kiter in Clusts]))
    if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    
    if(iterer % 5 == 1):
        Clusts = PotFunction(Clusts, imgMask)
        imgIDs *= 0
        imgIDs -= 1
    
    for k in range(len(Clusts)):
        Clusts[k] = ((int(Clusts[k][0][0]), int(Clusts[k][0][1])), Clusts[k][1], 0)
    
cv2.destroyAllWindows()
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        pos = (x, y)
        imgS[pos] = Clusts[imgIDs[pos]][1]


cv2.destroyAllWindows()
fig = plt.figure(figsize=(8, 7)) 
rows = 2
columns = 2
fig.add_subplot(rows, columns, 1) 
plt.imshow(img)
plt.title("Original")
fig.add_subplot(rows, columns, 2) 
plt.imshow(MarkCircsGrayScale(imgS, [C[0] for C in Clusts]))
plt.title("SuperPixel")
fig.add_subplot(rows, columns, 3) 
plt.imshow(np.log(np.maximum(1e-2, imgMask)))
plt.title("Mask")
fig.add_subplot(rows, columns, 4) 
plt.imshow(imgIDs)
plt.title("IDs")

cv2.imwrite('Example Photos/Fin5W.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite('Example Photos/Fin5SLICINT1.jpg', cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR))