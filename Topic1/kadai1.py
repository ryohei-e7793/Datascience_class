from math import radians, cos, sin, asin, sqrt
import numpy as np

all3file = open("../class-master_2/all3.csv", "r")
all3line = all3file.read()
all3line = [[format(s1)for s1 in s0.split(",")]for s0 in all3line.strip().split("\n")]

jushofile = open("../class-master_2/yubin_jusho.tsv", "r")
jusholine = jushofile.read()
jusholine = [[format(s1)for s1 in s0.split("\t")]for s0 in jusholine.strip().split("\n")]

outfile = open("/amedastiten_timei.txt", "w")

dis = []

def haversine(lon1, lat1, lon2, lat2):

    """
    
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    reference : https://www.it-swarm-ja.tech/ja/python/2%e3%81%a4%e3%81%ae%ef%bc%88%e7%b7%af%e5%ba%a6%e3%80%81%e7%b5%8c%e5%ba%a6%ef%bc%89%e3%83%9d%e3%82%a4%e3%83%b3%e3%83%88%e9%96%93%e3%81%ae%e8%b7%9d%e9%9b%a2%e3%82%92%e3%81%99%e3%81%b0%e3%82%84%e3%81%8f%e6%8e%a8%e5%ae%9a%e3%81%99%e3%82%8b%e3%81%ab%e3%81%af%e3%81%a9%e3%81%86%e3%81%99%e3%82%8c%e3%81%b0%e3%82%88%e3%81%84%e3%81%a7%e3%81%99%e3%81%8b%ef%bc%9f/1072488907/

    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

#print(haversine(float(all3line[1][4]), float(all3line[1][3]), float(jusholine[1][3]), float(jusholine[1][2])))

for i in range(1, len(all3line)):
    for j in range(0, len(jusholine)):
        dis.append(haversine(float(all3line[i][4]), float(all3line[i][3]), float(jusholine[j][3]), float(jusholine[j][2])))
        index_min = dis.index(min(dis))
        outfile.write(str(jusholine[index_min][0]))
        outfile.write("\t")
        outfile.write(str(jusholine[index_min][1]))
        outfile.write("\n")
        
