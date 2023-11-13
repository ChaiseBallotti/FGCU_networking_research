from ultralytics import YOLO
import numpy
import csv

cat = "DDM"
spec = "5"
encode = "265"
res = "1080p"
video = cat + spec + "_" + encode + "_" + res + "_TCP_4000"

videoPath = "dataset/" + cat + "/" + video + ".mp4"
accPath = "runs/" + cat + "/" + cat + spec + "/" + video + "_acc.csv"
speedPath = "runs/" + cat +"/" + cat + spec + "/" + video + "_speed.csv"

model = YOLO("yolov8n.pt")

results = model(videoPath, stream=True)

accs = [[]]
spds = [[]]

for result in results:
    boxes = result.boxes
    confs = boxes.conf
    speeds = result.speed
 
    spd = []
    speed = speeds["inference"]
    speed = round(1000/speed, 2)
    spd.append(speed)
    spds.append(spd)
    
    acc=[]
    i = 0
    for conf in confs:
        if i > 5:
            break
        final = conf.item()
        final = round(final, 2)
        acc.append(final)
        i += 1
    accs.append(acc)

accs.pop(0)
spds.pop(0)

with open(accPath, 'w') as accFile, open(speedPath, 'w') as speedFile:
    awriter = csv.writer(accFile)
    awriter.writerows(accs)
    swriter = csv.writer(speedFile)
    swriter.writerows(spds)

#with open('start.csv', newline = '') as f:
#    reader = csv.reader(f)
#    for row in reader:
#        videoCat = row[0]
#        videoNum = row[1]
#        videoEnc = row[2]
#        videoRes = row[3]
#        dataOut(videoCat, videoNum, videoEnc, videoRes)
