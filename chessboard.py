#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:07:29 2024

@author: sabo4ever
"""

#%%
# coding:utf-8
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


#%%
corners115 = np.array([[798,304],[842,304],[880,305],[919,306],[963,307],[1002,310],[1045,314],[1088,319],[1134,320],[1167,324],[1195,326],
[790,336],[841,333],[884,333],[926,333],[971,334],[1018,336],[1066,337],[1110,340],[1155,344],[1193,345],[1231,348],
[783,370],[839,371],[888,370],[934,371],[983,371],[1035,371],[1090,371],[1144,373],[1188,373],[1239,376],[1278,377],
[780,404],[836,404],[891,405],[943,405],[994,405],[1051,403],[1109,400],[1167,398],[1214,397],[1261,395],[1295,393],
[774,442],[834,442],[892,442],[950,441],[1004,438],[1067,434],[1129,431],[1188,424],[1239,421],[1287,417],[1318,412],
[767,473],[832,474],[896,474],[956,474],[1014,473],[1082,471],[1150,464],[1214,458],[1268,455],[1318,445],[1351,443],
[758,517],[833,516],[899,517],[964,517],[1026,514],[1097,507],[1171,499],[1233,489],[1285,478],[1336,468],[1372,458],
[746,564],[833,564],[902,564],[970,562],[1037,558],[1111,548],[1191,536],[1254,524],[1308,512],[1360,497],[1394,483]]).reshape(-1, 2)

#%%

corners116 = np.array([[204,322],[227,310],[259,296],[290,284],[326,273],[361,261],[418,242],[482,229],[540,218],[627,205],[707,198],
[212,343],[237,337],[272,326],[305,317],[340,310],[378,302],[433,288],[494,276],[552,267],[633,260],[709,258],
[218,360],[242,353],[278,345],[313,338],[350,333],[388,326],[441,320],[504,315],[563,311],[641,307],[714,305],
[224,376],[250,369],[286,365],[323,363],[358,362],[398,361],[454,360],[517,360],[573,358],[646,355],[716,354],
[234,399],[261,397],[300,398],[336,398],[374,401],[420,402],[467,403],[529,404],[585,407],[650,409],[718,413],
[243,418],[271,420],[311,424],[351,428],[388,433],[434,437],[483,442],[539,446],[594,449],[657,454],[721,457],
[251,434],[281,441],[322,449],[364,458],[403,466],[448,474],[498,483],[549,491],[604,502],[664,513],[725,519],
[260,463],[298,473],[337,485],[380,499],[423,511],[470,525],[515,537],[560,546],[615,557],[669,563],[730,573]]).reshape(-1, 2)

#%%

corners117 = np.array([[489,337],[502,336],[515,335],[537,334],[556,334],[584,334],[606,334],[633,333],[655,334],[676,333],[699,333],
[467,346],[483,346],[500,346],[519,346],[542,346],[571,346],[592,346],[629,346],[641,346],[665,346],[690,346],
[433,361],[457,361],[477,361],[499,360],[523,361],[550,362],[573,363],[600,364],[625,364],[651,365],[679,366],
[404,376],[429,377],[446,379],[473,380],[500,381],[528,382],[555,383],[581,385],[607,386],[635,387],[668,387],
[372,392],[397,394],[418,397],[447,399],[476,401],[505,403],[535,404],[562,406],[588,407],[618,408],[652,410],
[347,407],[375,409],[400,410],[431,413],[460,414],[488,417],[518,419],[548,421],[575,423],[606,423],[642,425],
[327,421],[348,423],[377,426],[410,428],[441,430],[472,432],[502,435],[534,437],[563,439],[596,441],[634,443],
[305,434],[328,438],[357,442],[389,445],[421,449],[453,453],[482,458],[516,461],[549,462],[581,467],[623,472]]).reshape(-1, 2)


#%%
corners118 = np.array([[545,186],[561,182],[573,180],[587,177],[605,175],[620,174],[641,172],[664,171],[682,170],[704,169],[728,168],
[549,201],[563,198],[576,196],[589,194],[606,192],[622,191],[623,189],[665,188],[683,188],[704,188],[728,188],
[552,215],[565,213],[579,212],[591,210],[607,208],[625,206],[644,205],[665,203],[685,201],[706,201],[729,200],
[556,233],[569,231],[581,229],[595,229],[611,228],[628,227],[647,224],[668,223],[687,222],[708,221],[731,219],
[559,249],[572,248],[586,246],[600,244],[613,244],[630,242],[648,240],[668,240],[688,239],[709,237],[732,235],
[564,268],[577,267],[589,265],[601,263],[617,262],[634,260],[651,260],[671,258],[689,257],[709,258],[734,256],
[568,288],[580,286],[593,284],[605,283],[620,282],[637,280],[654,280],[673,279],[691,278],[711,278],[735,277],
[570,310],[585,309],[599,308],[609,307],[625,305],[642,303],[658,303],[674,303],[693,303],[713,303],[738,303]]).reshape(-1, 2)

#%%
corners119 = np.array([[190,417],[210,419],[231,421],[253,424],[289,429],[328,433],[361,437],[399,442],[445,446],[516,452],[574,458],
[178,419],[193,423],[214,427],[233,432],[257,439],[293,446],[328,453],[369,462],[412,471],[472,484],[532,498],
[162,424],[175,429],[194,435],[214,441],[233,448],[262,457],[294,468],[338,479],[377,490],[440,507],[494,525],
[149,429],[157,436],[172,444],[187,449],[207,457],[230,467],[267,481],[307,495],[348,509],[409,531],[464,549],
[133,436],[140,442],[152,451],[164,458],[183,466],[207,478],[240,493],[276,509],[317,527],[380,553],[437,572],
[111,444],[118,451],[128,459],[140,468],[155,479],[173,491],[204,510],[236,530],[272,551],[337,583],[396,604],
[97,451],[103,457],[111,466],[121,475],[134,486],[150,502],[177,523],[209,545],[241,567],[300,604],[355,631],
[82,457],[87,463],[94,472],[100,482],[113,498],[127,510],[152,535],[183,561],[209,583],[266,627],[312,658]]).reshape(-1, 2)

#%%
corners120 = np.array([[791,421],[845,423],[894,423],[959,424],[1007,426],[1071,425],[1122,425],[1166,425],[1204,421],[1242,420],[1267,417],
[786,448],[845,449],[896,452],[961,452],[1011,451],[1081,449],[1133,445],[1180,442],[1216,439],[1256,438],[1283,434],
[779,491],[842,493],[899,494],[972,495],[1026,491],[1101,488],[1156,482],[1201,475],[1237,469],[1274,460],[1304,453],
[773,524],[841,526],[905,527],[980,527],[1041,523],[1116,517],[1177,508],[1220,501],[1260,493],[1298,483],[1331,474],
[758,566],[839,569],[918,570],[994,569],[1072,565],[1147,555],[1210,545],[1253,535],[1292,524],[1332,512],[1364,499],
[743,617],[839,619],[932,619],[1019,616],[1105,607],[1177,589],[1237,571],[1281,557],[1320,545],[1359,531],[1392,518],
[724,685],[841,689],[954,682],[1046,672],[1145,652],[1217,630],[1282,607],[1321,590],[1361,574],[1398,557],[1428,542],
[696,759],[844,757],[975,750],[1082,733],[1196,708],[1276,677],[1337,646],[1373,627],[1411,604],[1442,584],[1470,564]]).reshape(-1, 2)
#%%
corners121 = np.array([[592,211],[675,201],[742,200],[827,199],[906,207],[996,211],[1084,226],[1164,243],[1231,255],[1300,275],[1360,291],
[603,264],[682,258],[747,256],[827,256],[902,259],[988,263],[1075,273],[1151,283],[1220,298],[1285,310],[1344,327],
[610,308],[689,307],[752,306],[827,306],[900,307],[984,312],[1065,318],[1138,325],[1211,332],[1273,342],[1336,349],
[619,365],[692,368],[757,369],[828,370],[898,370],[980,372],[1054,377],[1126,380],[1194,385],[1261,389],[1321,388],
[627,431],[694,432],[760,436],[829,439],[897,439],[976,439],[1045,440],[1115,440],[1179,437],[1247,432],[1307,425],
[629,488],[699,492],[763,499],[831,501],[898,501],[972,502],[1040,500],[1106,499],[1171,493],[1228,483],[1285,479],
[635,549],[704,554],[767,557],[834,558],[898,559],[971,557],[1036,555],[1103,549],[1167,544],[1218,538],[1264,531],
[640,607],[709,611],[771,616],[837,621],[906,622],[973,620],[1037,617],[1103,604],[1169,591],[1214,579],[1255,567]]).reshape(-1, 2)
 #%%
corners122 = np.array([[954,56],[1007,65],[1046,71],[1083,79],[1125,88],[1165,101],[1208,114],[1246,127],[1274,137],[1303,149],[1324,159],
[950,93],[998,100],[1037,103],[1074,109],[1114,119],[1155,128],[1199,141],[1235,152],[1264,162],[1290,174],[1314,184],
[947,123],[991,128],[1029,135],[1067,140],[1104,148],[1145,157],[1189,169],[1222,180],[1252,189],[1280,199],[1303,209],
[943,152],[986,156],[1023,161],[1061,168],[1097,175],[1136,184],[1179,193],[1214,203],[1242,212],[1271,221],[1293,229],
[940,182],[980,186],[1018,190],[1053,196],[1090,201],[1129,208],[1171,219],[1203,227],[1232,235],[1261,243],[1286,252],
[936,210],[976,212],[1012,216],[1046,221],[1083,227],[1121,232],[1163,242],[1196,248],[1224,257],[1253,263],[1278,269],
[933,238],[973,240],[1007,242],[1041,247],[1076,252],[1115,259],[1156,265],[1188,271],[1216,277],[1243,286],[1267,292],
[929,270],[968,274],[1003,276],[1034,279],[1070,284],[1107,288],[1148,292],[1179,298],[1208,303],[1234,307],[1259,312]]).reshape(-1, 2) 
#[927,309],[964,311],[996,312],[1029,313],[1062,316],[1101,319],[1140,322],[1170,326],[1199,329],[1225,333],[1250,336]
#%%
corners123 = np.array([[1107,133],[1102,148],[1098,160],[1094,173],[1089,186],[1086,198],[1083,211],[1080,223],[1076,237],[1073,250],[1071,263],
[1090,129],[1084,144],[1080,157],[1077,170],[1073,183],[1070,195],[1066,208],[1063,222],[1060,235],[1056,248],[1053,261],
[1071,124],[1066,141],[1062,154],[1058,166],[1055,180],[1052,192],[1049,205],[1045,219],[1042,232],[1040,246],[1037,259],
[1052,122],[1066,141],[1062,153],[1058,166],[1056,179],[1052,193],[1049,205],[1045,219],[1042,233],[1040,245],[1037,259],
[1052,121],[1048,137],[1044,150],[1040,163],[1038,176],[1035,189],[1031,203],[1028,216],[1025,230],[1023,243],[1020,257],
[1033,119],[1028,134],[1025,147],[1022,160],[1019,173],[1016,186],[1013,201],[1010,214],[1007,228],[1005,241],[1003,256],
[1013,115],[1010,130],[1007,143],[1003,157],[1000,171],[997,184],[994,199],[991,213],[989,226],[986,240],[985,254],
 [992,111],[989,127],[985,142],[983,155],[980,169],[978,183],[975,197],[972,211],[970,224],[968,238],[966,253]]).reshape(-1, 2)   #
 #[972,107],[968,125],[965,139],[963,153],[961,168],[958,181],[956,195],[954,210],[952,224],[950,237],[948,252]      
#%%
corners124 = np.array([[44,350],[51,346],[57,344],[68,336],[83,327],[98,316],[125,296],[162,275],[206,252],[260,229],[336,198],
[54,366],[60,363],[67,362],[76,355],[94,350],[110,343],[139,332],[177,318],[224,304],[279,284],[358,260],
[59,376],[65,375],[72,373],[84,369],[100,365],[118,361],[147,355],[187,347],[238,339],[301,331],[379,319],
[64,387],[70,386],[79,388],[93,388],[112,389],[133,390],[161,391],[204,391],[259,392],[323,392],[406,394],
[69,393],[75,394],[84,398],[101,402],[120,406],[144,412],[174,419],[222,428],[283,442],[346,451],[433,465],
[72,401],[80,404],[89,408],[108,416],[129,423],[152,433],[186,446],[240,465],[300,485],[369,508],[453,530],
[79,413],[85,415],[96,421],[116,432],[142,448],[164,459],[202,481],[261,512],[329,543],[393,566],[475,589],
[87,427],[93,430],[107,439],[125,450],[152,469],[178,488],[211,509],[279,552],[348,583],[412,606],[494,635]] ).reshape(-1, 2)  
#%%
corners125 = np.array([[1379,267],[1443,290],[1491,311],[1515,320],[1534,329],[1547,337],[1558,342],[1563,346],[1568,349],[1573,352],[1575,353],
[1361,308],[1430,325],[1477,341],[1504,348],[1524,355],[1539,361],[1548,364],[1554,367],[1560,370],[1564,372],[1569,374],
[1338,367],[1413,374],[1456,377],[1489,382],[1509,383],[1525,384],[1538,386],[1545,387],[1551,388],[1557,388],[1562,389],
[1311,431],[1390,424],[1436,419],[1473,415],[1495,411],[1513,408],[1529,406],[1536,406],[1542,404],[1549,403],[1555,402],
[1279,495],[1362,480],[1410,472],[1451,463],[1471,456],[1493,448],[1508,443],[1518,440],[1526,437],[1533,436],[1542,432],
[1247,561],[1332,535],[1387,519],[1430,503],[1452,496],[1475,486],[1492,477],[1505,471],[1514,465],[1521,462],[1530,459],
[1217,623],[1300,596],[1360,571],[1407,550],[1433,537],[1458,524],[1473,516],[1488,507],[1500,500],[1508,494],[1518,489],
[1177,694],[1272,654],[1334,622],[1385,593],[1417,572],[1442,557],[1461,544],[1478,534],[1490,526],[1498,521],[1507,516]]).reshape(-1, 2) 
#%%
corners126 = np.array([[720,305],[792,302],[878,303],[966,306],[1049,313],[1113,319],[1174,323],[1238,331],[1281,337],[1315,345],[1342,351],
[722,354],[793,353],[874,353],[959,355],[1040,360],[1103,364],[1164,367],[1221,372],[1265,377],[1300,381],[1326,384],
[724,433],[793,433],[871,434],[950,435],[1023,436],[1088,436],[1144,435],[1198,434],[1242,431],[1277,427],[1306,427],
[726,506],[796,511],[869,512],[940,513],[1007,511],[1070,509],[1126,504],[1175,499],[1217,494],[1247,491],[1274,486],
[728,562],[798,565],[871,567],[935,565],[997,559],[1057,553],[1114,546],[1158,542],[1201,535],[1232,530],[1257,524],
[731,615],[801,615],[873,617],[930,615],[990,612],[1041,605],[1097,595],[1141,586],[1184,577],[1213,570],[1240,562],
[734,656],[803,659],[874,659],[927,657],[982,654],[1030,648],[1082,639],[1129,626],[1167,617],[1198,611],[1218,603],
[737,683],[801,687],[874,689],[925,687],[978,684],[1023,679],[1074,666],[1119,656],[1154,646],[1187,636],[1207,628]]).reshape(-1, 2) 
#%% big grid 10*21
corners127 = np.array([
    [55,350],[76,335],[104,318],[130,301],[166,278],[217,244],[285,206],[368,166],[464,130],[573,101],[720,77],[849,68],[1005,83],[1149,116],[1261,155],[1351,188],[1427,230],[1488,268],[1534,298],[1561,319],[1581,332],
    [60,359],[81,348],[109,333],[139,319],[178,300],[230,279],[304,256],[389, 228],[482,202],[588,179],[724,161],[849,155],[986,168],[1127,191],[1240,218],[1325,250],[1404,277],[1470,302],[1520,323],[1549,333],[1574,347],
    [64,366],[86,357],[116,349],[146,339],[189,325],[246,313],[317,294],[407,273],[497,257],[600,243],[727,236],[848,237],[971,241],[1106,257],[1218,279],[1305,296],[1385,319],[1454,336],[1505,351],[1537,361],[1562,369],
    [67,372],[92,369],[122,362],[157,357],[199,351],[254,339],[329,327],[420,315],[509,303],[609,294],[733,288],[849,288],[965,294],[1093,303],[1206,316],[1290,329],[1377,344],[1442,357],[1498,368],[1531,373],[1559,378],
    [71,380],[98,381],[129,377],[163,377],[207,374],[264,370],[343,367],[435,365],[524,366],[621,364],[737,364],[846,368],[956,370],[1074,370],[1187,374],[1273,375],[1361,377],[1431,380],[1489,385],[1523,389],[1551,389],
    [76,389],[103,393],[136,395],[175,401],[222,407],[279,412],[361,419],[454,429],[539,434],[633,439],[743,442],[844,446],[946,449],[1054,447],[1161,441],[1249,435],[1337,429],[1410,422],[1474,414],[1511,410],[1542,404],
    [82,403],[114,414],[150,425],[190,437],[238,448],[300,464],[383,485],[473,499],[555,509],[648,516],[748,526],[844,528],[935,526],[1035,521],[1139,509],[1227,495],[1311,481],[1388,467],[1452,453],[1493,442],[1528,429],
    [91,421],[125,437],[164,457],[207,476],[257,497],[323,522],[404,551],[493,575],[574,594],[662,605],[752,614],[843,617],[925,614],[1012,605],[1114,586],[1199,569],[1277,553],[1357,530],[1423,509],[1470,489],[1503,474],
    [105,449],[144,475],[186,505],[234,536],[284,563],[355,594],[433,627],[518,654],[590,671],[673,683],[756,694],[843,693],[922,685],[997,675],[1089,661],[1173,638],[1251,616],[1329,583],[1402,552],[1447,530],[1488,505],
    [118,480],[163,518],[208,554],[254,584],[304,619],[379,658],[458,695],[538,718],[602,737],[684,754],[761,762],[841,763],[914,760],[981,753],[1063,741],[1147,725],[1217,702],[1294,657],[1370,612],[1427,574],[1473,543]]).reshape(-1, 2) 
#%%
num = str(127)
corners = corners127.reshape(-1,1,2)
img = cv2.imread('chessboard/bkg-' + num +'.jpg')
plt.imshow(img)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x,y),3,255,-1)
plt.imshow(img)
np.save("gridpoints/corners" + num, corners)


#%%
 
# # 找棋盘格角点
# # 阈值
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # 棋盘格模板规格
# w = 9
# h = 6
# # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
# objp = np.zeros((w * h, 3), np.float32)
# objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# # 储存棋盘格角点的世界坐标和图像坐标对
# objpoints = []  # 在世界坐标系中的三维点
# imgpoints = []  # 在图像平面的二维点
 
# images = glob.glob('../fisheye_cali/chess*.png')
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 找到棋盘格角点
#     ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
#     # 如果找到足够点对，将其存储起来
#     if ret == True:
#         cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#         objpoints.append(objp)
#         imgpoints.append(corners)
#         # 将角点在图像上显示
#         cv2.drawChessboardCorners(img, (w, h), corners, ret)
#         plt.imshow(img)
#         # cv2.imwrite('D:images\\grid_out.png', img)
#         cv2.waitKey(1)
# cv2.destroyAllWindows()
#%%
w = 11
h = 8
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in [117,118,119,120,123,125,126]:#,116,117,118,119,120,121,122,123]:#,124,125,126]:
    corners = np.load("gridpoints/corners" + str(i) + ".npy").astype(np.float32)
    objpoints.append(objp)
    imgpoints.append(corners)


#%%
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(mtx)
# print(dist)
# print(rvecs)
# print(tvecs)
# 去畸变
# img2 = cv2.imread('../WechatIMG17.png')
img2 = cv2.imread('chessboard/bkg-120.jpg')
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
plt.imshow(dst)
# 根据前面ROI区域裁剪图片
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]

# cv2.imwrite('undistorted.png', dst)
 

# #%%
# # 反投影误差
# total_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     total_error += error
# print("total error: ", total_error / len(objpoints))
 
# # 校正视频
# cap = cv2.VideoCapture('D:video\\video.mp4')
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_size = (width, height)
# video_writer = cv2.VideoWriter('D:video\\result2.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
# for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
#     ret, frame = cap.read()
#     if ret:
#       image_ = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#       cv2.imshow('jiaozheng', image_)
#       # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#       video_writer.write(image_)
#     if cv2.waitKey(10) & 0xFF== ord('q'):
#         break
# cap.release()
# # cv2.destroyALLWindows()