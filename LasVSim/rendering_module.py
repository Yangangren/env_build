# # coding=utf-8
#
# """
# 2D Display Modal of Autonomous Car Simulation System
# Author: Li Bing
# Date: 2017-8-23
# """
# import numpy as np
# import cv2
# from math import pi, cos, sin, ceil, copysign
# from shapely import geometry
# from map_module import *
# from data_structures import MAPS, VehInfo
#
# DISPLAY_PIX_SIZE = (400, 700)
# DISPLAY_GEO_LENGTH=622.0/3360*DISPLAY_PIX_SIZE[1] # 129.6 m
# TEST_MASS_CENTER_2_SHAPE_CENTER = {1000: 1.1, 2000: 1.6}
#
#
# def get_rotate_point(center, angle, pt):
#     """
#     get rotated position of point pt with the center and angle
#     """
#     x0, y0 = center
#     x, y = pt
#     dx, dy = x - x0, y - y0
#     theta = float(angle) / 180.0 * pi
#     x1 = dx * cos(theta) - dy * sin(theta) + x0
#     y1 = dx * sin(theta) + dy * cos(theta) + y0
#     return x1, y1
#
#
# class Display:
#     """
#     Display Class
#     """
#     k_g2p = None    # geo to pix ratio
#     sz_pix = None   # pix size
#     sz_geo = None   # geo size
#     map = None        # map display layer object
#     center = None     # display center
#     angle = None      # display angle
#     canvas = None     # display canvas object
#     vehicle_images = dict()   # vehicle images for all types
#     vehicle_sizes = dict()    # vehicle sizes for all types
#     lights_images = dict()    # images for traffic lights
#     lights_info = []          # traffic lights infomation
#     lights_status = None      # traffic lights status
#     m_g2p = None               # transform matrix for geo to pix
#     vehicles = None           # vehicles data
#     veh_info = None
#     own_car_pos = None        # self car position
#     display_info = None       # other display data
#
#     start_pos = None          # start point position
#     target_pos = None         # target point position
#
#     future_path = None        # route path to display
#
#     own_car_type = None     # 自车类型
#     ego_length = 0.0
#     ego_width = 0.0
#
#     # 中间变量
#     dis = None
#     geo_pt = None
#     geo_angle = None
#     veh_turn_state = None
#     veh_turn_signal = None
#     px = None
#     py = None
#     px_pt = None
#     ang = None
#     img = None
#     mask = None
#     mx = None
#     my = None
#     msz = None
#     mw = None
#     mh = None
#     mr = None
#     mdx = None
#     mx1 = None
#     mx2 = None
#     my1 = None
#     my2 = None
#
#     def __init__(self, canvas=None, map_modal=None, settings=None):
#         self.zoom_ratio = 1.0
#         self.sz_pix = DISPLAY_PIX_SIZE
#         self.k_g2p = DISPLAY_GEO_LENGTH/DISPLAY_PIX_SIZE[1]/self.zoom_ratio
#         if canvas is None:
#             self.canvas = np.zeros((self.sz_pix[1], self.sz_pix[0], 3), dtype='uint8')
#         else:
#             self.canvas=canvas
#             h,w,d=canvas.shape
#             self.sz_pix=(w,h)
#         self.sz_geo = (float(self.sz_pix[0])*self.k_g2p,float(self.sz_pix[1])*self.k_g2p)
#         self.map_modal = map_modal
#         self.map = MapLayer(self.canvas,self.k_g2p,self.map_modal)
#         self.load_traffic_light()
#         self.sensor_enabled=True
#         self.track_enabled=True
#         self.experiment2=False
#         if settings is None:
#             self.ego_length = 4.5
#             self.ego_width = 1.8
#         else:
#             self.ego_length = settings.car_length
#             self.ego_width = settings.car_width
#
#     def zoom(self, mode):
#         k=1.2
#         if mode is 'zoom_in':
#             if self.zoom_ratio<2.5:
#                 self.zoom_ratio*=k
#         elif mode is 'zoom_out':
#             if self.zoom_ratio>0.25:
#                 self.zoom_ratio/=k
#         else:
#             self.zoom_ratio=1.0
#         self.k_g2p=DISPLAY_GEO_LENGTH/DISPLAY_PIX_SIZE[1]/self.zoom_ratio
#         self.sz_geo = (float(self.sz_pix[0])*self.k_g2p,float(self.sz_pix[1])*self.k_g2p)
#         self.map=MapLayer(self.canvas,self.k_g2p,self.map_modal)
#         self.set_start_target(self.start_pos,self.target_pos)
#         self.set_pos(self.center,self.angle)
#         self.load_vehicle_model()
#         self.load_traffic_light()
#
#     def init_canvas(self, canvas):
#         self.sz_pix = DISPLAY_PIX_SIZE
#         self.k_g2p = DISPLAY_GEO_LENGTH / DISPLAY_PIX_SIZE[1] / self.zoom_ratio
#         self.canvas = canvas
#         h, w, d = canvas.shape
#         self.sz_pix = (w, h)
#         self.sz_geo = (
#         float(self.sz_pix[0]) * self.k_g2p, float(self.sz_pix[1]) * self.k_g2p)
#         self.map = MapLayer(self.canvas, self.k_g2p, self.map_modal)
#
#     def reset_canvas(self, canvas):
#         self.sz_pix=DISPLAY_PIX_SIZE
#         self.k_g2p=DISPLAY_GEO_LENGTH/DISPLAY_PIX_SIZE[1]/self.zoom_ratio
#         self.canvas=canvas
#         h,w,d=canvas.shape
#         self.sz_pix=(w,h)
#         self.sz_geo = (float(self.sz_pix[0])*self.k_g2p,float(self.sz_pix[1])*self.k_g2p)
#         self.map=MapLayer(self.canvas,self.k_g2p,self.map_modal)
#         self.load_traffic_light()
#
#         self.set_start_target(self.start_pos,self.target_pos)
#         self.set_pos(self.center,self.angle)
#         self.load_vehicle_model()
#
#         self.draw()
#
#     def setVehicleModel(self, vehicle_model):
#         self.vehicle_model=vehicle_model
#         self.load_vehicle_model()
#
#     def set_pos(self, center, angle):
#         """根据自车当前位置移动渲染画面，保持自车中心始终位于画面正中央"""
#         self.center = center
#         self.angle = 0.0  # 车转地图不转
#         # self.angle = angle  # 车不转地图转
#         cx, cy = center
#         sx, sy = self.sz_geo
#         matrix = cv2.getRotationMatrix2D(center, self.angle, 1)
#
#         display_offset = 0.2
#         dy=display_offset*sy
#         corners_raw = np.array(
#             [[[cx - sx / 2, cy - sy / 2], [cx - sx / 2, cy + sy / 2],
#               [cx + sx / 2, cy + sy / 2],
#               [cx + sx / 2, cy - sy / 2]]],
#             dtype='float32')
#         p1, p2, p3, p4 = cv2.transform(corners_raw, matrix)[0]
#
#         pix_p1 = (0, self.sz_pix[1]-1)
#         pix_p2 = (0, 0)
#         pix_p3 = (self.sz_pix[0]-1, 0)
#         pix_p4 = (self.sz_pix[0]-1, self.sz_pix[1]-1)
#         self.m_g2p = cv2.getAffineTransform(np.array([p1, p2, p3], 'float32'),
#                                             np.array([pix_p1, pix_p2, pix_p3],
#                                                      'float32'))
#         self.map.set_pos(self.m_g2p, [p1, p2, p3, p4])
#
#     def geo2pix(self,pt):
#         pt_px=cv2.transform(np.array([[pt]],'float32'),self.m_g2p)
#         return [int(round(x)) for x in pt_px[0][0]]
#     def set_data(self, vehicles, lights, own_car_pos, detected_vehicles,detected_man,
#                  detected_traffic_sign,
#                  veh_info,state_estimate_kf,state_estimate_aof):
#         self.veh_info = veh_info
#         # for veh in veh_info:
#         #     if veh['x'] < 999999.99:
#         #         print(veh)
#         self.vehicles = veh_info
#         self.lights_status = lights
#         self.own_car_pos = own_car_pos
#         self.detected_vehicles = detected_vehicles
#         self.detected_man=detected_man
#         self.detected_traffic_sign=detected_traffic_sign
#         #毕业
#         self.state_estimate=copy.deepcopy(state_estimate_kf)
#         self.state_estimate_aof=copy.deepcopy(state_estimate_aof)
#
#     def set_info(self, info, path):
#         self.display_info = info
#         self.future_path = path
#
#     def set_sensor_enabled(self,enabled):
#         self.sensor_enabled=enabled
#
#     def set_track_enabled(self,enabled):
#         self.track_enabled=enabled
#
#     def draw(self):
#         self.canvas[:]=200
#         self.map.draw()
#         if self.track_enabled:
#             self.draw_path()
#         # self.draw_start_target()
#         self.draw_vehicles()
#         if self.map.map.map_type == MAPS[0]:
#             self.draw_traffic_light()
#         elif self.map.map.map_type == MAPS[1]:
#             pass
#         if self.sensor_enabled:
#             self.draw_sensor_range()
#         # v=self.own_car_pos[2]*3.6
#         # cv2.putText(self.canvas,'speed:%0.1f km/h'%v,(0,695),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,200))
#         # cv2.putText(self.canvas,'time:%0.1f s'%self.display_info['t'],(0,695-15),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,200))
#
#     def uishow(self):
#         cv2.imshow('Autonomous Car Simulation',self.canvas)
#         key=cv2.waitKey(1)
#         return key
#
#     def draw_vehicles(self):
#         """渲染交通流"""
#         for i in range(len(self.veh_info)):
#             if self.veh_info[i].render_flag:
#                 self.__draw_vehicle(self.veh_info[i])
#         if self.sensor_enabled:
#             self.draw_detected_pos()
#         self.draw_own_car()
#
#     def draw_own_car(self):
#         (x, y, s, a) = self.own_car_pos
#         #self.own_car_type = 'Truck'  # TODO(Chason)：回放模式下自车默认为小车，待修改
#         if self.own_car_type == 'Truck':
#             type = 2000
#         elif self.own_car_type == 'AMT Car':
#             type = 0
#         elif self.own_car_type == 'CVT Car':
#             type = 2
#         elif self.own_car_type == 'EV':
#             type = 1
#         v = VehInfo(veh_turn_signal=0, veh_brake_signal=0, veh_emergency_signal=0,
#                     veh_type=type, veh_id=-1, veh_width=self.ego_width,
#                     veh_length=self.ego_length, veh_height=0.0, veh_x=x, veh_y=y,
#                     veh_z=0.0, veh_dx=0.0, veh_dy=0.0, veh_dz=0.0, veh_heading=a,
#                     veh_pitch=0.0, car_follow_model=0, lane_change_model=0,
#                     max_acc=0.0, max_dec=0.0, render_flag=True, turn_state=0)
#         self.__draw_vehicle(v)
#
#     def set_start_target(self, s, t):
#         self.start_pos = s
#         self.target_pos = t
#         self.map.set_start_target(s[0],t[0])
#
#     def draw_path(self):
#         """期望轨迹渲染"""
#         #print(self.future_path)
#         if self.future_path is None or len(self.future_path) < 2:
#             return
#         # for x1, y1 in self.future_path:
#         #     print('x: %.1f y: %.1f  |' %(x1,y1)),
#         # print(' ')
#
#         x0, y0 = self.future_path[1]
#         for x1, y1 in self.future_path[2:]:
#             self.draw_line((x0, y0), (x1, y1))
#             x0, y0 = x1, y1
#
#     def draw_line(self, p1, p2, color=(0,180,0)):
#         x1, y1 = [int(f) for f in self.geo2pix(p1)]
#         x2, y2 = [int(f) for f in self.geo2pix(p2)]
#         # if x1>=0 and x1<=self.sz_pix[0] and y1>=0 and y1<=self.sz_pix[1]\
#         #     and x2>=0 and x2<=self.sz_pix[0] and y2>=0 and y2<=self.sz_pix[1]:
#         cv2.line(self.canvas, (x1, y1), (x2, y2), color, 1,)
#
#     def draw_polygon(self, polygon, color):
#         for i in range(len(polygon)):
#             p1=polygon[i]
#             p2=polygon[(i+1)%len(polygon)]
#             self.draw_line(p1,p2,color)
#     def draw_id(self,id,x,y):
#         tmp= self.geo2pix((x, y))
#         locx=int(tmp[0])
#         locy=int(tmp[1])
#         cv2.putText(self.canvas, id, (locx,locy), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.4,(70,255,194) , 1, cv2.LINE_AA)#B,G,R(112, 242, 112)
#     def draw_detected_pos(self):
#         """Draw detected vehicle's detection result outline.
#         """
#         if self.experiment2==False:
#             for i, x, y, v, a, w, l in self.detected_vehicles:
#                 rect = [get_rotate_point((x, y), a-90, p) for
#                         p in [(x-w/2, y-l/2), (x-w/2, y+l/2),
#                               (x+w/2, y+l/2), (x+w/2, y-l/2)]]
#                 self.draw_polygon(rect, (0, 0, 255))
#                 #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
#                 self.draw_id(str(i),x-w/2,y-l/2)
#             for i,x,y,v,a,w,l in self.detected_man:
#                 rect = [get_rotate_point((x, y), a - 90, p) for
#                         p in [(x - w / 2, y - l / 2), (x - w / 2, y + l / 2),
#                               (x + w / 2, y + l / 2), (x + w / 2, y - l / 2)]]
#                 self.draw_polygon(rect, (0, 255, 0))
#                 # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
#                 self.draw_id(str(i), x - w / 2, y - l / 2)
#             for x, y, a, w, l in self.detected_traffic_sign:
#                 rect = [get_rotate_point((x, y), a, p) for
#                         p in [(x - w / 2, y - l / 2), (x - w / 2, y + l / 2),
#                               (x + w / 2, y + l / 2), (x + w / 2, y - l / 2)]]
#                 self.draw_polygon(rect, (0, 0, 255))
#         #汤凯明毕业用
#         # for key in self.state_estimate.keys():
#         #     est = self.state_estimate[key]
#         #     est = est.astype(np.float32).reshape(8)
#         #     #print("est", est)
#         #     #print("in render", np.shape(est))
#         #     x, y, vx, vy, a, yawrate, w, l = est
#         #     a = a / np.pi * 180
#         #     #print(x,y,vx,vy,a,yawrate,w,l)
#         #     rect = [get_rotate_point((x, y), a - 90, p) for
#         #             p in [(x - w / 2, y - l / 2), (x - w / 2, y + l / 2),
#         #                   (x + w / 2, y + l / 2), (x + w / 2, y - l / 2)]]
#         #     self.draw_polygon(rect, (0, 255, 255))
#         else:
#             for key in self.state_estimate_aof.keys():
#                 est = self.state_estimate_aof[key]
#                 est = est.astype(np.float32).reshape(8)
#                 # print("est", est)
#                 # print("in render", np.shape(est))
#                 x, y, vx, vy, a, yawrate, w, l = est
#                 a = a / np.pi * 180
#                 # print(x,y,vx,vy,a,yawrate,w,l)
#                 rect = [get_rotate_point((x, y), a - 90, p) for
#                         p in [(x - w / 2, y - l / 2), (x - w / 2, y + l / 2),
#                               (x + w / 2, y + l / 2), (x + w / 2, y - l / 2)]]
#                 self.draw_polygon(rect, (255, 0 ,255))
#                 self.draw_id(key, x - w / 2, y - l / 2)
#                 # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
#                 #BGR
#
#
#     def load_traffic_light(self):
#         light_path='Resources/Rendering'
#         sz=(int(7*self.zoom_ratio)*3,int(7*self.zoom_ratio))
#         cnt=(sz[0]/2,sz[1]/2)
#
#         for color in ['red','yellow','green']:
#             self.lights_images[color]=self.__load_obj_image('%s/%s.png'%(light_path,color),sz,cnt)
#         for i in range(-1, 2):
#             for j in range(-1, 2):
#                 cross = (622.0*i, 622.0*j)
#                 dx = 9
#                 dy = 18
#                 self.lights_info.append(dict(pos=(cross[0]+dx, cross[1]-dy),
#                                              angle=0, dir='v'))
#                 self.lights_info.append(dict(pos=(cross[0]-dx, cross[1]+dy),
#                                              angle=180, dir='v'))
#                 self.lights_info.append(dict(pos=(cross[0]-dy, cross[1]-dx),
#                                              angle=90, dir='h'))
#                 self.lights_info.append(dict(pos=(cross[0]+dy, cross[1]+dx),
#                                              angle=270, dir='h'))
#
#     def draw_traffic_light(self):
#         if self.map_modal.map_type != 'Map1_Urban Road':
#             return
#         for light in self.lights_info:
#             img, mask = self.lights_images[self.lights_status[light['dir']]]
#             px, py = self.geo2pix(light['pos'])
#
#             px_pt = (int(px), int(py))
#             px_angle = self.angle+light['angle']
#             self.__draw_obj_image(img, mask, px_pt, px_angle)
#
#     def load_vehicle_model(self):
#         types = self.vehicle_model.get_types()
#         for t in types:
#             h, w, x, y, img_path = self.vehicle_model.get_vehicle(t)
#             k = self.k_g2p
#             sz=[int(w/k),int(h/k)]
#             sz[0]=sz[0]-sz[0]%2+1
#             center=(sz[0]/2,int(y/k))
#             self.vehicle_images[t]=self.__load_obj_image(img_path,sz,center)
#             self.vehicle_sizes[t] = (sz[0]/2, int(y/k), sz)
#
#     def __load_obj_image(self,img_path,size,center):
#         img_raw=cv2.resize(cv2.imread(img_path),tuple(size))
#         mask_raw=cv2.resize(cv2.imread(img_path,-1)[:,:,3],tuple(size))
#         px,py=center
#         #print('img_raw',img_raw)
#         #print('mask_raw',mask_raw)
#         px = int(px)
#         py = int(py)
#         #print('px = ',px)
#         #print('py = ',py)
#         r=int(sqrt(px*px+py*py))+1
#         #print('r-py=', r - py)
#         img=cv2.copyMakeBorder(img_raw,r-py,r-(size[1]-py-1),r-px,r-px,borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
#         mask=np.zeros(img_raw.shape,'uint8')
#         mask[:,:,0]=mask_raw
#         mask[:,:,1]=mask_raw
#         mask[:,:,2]=mask_raw
#         mask=cv2.copyMakeBorder(mask,r-py,r-(size[1]-py-1),r-px,r-px,borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
#         return (img,mask)
#
#     def __draw_obj_image(self, img, mask, px_pt, px_ang):
#         """Draw all vehicle's graph, turn light and ego vehicle's outline.
#
#         This function doesn't draw detected vehicle's outline.
#         """
#         # x, y, (w, h) = self.vehicle_sizes[type]
#         # r = img.shape[0] / 2
#         # p1 = (r - w / 2, r - y)
#         # p2 = (r + w / 2, r + h - y)
#         # cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
#         h, w, d = img.shape
#         r = int(h/2)
#         m = cv2.getRotationMatrix2D((r, r), px_ang, 1)
#         img_rot = cv2.warpAffine(img, m, (w, h))
#         mask = cv2.warpAffine(mask, m, (w, h)) > 0
#         x1, y1, x2, y2 = 0, 0, w, h
#         if px_pt[0]-r < 0:
#             x1 = r-px_pt[0]
#         if px_pt[0]-r+x2 > self.sz_pix[0]:
#             x2 = self.sz_pix[0]-px_pt[0]+r
#         if px_pt[1]-r < 0:
#             y1 = r-px_pt[1]
#         if px_pt[1]-r+y2 > self.sz_pix[1]:
#             y2 = self.sz_pix[1]-px_pt[1]+r
#         if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
#             return
#         # print('px_pt[1]',px_pt[1])
#         # print('r',r)
#         # print('y1',y1)
#         # print('y2',y2)
#         # print('px_pt[0]',px_pt[0])
#         # print('x1',x1)
#         # print('x2',x2)
#         np.copyto(self.canvas[px_pt[1]-r+y1:px_pt[1]-r+y2,
#                   px_pt[0]-r+x1:px_pt[0]-r+x2, :],
#                   img_rot[y1:y2, x1:x2, :], where=mask[y1:y2, x1:x2, :])
#
#     def __draw_vehicle(self, veh):
#         # print('done')
#         if veh.veh_type == 1000:
#             self.dis = 1.1
#         elif veh.veh_type == 2000:
#             self.dis = 1.6
#         elif veh.veh_type == 100:
#             self.dis = 2.4
#         elif veh.veh_type == 200:
#             self.dis = 1.1
#         else:
#             self.dis = 1.9
#         self.geo_pt = (veh.veh_x - cos(veh.veh_heading / 180.0 * pi) * self.dis,
#                        veh.veh_y - sin(veh.veh_heading / 180.0 * pi) * self.dis)
#         self.geo_angle = veh.veh_heading - 90.0  # 神仙坐标系
#         self.veh_turn_state = veh.turn_state
#         self.veh_turn_signal = veh.veh_turn_signal
#         self.px, self.py = self.geo2pix(self.geo_pt)
#         self.px_pt = (int(round(self.px)), int(round(self.py)))
#         self.ang = self.geo_angle+self.angle
#         self.img, self.mask = self.vehicle_images[veh.veh_type]
#         self.img = np.array(self.img)
#         self.mask = np.array(self.mask)
#         # 转向灯渲染
#         if self.veh_turn_state and self.veh_turn_signal:
#             self.mx, self.my, self.msz = self.vehicle_sizes[veh.veh_type]
#             self.mw, self.mh = self.msz
#             self.mr = self.img.shape[0]/2
#             if self.veh_turn_signal == 1:
#                 self.mdx = -self.mw/2  # 左转向灯
#             else:
#                 self.mdx = self.mw/2  # 右转向灯
#             self.mx1 = int(self.mr+self.mdx)
#             self.mx2 = int(self.mr+self.mdx)
#             self.my1 = int(self.mr-self.my)
#             self.my2 = int(self.mr+self.mh-self.my)
#             cv2.circle(self.img, (self.mx1, self.my1), 2, (128, 128, 255), -1)
#             cv2.circle(self.img, (self.mx2, self.my2), 2, (128, 128, 255), -1)
#             cv2.circle(self.mask, (self.mx1, self.my1), 2, (1, 1, 1), -1)
#             cv2.circle(self.mask, (self.mx2, self.my2), 2, (1, 1, 1), -1)
#         # draw all vehicle's graph, turn light and ego vehicle's outline.
#         # if veh.veh_type not in [1000, 2000]:
#         self.__draw_obj_image(self.img, self.mask, self.px_pt, self.ang)
#         if veh.veh_type in [1000, 2000]:  # 画自车的外形轮廓
#             veh.veh_length = 6.2  # TODO(Chason)：回放模式下自车默认为小车，待修改
#             veh.veh_width = 2.1  # TODO(Chason)：回放模式下自车默认为小车，待修改
#             rect = [get_rotate_point((veh.veh_x, veh.veh_y), veh.veh_heading-90,
#                                      p) for
#                     p in [(veh.veh_x-veh.veh_width/2, veh.veh_y-veh.veh_length/2),
#                           (veh.veh_x-veh.veh_width/2, veh.veh_y+veh.veh_length/2),
#                           (veh.veh_x+veh.veh_width/2, veh.veh_y+veh.veh_length/2),
#                           (veh.veh_x+veh.veh_width/2, veh.veh_y-veh.veh_length/2)]]
#             self.draw_polygon(rect, (0, 180, 0))
#
#     def set_sensors(self,sensors):
#         self.sensors=sensors
#
#     def draw_sensor_range(self):
#         x,y,v,a=self.own_car_pos
#         a=a-90
#         for s in self.sensors:
#             dx=s.installation_lateral_bias
#             dy=s.installation_longitudinal_bias
#             theta=a/180.0*pi
#             x1=x+dx*cos(theta)-dy*sin(theta)
#             y1=y+dx*sin(theta)+dy*cos(theta)
#             self.draw_sector((x1, y1), s.detection_range,
#                              a+s.installation_orientation_angle,
#                              s.detection_angle)
#
#     def draw_sector(self, center, radius, a_dir, a_range):
#         x,y = self.geo2pix(center)
#         r=int(radius/self.k_g2p)
#         self_ang=270.0-(a_dir+self.angle+a_range/2)  # TODO(Chason): 删去+self.angle
#         self.draw_pix_sector((x,y),r,self_ang,0.0,a_range)
#
#     def draw_pix_sector(self, center, radius, self_ang, a1, a2):
#         img=np.zeros(self.canvas.shape,'uint8')
#         cv2.ellipse(img,center,(radius,radius),self_ang,a1,a2,(255,255,255),-1)
#         self.canvas[:]=cv2.addWeighted(self.canvas,1,img,0.08,0)
#
#
# class MapLayer:
#     """
#     Map Layer Class : static map layer to display
#     """
#     k_g2p = None
#     sz_pix = None
#     pix_grid = None
#     len_grid = None
#     canvas = None
#     X_MIN = -933.0
#     X_MAX = 933.0
#     Y_MIN = -933.0
#     Y_MAX = 933.0
#     GX_MIN = None
#     GX_MAX = None
#     GY_MIN = None
#     GY_MAX = None
#     grid_cycle=None
#     m_g2p=None
#     disp_quad=None
#
#     def __init__(self,canvas,k_g2p,map):
#         self.grid_img = []
#         if map.map_type == MAPS[0]:
#             self.__load_map('Resources/Rendering/map1_res.jpg')
#         elif map.map_type == MAPS[1]:
#             self.__load_map('Resources/Rendering/map2_res.jpg')
#         self.__load_start('Resources/Rendering/start.jpg')
#         self.canvas=canvas
#         h,w,d=canvas.shape
#         self.sz_pix=(w,h)
#         self.k_g2p=k_g2p
#         self.GX_MIN=self.__geo2grid_1d(self.X_MIN)
#         self.GX_MAX=self.__geo2grid_1d(self.X_MAX)
#         self.GY_MIN=self.__geo2grid_1d(self.Y_MIN)
#         self.GY_MAX=self.__geo2grid_1d(self.Y_MAX)
#         self.map=map
#
#     def __grid_geo_rect(self,gx,gy):
#         return ((gx-0.5)*self.len_grid, (gy-0.5)*self.len_grid, self.len_grid,
#                 self.len_grid)
#
#     def __geo2grid_1d(self,x):
#         a=abs(float(x)/self.len_grid)
#         if a <= 0.5:
#             gx=0
#         else:
#             gx=ceil(a-0.5)
#         return int(copysign(gx,x))
#
#     def __geo2grid(self,x,y):
#         return (self.__geo2grid_1d(x),self.__geo2grid_1d(y))
#
#     def set_pos(self, m_g2p, disp_quad):
#         self.m_g2p = m_g2p
#         self.disp_quad = disp_quad
#
#     def get_starg_target_pos(self,x,y):
#         car_offset=1.4
#         road_offset=3.75
#         status,data=self.map.map_position(x,y)
#         if status is not MAP_IN_ROAD:
#             return x,y
#         x1,y1=self.map.get_road_center_point((x,y))
#         dir=data['direction']
#         if dir is 'N':
#             y1+=car_offset
#             x1+=road_offset
#         elif dir is 'S':
#             y1-=car_offset
#             x1-=road_offset
#         elif dir is 'E':
#             x1+=car_offset
#             y1-=road_offset
#         elif dir is 'W':
#             x1-=car_offset
#             y1+=road_offset
#         return x1,y1
#
#     def set_start_target(self,s,t):
#         self.start_pos=self.get_starg_target_pos(*s)
#         self.target_pos=self.get_starg_target_pos(*t)
#
#     def get_grid_image(self,gx,gy):
#         gx=gx%self.grid_cycle
#         gy=gy%self.grid_cycle
#         if gx!=0 and gy!=0:
#             return None
#         elif gy==0:
#             if gx<=self.grid_cycle/2:
#                 return self.grid_img[gx]
#             else:
#                 return cv2.flip(self.grid_img[self.grid_cycle-gx],-1)
#         else:
#             if gy<=self.grid_cycle/2:
#                 return cv2.flip(cv2.transpose(self.grid_img[gy]),0)
#             else:
#                 return cv2.flip(cv2.transpose(self.grid_img[self.grid_cycle-gy]),1)
#
#     def draw(self):
#         p1, p2, p3, p4 = self.disp_quad
#         self.rect = geometry.Polygon([p1, p2, p3, p4, p1])
#         x1 = min(p1[0], p2[0], p3[0], p4[0])
#         x2 = max(p1[0], p2[0], p3[0], p4[0])
#         y1 = min(p1[1], p2[1], p3[1], p4[1])
#         y2 = max(p1[1], p2[1], p3[1], p4[1])
#         gx1 = max(self.__geo2grid_1d(x1), self.GX_MIN)
#         gx2 = min(self.__geo2grid_1d(x2), self.GX_MAX)
#         gy1 = max(self.__geo2grid_1d(y1), self.GY_MIN)
#         gy2 = min(self.__geo2grid_1d(y2), self.GY_MAX)
#         for gx in range(gx1, gx2+1):
#             for gy in range(gy1, gy2+1):
#                 if self.map.map_type == MAPS[1] and gy != 0:
#                     continue
#                 r = self.__grid_geo_rect(gx, gy)
#                 if not self.is_to_display(r):
#                     continue
#                 self.draw_rect_image(self.get_grid_image(gx, gy),
#                                      self.__grid_geo_rect(gx, gy))
#         if self.start_pos:
#             self.draw_start_target(self.start_pos)
#         # if self.target_pos:
#         #     self.draw_start_target(self.target_pos)
#
#     def is_to_display(self, r):
#         grid_rect=geometry.Polygon([(r[0],r[1]),(r[0]+r[2],r[1]),(r[0]+r[2],r[1]+r[3]),(r[0],r[1]+r[3]),(r[0],r[1])])
#         return grid_rect.intersects(self.rect)
#
#     def draw_rect_image(self, im, r):
#         if im is not None:
#             h, w, d = im.shape
#             rect_tr = cv2.transform(np.array([[(r[0], r[1]), (r[0]+r[2], r[1]),
#                                                (r[0]+r[2], r[1]+r[3])]],
#                                              'float32'), self.m_g2p)
#             mat = cv2.getAffineTransform(np.array([(0, h-1), (w-1, h-1),
#                                                    (w-1, 0)], 'float32'),
#                                          rect_tr)
#             cv2.warpAffine(im, mat, self.sz_pix, self.canvas,
#                            borderMode=cv2.BORDER_TRANSPARENT)
#
#     def __load_map(self,res_path):
#         im = cv2.imread(res_path)
#         h, w, d = im.shape
#         self.pix_grid = h
#         cnt = [float(h-1)/2, float(h-1)/2]
#         sz = (h, h)
#         for i in range(int(w/h)):
#             self.grid_img.append(cv2.getRectSubPix(im, sz, tuple(cnt)))
#             cnt[0] = cnt[0]+h
#         self.grid_cycle = len(self.grid_img)*2-1
#         self.len_grid = 622.0/(self.grid_cycle)
#
#     def __load_start(self,res_path):
#         self.start_image=cv2.imread(res_path)
#
#     def draw_start_target(self,pos):
#         x,y=pos
#         w=7.5
#         h=7.5
#         geo_rect=[x-w/2,y-h/2,w,h]
#         if self.is_to_display(geo_rect):
#             self.draw_rect_image(self.start_image,geo_rect)
#
#
# class MapView:
#     """
#     Map View Class: small map and mission route display
#     """
#     def __init__(self, map_type=None):
#         if map_type == MAPS[0] or map_type is None:
#             self.bg_image = cv2.imread(
#                 'Resources/Rendering/urbanroad_map_view.png')
#         elif map_type == MAPS[1]:
#             self.bg_image = cv2.imread(
#                 'Resources/Rendering/highway_map_view.png')
#         self.image=self.bg_image.copy()
#         h,w,d=self.bg_image.shape
#         self.p2g=float(w)/2/933
#         self.width=w
#         self.arrow_image,self.mask=self.__load_arrow()
#
#     def geo2pix(self,p):
#         x,y=p
#         return (int(x*self.p2g)+self.width/2,self.width/2-int(y*self.p2g))
#
#     def set_route(self, route):
#         #r=[self.geo2pix(p) for p in route]
#         r = []
#         for p in route:
#             temp = self.geo2pix(p)
#             x = int(temp[0])
#             y = int(temp[1])
#             r.append((x,y))
#
#         #print('self.bg_image',self.bg_image)
#         # print('r[0]',r[0])
#         # print('r[1]',r[1])
#         for i in range(len(r)-1):
#             cv2.line(self.bg_image,r[i],r[i+1],(0,255,255),2)
#
#     def set_vehicle_pos(self,pos):
#         x,y,a=pos
#         self.__draw_arrow(self.geo2pix((x,y)),a)
#
#     def __load_arrow(self):
#         img_path='Resources/Rendering/arrow.png'
#         img=cv2.imread(img_path)
#         mask_raw=cv2.imread(img_path,-1)[:,:,3]
#         h,w,d=img.shape
#         px,py=w/2+1,h/2+1
#         r=int(sqrt(px*px+py*py))+1
#         mask=np.zeros(img.shape,'uint8')
#         mask[:,:,0]=mask_raw
#         mask[:,:,1]=mask_raw
#         mask[:,:,2]=mask_raw
#         return (img,mask)
#
#     def __draw_arrow(self,px_pt,px_ang):
#         self.image=self.bg_image.copy()
#         h,w,d=self.arrow_image.shape
#         r=int(h/2)
#         m=cv2.getRotationMatrix2D((r,r),px_ang,1)
#         img_rot=cv2.warpAffine(self.arrow_image,m,(w,h))
#         mask=cv2.warpAffine(self.mask,m,(w,h))>0
#         x1,y1,x2,y2=0,0,w,h
#         if px_pt[0]-r<0:
#             x1=r-px_pt[0]
#         if px_pt[0]-r+x2>self.width:
#             x2=self.width-px_pt[0]+r
#         if px_pt[1]-r<0:
#             y1=r-px_pt[1]
#         if px_pt[1]-r+y2>self.width:
#             y2=self.width-px_pt[1]+r
#         if x1>=x2 or y1>=y2 or x1<0 or y1<0 or x2>w or y2>h:
#             return
#         # print('r',r)
#         # print('x1',x1)
#         # print('y1',y1)
#         # print('x2',x2)
#         # print('y2',y2)
#         # print('px_pt[0]',px_pt[0])
#         # print('px_pt[1]',px_pt[1])
#         # np.copyto(self.image[int(px_pt[1])-r+y1:int(px_pt[1])-r+y2,int(px_pt[0])-r+x1:int(px_pt[0])-r+x2,:],img_rot[y1:y2,x1:x2,:],where=mask[y1:y2,x1:x2,:])
#
#         np.copyto(self.image[int(px_pt[1] - r + y1):int(px_pt[1] - r + y2), int(px_pt[0] - r + x1):int(px_pt[0] - r + x2), :],img_rot[int(y1):int(y2), int(x1):int(x2), :], where=mask[int(y1):int(y2), int(x1):int(x2), :])