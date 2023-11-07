from glass_engine import *
from glass_engine.Geometries import * # 导入所有的基本几何体

scene, camera, light, floor = SceneRoam() # 创建基本场景

sphere = Sphere() # 创建一个球体模型
sphere.position.z = 1 # 设置球体位置
scene.add(sphere) # 将球体添加到场景中

camera.screen.show() # 相机显示屏显示渲染结果