# 基于tesseract 证件识别
环境依赖：
ubuntu16.04
opencv3.4.0
numpy1.14.2
python2.7
PIL

相关库安装：
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
sudo pip install pytesseract
sudo apt-get install python-imaging
程序说明：

证件位置检测(Blue通道颜色加强+腐蚀膨胀去噪+最小矩形选取）->证件相关区域位置检测<日期、姓名、证件号>(膨胀、腐蚀、规则）->证件识别(灰度化+自适应二值化+多区域腐蚀膨胀+tesseract)

程序使用示例：
python card_recognition [image_path]
其中， image_path使用图片全路径

证件示例：请参见博客
https://blog.csdn.net/small_ARM/article/details/80816902

