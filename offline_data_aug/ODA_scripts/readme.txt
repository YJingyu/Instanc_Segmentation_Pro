data process pipeline
1. json2txt.py  将train和val的数据合并，根据mask得到每个图像轮廓，并将图片中每个类别的轮廓坐标信息存储为txt文件
2. crop_objects.py  将human和ball从原图中crop出来，裁剪后的图片存在c_images下面,坐标信息存放在c_labels下面
3. sort_images.py  将images根据图片中实例的数量进行分类，存放在src_images下面，文件夹名称表示图片中含实例的个数
4. copy_file.py  将src_images中的分类后的图片每张复制相同的20份，对应的标注也是复制20份
5. add_object.py  主要是对没有实例的图片进行处理，paste一个固定的人上去，然后将src_images中各个分类的图片汇总到aug_images中，标注也进行类似的操作
6. convert2coco  将aug_labels中的格式转化为标注的coco数据集格式