import matplotlib.pyplot as plt

masks_dir = '../../runs/pred_pic/'
images_dir = '../../datasets/'
images_name = ['2007_000033', '2007_000042', '2007_000061', '2007_000123', '2007_000129', '2007_000175']
models_name = ['original', 'fcn32s_vgg16_pascal_voc', 'fcn16s_vgg16_pascal_voc', 'fcn8s_vgg16_pascal_voc', 'deeplabv3_resnet101_pascal_voc']

plot_index = 0
row = len(images_name)
col = len(models_name)

figure = plt.figure(num='test result')

for i in range(row):
    for j in range(col):
        if j == 0:
            subplot = plt.imread('/home/kuangbixia/projects/awesome-semantic-segmentation-pytorch/datasets/voc/VOC2012/JPEGImages/' + images_name[i] + '.jpg')
        else:
            subplot = plt.imread(masks_dir + '/' + models_name[j] + '/' + images_name[i] + '.png')
        plot_index += 1
        plt.subplot(row, col, plot_index)
        plt.imshow(subplot)
        plt.axis('off')
        if i == 0:
            plt.title(models_name[j])
plt.show()
