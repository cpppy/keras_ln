from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from keras.utils import plot_model
from matplotlib import pyplot as plt



if __name__ == "__main__":

    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = InceptionV3(include_top=False)


    print(base_model.summary())     # 画出模型结构图，并保存成图片
    plot_model(base_model, to_file='picture/InceptionV3.png')


    # 增加一个空域全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 增加两个全连接层
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # 合并层，构建一个待 fine-turn 的新模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.load_weights("../model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    #                    by_name=True)


    print(base_model.summary())
    plot_model(model, to_file='picture/InceptionV3_new.png')


    # 冻结特征提取层（从 InceptionV3 copy来的层）
    for layer in base_model.layers:
        layer.trainable = False

    # 冻结层后，编译模型
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # 训练数据生成器
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)



    # train_generator = train_datagen.flow_from_directory(
    #     # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
    #     '/data1/zhanglf/myDLStudying/myDataSet/dog_cat_data/train/',
    #     # 训练数据路径，train 文件夹下包含每一类的子文件夹 target_size=(150, 150),
    #     #  图片大小resize成 150x150
    #     batch_size=32, class_mode='categorical')
    #
    # # 使用二分类，返回1-D 的二值标签
    # # 【7】测试数据生成器
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # validation_generator = test_datagen.flow_from_directory('/data1/zhanglf/myDLStudying/myDataSet/dog_cat_data/validation/',
    #                                                         target_size=(150, 150),
    #                                                         batch_size=32,
    #                                                         class_mode='categorical')
    #
    # # 【8】用新数据训练网络分类层
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=2000,
    #                     epochs=1,
    #                     validation_data=validation_generator,
    #                     validation_steps=800)



    # 冻结网络的前两个inception blocks (即前249层)，训练剩余层
    for i, layer in enumerate(base_model.layers):
        # 打印各卷积层的名字
        print(i, layer.name)

    for layer in model.layers[:249]:
        layer.trainable = False

    for layer in model.layers[249:]:
        layer.trainable = True

    # 重新编译模型
    from keras.optimizers import SGD

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy')





    # # 【9】再次训练模型
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=2000,
    #                     epochs=1,
    #                     validation_data=validation_generator,
    #                     validation_steps=800)
