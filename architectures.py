import keras.applications as classifiers

class InceptionResNetV2:
    def build_network(self, input_shape, num_classes):
        model = classifiers.inception_resnet_v2.InceptionResNetV2(include_top=True,
                                                                  weights=None,
                                                                  input_tensor=None,
                                                                  input_shape=input_shape,
                                                                  pooling='max',
                                                                  classes=num_classes)
        return model

class NASNetLarge:
    def build_network(self, input_shape, num_classes):
        model = classifiers.nasnet.NASNetLarge(include_top=True,
                                               weights=None,
                                               input_tensor=None,
                                               input_shape=input_shape,
                                               pooling='max',
                                               classes=num_classes)
        return model
class Xception:
    def build_network(self, input_shape, num_classes):
        model = classifiers.xception.Xception(include_top=True,
                                              weights=None,
                                              input_tensor=None,
                                              input_shape=input_shape,
                                              pooling='max',
                                              classes=num_classes)
        return model

class InceptionV3:
    def build_network(self, input_shape, num_classes):
        model = classifiers.inception_v3.InceptionV3(include_top=True,
                                                     weights=None,
                                                     input_tensor=None,
                                                     input_shape=input_shape,
                                                     pooling='max',
                                                     classes=num_classes)
        return model

class MobileNet:
    def build_network(self, input_shape, num_classes):
        model = classifiers.mobilenet.MobileNet(input_shape=input_shape,
                                                alpha=1.0,
                                                depth_multiplier=1,
                                                dropout=1e-3,
                                                include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                pooling='max',
                                                classes=num_classes)
        return model

class MobileNetV2:
    def build_network(self, input_shape, num_classes):
        model = classifiers.mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                     alpha=0.2,
                                                     include_top=True,
                                                     weights=None,
                                                     input_tensor=None,
                                                     pooling='max',
                                                     classes=num_classes)
        return model

class ResNet50:
    def build_network(self, input_shape, num_classes):
        model = classifiers.resnet50.ResNet50(include_top=True,
                                              input_shape=input_shape,
                                              weights=None,
                                              input_tensor=None,
                                              pooling='max',
                                              classes=num_classes)
        return model

class ResNet50V2:
    def build_network(self, input_shape, num_classes):
        model = classifiers.resnet_v2.ResNet50V2(include_top=True,
                                                 input_shape=input_shape,
                                                 weights=None,
                                                 input_tensor=None,
                                                 pooling='max',
                                                 classes=num_classes)
        return model

class ResNext:
    def build_network(self, input_shape, num_classes):
        model = classifiers.resnext.ResNeXt50(include_top=True,
                                              input_shape=input_shape,
                                              weights=None,
                                              input_tensor=None,
                                              pooling='max',
                                              classes=num_classes)
        return model

class VGG16:
    def build_network(self, input_shape, num_classes):
        model = classifiers.vgg16.VGG16(include_top=True,
                                        input_tensor=None,
                                        weights=None,
                                        input_shape=input_shape,
                                        pooling='max',
                                        classes=num_classes)
        return model


class VGG19:
    def build_network(self, input_shape, num_classes):
        model = classifiers.vgg16.VGG16(include_top=True,
                                        weights=None,
                                        input_tensor=None,
                                        input_shape=input_shape,
                                        pooling='max',
                                        classes=num_classes)
        return model
