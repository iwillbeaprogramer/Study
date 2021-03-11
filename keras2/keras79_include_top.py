from tensorflow.keras.applications import VGG16

model = VGG16(include_top = False)

model.trainable=False
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

"""
model = VGG16(weights = 'imagenet',include_top=False,input_shape = (224,224,3))
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
26
0
"""


'''
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
32
0
'''