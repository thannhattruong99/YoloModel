from tensorflow.keras import callbacks, optimizers
from yolov4.tf import SaveWeightsCallback, YOLOv4
import time

yolo = YOLOv4(tiny=True)
yolo.classes = "/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/main_me/custom.names"
yolo.input_size = 608
yolo.batch_size = 32

yolo.make_model()
# yolo.load_weights(
#     "/content/drive/My Drive/Hard_Soft/NN/yolov4/yolov4-tiny.conv.29",
#     weights_type="yolo"
# )

train_data_set = yolo.load_dataset(
    "/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/main_me/custom_train2017.txt",
    image_path_prefix="/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/train",
    label_smoothing=0.05
)
val_data_set = yolo.load_dataset(
    "/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/main_me/custom_valid2017.txt",
    image_path_prefix="/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/valid",
    training=False
)

epochs = 400
lr = 1e-4

optimizer = optimizers.Adam(learning_rate=lr)
yolo.compile(optimizer=optimizer, loss_iou_type="ciou")

def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01

_callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir="/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/main_me/log",
    ),
    SaveWeightsCallback(
        yolo=yolo, dir_path="/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Aerial Maritime.v9-tiled.coco/main_me/trained",
        weights_type="yolo", epoch_per_save=10
    ),
]

yolo.fit(
    train_data_set,
    epochs=epochs,
    callbacks=_callbacks,
    validation_data=val_data_set,
    validation_steps=50,
    validation_freq=5,
    steps_per_epoch=100,
)