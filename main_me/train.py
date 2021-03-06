from tensorflow.keras import callbacks, optimizers
from yolov4.tf import SaveWeightsCallback, YOLOv4

yolo = YOLOv4(tiny=True)
yolo.classes = "../data/classes.names"
# yolo.input_size = 32
yolo.batch_size = 16
# yolo.input_size = 608
# yolo.batch_size = 32


yolo.make_model()
#yolo.load_weights(
#    "./weight/weight/yolov4-tiny-final.weights",
#    weights_type="yolo"
#)

train_data_set = yolo.load_dataset(
    "../data/train/trainPerson.txt",
    image_path_prefix="../data/train/Person",
    label_smoothing=0.05
)

val_data_set = yolo.load_dataset(
    "../data/validation/ValPerson.txt",
    image_path_prefix="../data/validation/Person",
    training=False
)

# epochs = 400
epochs = 50
# lr = 1e-4
lr = 0.01

optimizer = optimizers.Adam(learning_rate=lr)
yolo.compile(loss_iou_type="ciou")

def lr_scheduler(epoch):
        return lr

checkpoint_filepath = "./log/checkpoint"
_callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir="./log/person",
    ),
    callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True
    ),
    SaveWeightsCallback(
        yolo=yolo, dir_path="./weight/weight",
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
    steps_per_epoch=50,
)
