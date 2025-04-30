from ultralytics import YOLO
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("yolo11n.pt")


results = model.train(
    data='data.yaml',  # путь к файлу с описанием датасета
    epochs=100,  # количество эпох
    batch=4,  # размер батча
    imgsz=640,  # размер изображения
    device='cpu',  # использовать GPU 0, для CPU укажите 'cpu'
    workers=2,  # количество workers для загрузки данных
    optimizer='Adam',  # автоматический выбор оптимизатора
    lr0=0.01,  # начальный learning rate
    lrf=0.01,  # конечный learning rate
    patience=10,  # ранняя остановка после patience эпох без улучшений
    project='runs',  # папка для сохранения результатов
    name='exp',  # имя эксперимента
    pretrained=True,  # использовать предобученные веса
    seed=42  # фиксируем seed для воспроизводимости
)
