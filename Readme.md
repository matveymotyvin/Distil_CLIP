# Knowledge Distillation CLIP-B/32

## 1. Основная идея
Knowledge distillation (KD) – это подход к машинному обучению, который позволяет передавать знания от сложной модели (учитель) к более простой модели (ученик). Суть этого подхода заключается в том, что модель учителя обучается на большом наборе данных и является дорогостоящей с точки зрения вычислений, в то время как модель ученика проще и компактнее. Цель дистилляции состоит в том, чтобы обучить ученика таким образом, чтобы он достиг точности прогнозирования, сравнимой с точностью учителя.
Учитывая, что графическим процессорам потребительского класса сложно одновременно обрабатывать две модели CLIP, в этом проекте отдельно дистиллируются кодировщики текста и изображений. По итогу работы программы получаются две модели.
## 2. Подготовка данных
⚠️ Все данные должны быть помещены в одну директорию. 

### Датасет изображений
В качестве датасета изображений, можно использовать любые датасеты изображений, например, ImageNet, COCO, Lsun, Open Images и др. Но учитывая специфику модели CLIP и ее направленность на решение сложных задач, ориентированных на поиск различных деталей на изображениях, рекомендуется брать датасеты с более сложными и разнообразными визуальными сценариями, а также более мелкими деталями. 
- Примечание: следует учитывать, что модели VIT требуют большого объема данных для достижения хороших результатов (не менее 1 миллиона изображений).

В качестве валидационного датасета используется [MSCOCO2017Val](http://images.cocodataset.org/zips/val2014.zip). Если вы хотите чтобы валидация корректно работала без дополнительных изменений в коде программы, просим загрузить данный датасет.

### Текстовый данные
В качестве текстовых данных для обучения кодировщика текста предлагается использовать два приведенных датасета (вы можете использовать другие текстовые датасеты):
1. Аннотации [MSCOCO2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
2. GCC (Google's Conceptual Captions) из [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

- После загрузки вы должны получить эти два файла (поместите их в папки COCO и CC соответственно, причем эти две папки должны находиться в одной дирректории)
  1. /path/to/data/COCO/annotations/captions_train2017.json
  2. /path/to/data/CC/Train_GCC-training.tsv


Наконец, вы получите папку с данными, как показано ниже:
```
/path/to/data/image/...   # в папке image должны быть только изображения без других вложенных папок
/path/to/data/COCO/val2017/...
/path/to/data/COCO/annotations/...
/path/to/data/CC/Train_GCC-training.tsv
```

## 3. Запуск процесса дистилляции моделей
Как описывалось выше процесс дистилляции осуществляется отдельно для двух моделей. Запускать обучение нужно также отдельно для каждой модели с помощью командной строки.

1.1. Запуск обучения кодировщика изображений на нескольких GPU
```
python main.py --data_dir /path/to/data/ --strategy=ddp --gpus=0,1,2,3 --precision=16 --max_epochs=100 --dataset=image_dataset --model_name=model_image_distilled
```
1.2. Запуск обучения кодировщика изображений на одном GPU
```
python main.py --data_dir /path/to/data/ --gpus=0 --precision=16 --max_epochs=100 --dataset=image_dataset --model_name=model_image_distilled
```
2.1. Запуск обучения кодировщика текста на нескольких GPU
```
python main.py --data_dir /path/to/data/ --strategy=ddp --gpus=0,1,2,3 --precision=16 --max_epochs=100 --dataset=text_dataset --model_name=model_text_distilled
```
2.2. Запуск обучения кодировщика текста на одном GPU
```
python main.py --data_dir /path/to/data/ --gpus=0 --precision=16 --max_epochs=100 --dataset=text_dataset --model_name=model_text_distilled
```
Обучение осуществляется с помощью PyTorch-Lightning, подробнее почитать о настройке параметров и о процессе обучения можно почитать [тут](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api).

### Настройка функции ошибок
В данном проекте в качестве функции ошибок используются три функции: [KL-дивергенция](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html), [кросс-энтропия](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html), [L1 loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html).

Если все три функции выбраны итоговая ошибка вычисляется, как взвещенная сумма трех функция:
```
loss = weight_1 * scale_1 * l1 + weight_2 * scale_2 * ce  + weight_3 * scale_3 * kl
```
Веса по умолчанию: 
```
parser.add_argument('--weight', default=[0.6, 0.35, 0.05], nargs='+', type=list)
```

### Настройка параметров моделей осуществляется также при запуске процесса обучения, указанием названий параметров и соответствующих значений.  
Эти параметры могут быть изменены для настройки структуры моделей учеников.
```
    # Параметры по умолчанию для кодировщика изображений
    parser.add_argument('--input_resolution', default=224, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--width', default=576, type=int)
    parser.add_argument('--layers', default=6, type=int)
    parser.add_argument('--heads', default=24, type=int)

    # Параметры по умолчанию для кодировщика текста
    parser.add_argument('--context_length', default=77, type=int)
    parser.add_argument('--vocab_size', default=49408, type=int)
    parser.add_argument('--transformer_width', default=128, type=int)
    parser.add_argument('--transformer_layers', default=6, type=int)
    parser.add_argument('--transformer_heads', default=8, type=int)
```
