#!/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os

# Função para plotar
color = [0, 0, 255]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Metadado das imagens
# Imagem é dividida em 4 slices
#      ------ 1920 ------              
#      |                |              ---960---
# 1080 |                |  =>  4x  540 | Slice |
#      |                |              --------- 
#      ------------------
# 
# Slices da Imagem
#  -------------------
#  |   0   |    1    |
#  -------------------
#  |   2   |    3    |
#  -------------------
#
# Coordenadas das imagens usa o padrão do opencv
#
# Eixo y: 0 -----------------|
#         |                  |
#         |                  |
#       ymax                 |
# Eixo x: 0---------------xmax

tam_total_width = 1920
tam_total_height = 1080
tam_slice_width = 960
tam_slice_height = 540

# Input, bounding boxes de cada slice
# A entrada é uma lista de boxes
#
# OBS: O campo ID é só para lhe ajudar na hora do merge, no final vou remover)
#
# A entrada abaixo mostra que este slice detectou duas classes (0 e 1)
# Para a classe 0, detectou dois objetos (id 1 e 2) 
# Lembre que o id é só pra te ajudar na hora do merge aqui do teste, na hora não terá
# Ex:
#     [
#            {'id': 1, 'xmin': 1.0, 'ymin': 52.0, 'xmax': 79.0, 'ymax': 230.0, 'conf': 0.93}
#            {'id': 2, 'xmin': 371.0, 'ymin': 3.0, 'xmax': 442.0, 'ymax': 130.0, 'conf': 0.722}
#     ]

# Bounding boxes (entrada)
bb_slice0 = []  # Não detectou nada
bb_slice1 = [{'id': 1, 'xmin': 371.0, 'ymin': 456.0, 'xmax': 431.0, 'ymax': 540.0, 'conf': 0.3491985499858856}]
bb_slice2 = [{'id': 2, 'xmin': 888.0, 'ymin': 68.0, 'xmax': 952.0, 'ymax': 240.0, 'conf': 0.871227502822876},
             {'id': 3, 'xmin': 922.0, 'ymin': 63.0, 'xmax': 960.0, 'ymax': 228.0, 'conf': 0.6695037484169006}]
bb_slice3 = [{'id': 4, 'xmin': 1.0, 'ymin': 52.0, 'xmax': 79.0, 'ymax': 230.0, 'conf': 0.9308098554611206},
             {'id': 5, 'xmin': 370.0, 'ymin': 3.0, 'xmax': 442.0, 'ymax': 130.0, 'conf': 0.9237046241760254},
             {'id': 6, 'xmin': 163.0, 'ymin': 3.0, 'xmax': 243.0, 'ymax': 189.0, 'conf': 0.9022660851478577},
             {'id': 7, 'xmin': 232.0, 'ymin': 0.0, 'xmax': 311.0, 'ymax': 209.0, 'conf': 0.8999184966087341},
             {'id': 8, 'xmin': 88.0, 'ymin': 0.0, 'xmax': 169.0, 'ymax': 230.0, 'conf': 0.8105608224868774}]

# Visualizando os slices
# Img Slice
img_slice0 = os.path.join('data', 'samples', 'test1', '31930_0.jpg')
img_slice1 = os.path.join('data', 'samples', 'test1', '31930_1.jpg')
img_slice2 = os.path.join('data', 'samples', 'test1', '31930_2.jpg')
img_slice3 = os.path.join('data', 'samples', 'test1', '31930_3.jpg')

imgs = [img_slice0, img_slice1, img_slice2, img_slice3]
bbs = [bb_slice0, bb_slice1, bb_slice2, bb_slice3]
for i in range(len(imgs)):
    img = cv2.imread(imgs[i])  # BGR in HxWxC
    for bb in bbs[i]:
        print("aqui", bb)
        box = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]
        label = 'person' + str(bb['id'])
        plot_one_box(box, img, color, label)

    cv2.imshow("img", img)
    if cv2.waitKey(0) == ord('q'):  # q to quit
        raise StopIteration


# Entrada da função
# Tamanho da iamgem: tam_total_width, tam_total_height, tam_slice_width, tam_slice_height
# bounding boxes de cada slice (usando a ref de cada slice)
#
# Saída:
# uma única lista de bounding boxes, que leva em conta a dimensão da imagem como um todo
# o valor de confiança será o maior dos dois boxes e o quadrante será o maior que junta os dois
#
# Neste caso deve-se fazer o merge do bb de id 1 do slice 1 com o bb de id 5 slice 3
#

def fix_bounding_box(tam_total_width, tam_total_height, tam_slice_width, tam_slice_height,
                     bb_slice0, bb_slice1, bb_slice2, bb_slice3):
    bb_img = []

    # Img do quadrante 0
    # Nenhuma

    # Img do quadrante 1
    # Neste caso deve-se fazer o merge do bb de id 1 do slice 1 com o bb de id 5 slice 3
    # Também é importante corrigir a posição do xmin, ymin, etc para o quadrante da imagem
    # A bb acontece no slice 1, que é no topo direito, então temos que somar o tam_slice_width no ref x
    # Ex:
    # {'id': 1, 'xmin': 371.0, 'ymin': 456.0, 'xmax': 431.0, 'ymax': 540.0, 'conf': 0.3491985499858856}
    # {'id': 5, 'xmin': 370.0, 'ymin': 3.0, 'xmax': 442.0, 'ymax': 130.0, 'conf': 0.9237046241760254}
    # Saída:
    # {'id': 1, 'xmin': 370.0, 'ymin': 456.0, 'xmax': 442.0, 'ymax': 130.0, 'conf': 0.9237046241760254}
    # Ajuste do quadrante
    # {'id': 1, 'xmin': 370.0 + tam_slice_width, 'ymin': 456.0,
    #           'xmax': 442.0 tam_slice_width, 'ymax': 130.0 + tam_slice_height, 'conf': 0.9237046241760254}
    bb_img.append({'id': 1, 'xmin': 370.0 + tam_slice_width, 'ymin': 456.0,
                   'xmax': 442.0 + tam_slice_width, 'ymax': 130.0 + tam_slice_height, 'conf': 0.9237046241760254})

    # Imgs do quadrante 2
    bb_img.append(
        {'id': 2, 'xmin': 888.0, 'ymin': 68.0 + tam_slice_height, 'xmax': 952.0, 'ymax': 240.0 + tam_slice_height,
         'conf': 0.871227502822876})
    bb_img.append(
        {'id': 3, 'xmin': 922.0, 'ymin': 63.0 + tam_slice_height, 'xmax': 960.0, 'ymax': 228.0 + tam_slice_height,
         'conf': 0.6695037484169006})

    # Imgs do terceiro quadrante
    # Não adicionamos o 5 (que já teve merge)
    bb_img.append({'id': 4, 'xmin': 1.0 + tam_slice_width, 'ymin': 52.0 + tam_slice_height,
                   'xmax': 79.0 + tam_slice_width, 'ymax': 230.0 + tam_slice_height, 'conf': 0.9308098554611206})
    bb_img.append({'id': 6, 'xmin': 163.0 + tam_slice_width, 'ymin': 3.0 + tam_slice_height,
                   'xmax': 243.0 + tam_slice_width, 'ymax': 189.0 + tam_slice_height, 'conf': 0.9022660851478577})
    bb_img.append({'id': 7, 'xmin': 232.0 + tam_slice_width, 'ymin': 0.0 + tam_slice_height,
                   'xmax': 311.0 + tam_slice_width, 'ymax': 209.0 + tam_slice_height, 'conf': 0.8999184966087341})
    bb_img.append({'id': 8, 'xmin': 88.0 + tam_slice_width, 'ymin': 0.0 + tam_slice_height,
                   'xmax': 169.0 + tam_slice_width, 'ymax': 230.0 + tam_slice_height, 'conf': 0.8105608224868774})

    return bb_img


bb_img = fix_bounding_box(tam_total_width, tam_total_height, tam_slice_width, tam_slice_height,
                          bb_slice0, bb_slice1, bb_slice2, bb_slice3)

# Imagem Total
img_total = os.path.join('data', 'samples', 'test1', '31930.jpg')
img = cv2.imread(img_total)  # BGR in HxWxC

for bb in bb_img:
    box = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]
    label = 'person' + str(bb['id'])
    plot_one_box(box, img, color, label)

cv2.imshow("img", img)
if cv2.waitKey(0) == ord('q'):  # q to quit
    raise StopIteration
