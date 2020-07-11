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

'''
imgs = [img_slice0, img_slice1, img_slice2, img_slice3]
bbs = [bb_slice0, bb_slice1, bb_slice2, bb_slice3]
for i in range(len(imgs)):
    img = cv2.imread(imgs[i])  # BGR in HxWxC
    for bb in bbs[i]:
        box = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]
        label = 'person' + str(bb['id'])
        plot_one_box(box, img, color, label)

    cv2.imshow("img", img)
    if cv2.waitKey(0) == ord('q'):  # q to quit
        raise StopIteration

'''

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

def calc_dif(val1, val2):
    valf = val1 - val2
    if valf < 0:
        return valf*-1
    else:
        return valf

def merge_boxes(bb1, bb2, new_id):
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    conf = 0

    if bb1['ymin'] < bb2['ymin']:
        ymin = bb1['ymin']
    else:
        ymin = bb2['ymin']

    if bb1['xmin'] < bb2['xmin']:
        xmin = bb1['xmin']
    else:
        xmin = bb2['xmin']

    if bb1['xmax'] > bb2['xmax']:
        xmax = bb1['xmax']
    else:
        xmax = bb2['xmax']

    if bb1['ymax'] > bb2['ymax']:
        ymax = bb1['ymax']
    else:
        ymax = bb2['ymax']

    if bb1['conf'] > bb2['conf']:
        conf = bb1['conf']
    else:
        conf = bb2['conf'] 

    return {'id': new_id, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'conf': conf}


def analyze_list_boxes(bb_slice1, bb_slice2, ori="vert"):
    bb_img = []
    erase_list = []
    margin_h = 5
    margin_v = 15
    
    if ori == "vert":
        for bb1 in bb_slice1:
            if bb1['ymax'] > (tam_slice_height-margin_h):
                for bb2 in bb_slice2:
                    if bb2['id'] in erase_list:
                        continue
                
                    if bb2['ymin'] < margin_h:  
                        if calc_dif(bb1['xmin'],bb2['xmin']) < margin_v and calc_dif(bb1['xmax'],bb2['xmax']) < margin_v:
                            erase_list.append(bb1['id'])
                            erase_list.append(bb2['id'])
                            bb1_aux =  {'id': 0, 'xmin': bb1['xmin'] + tam_slice_width , 'ymin': bb1['ymin'] , 'xmax': bb1['xmax'] + tam_slice_width, 'ymax': bb1['ymax'], 'conf': bb1['conf']}
                            bb2_aux =  {'id': 0, 'xmin': bb2['xmin'] + tam_slice_width , 'ymin': bb2['ymin'] + tam_slice_height, 'xmax': bb2['xmax'] + tam_slice_width, 'ymax': bb2['ymax'] + tam_slice_height, 'conf': bb2['conf']}
                            bb_img.append(merge_boxes(bb1_aux,bb2_aux,bb1['id']))

    else:
        for bb1 in bb_slice1:
            if bb1['xmax'] > (tam_slice_width-margin_h):
                for bb2 in bb_slice2:
                    if bb2['id'] in erase_list:
                        continue

                    if bb2['xmin'] < margin_h:  
                        if calc_dif(bb1['ymin'],bb2['ymin']) < margin_v and calc_dif(bb1['ymax'],bb2['ymax']) < margin_v:
                                erase_list.append(bb1['id'])
                                erase_list.append(bb2['id'])
                                bb1_aux =  {'id': 0, 'xmin': bb1['xmin'], 'ymin': bb1['ymin'] + tam_slice_height, 'xmax': bb1['xmax'], 'ymax': bb1['ymax'] + tam_slice_height, 'conf': bb1['conf']}
                                bb2_aux =  {'id': 0, 'xmin': bb2['xmin'] + tam_slice_width , 'ymin': bb2['ymin'] + tam_slice_height, 'xmax': bb2['xmax'] + tam_slice_width, 'ymax': bb2['ymax'] + tam_slice_height, 'conf': bb2['conf']}
                                bb_img.append(merge_boxes(bb1_aux,bb2_aux,bb1['id'])) 
    
    return bb_img, erase_list
    
def fix_bounding_box(tam_total_width, tam_total_height, tam_slice_width, tam_slice_height,
                     bb_slice0, bb_slice1, bb_slice2, bb_slice3):
    bb_img = []
    erase_list = []

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

    
    
    #Merge entre Q0 e Q1
    bb_img_aux, erase_list_aux = analyze_list_boxes(bb_slice0, bb_slice1, ori="hori")
    bb_img += bb_img_aux
    erase_list += erase_list_aux

    #Merge entre Q0 e Q2
    bb_img_aux, erase_list_aux = analyze_list_boxes(bb_slice0, bb_slice2, ori="vert")
    bb_img += bb_img_aux
    erase_list += erase_list_aux
    

    #Merge entre Q1 e Q3
    bb_img_aux, erase_list_aux = analyze_list_boxes(bb_slice1, bb_slice3, ori="vert")
    bb_img += bb_img_aux
    erase_list += erase_list_aux

    #Merge entre Q2 e Q3
    bb_img_aux, erase_list_aux = analyze_list_boxes(bb_slice2, bb_slice3, ori="hori")
    bb_img += bb_img_aux
    erase_list += erase_list_aux

    
    for bb0 in bb_slice0:
        if bb0['id'] in erase_list:
            continue
        bb_img.append(bb0)

    for bb1 in bb_slice1:
        if bb1['id'] in erase_list:
            continue
        bb1 =  {'id': bb1['id'], 'xmin': bb1['xmin'] + tam_slice_width , 'ymin': bb1['ymin'] , 'xmax': bb1['xmax'] + tam_slice_width, 'ymax': bb1['ymax'], 'conf': bb1['conf']}
        bb_img.append(bb1)                       

    for bb2 in bb_slice2:
        if bb2['id'] in erase_list:
            continue
        bb2 =  {'id': bb2['id'], 'xmin': bb2['xmin'], 'ymin': bb2['ymin'] + tam_slice_height, 'xmax': bb2['xmax'], 'ymax': bb2['ymax'] + tam_slice_height, 'conf': bb2['conf']}
        bb_img.append(bb2)

    for bb3 in bb_slice3:
        if bb3['id'] in erase_list:
            continue
        bb3 =  {'id': bb3['id'], 'xmin': bb3['xmin'] + tam_slice_width , 'ymin': bb3['ymin'] + tam_slice_height, 'xmax': bb3['xmax'] + tam_slice_width, 'ymax': bb3['ymax'] + tam_slice_height, 'conf': bb3['conf']}
        bb_img.append(bb3)

   
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
