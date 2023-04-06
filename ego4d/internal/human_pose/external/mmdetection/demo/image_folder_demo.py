# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import os
from tqdm import tqdm
import tempfile
import shutil
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img_folder', help='Image Folder')
    parser.add_argument('--out_folder', default=None, help='Path to output Folder')
    parser.add_argument('--max_image_size', default=640, help='Path to output Folder')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    image_size = int(args.max_image_size)

    ## make a temporary folder and delete it after the inference is done
    temp_dir = tempfile.mkdtemp()

    for img_name in tqdm(sorted(os.listdir(args.img_folder))):
        ## read and resize the image 
        img = cv2.imread(os.path.join(args.img_folder, img_name))

        ## resize keeping the aspect ratio
        h, w, _ = img.shape
        if h > w:
            img = cv2.resize(img, (int(image_size * w / h), image_size))
        else:
            img = cv2.resize(img, (image_size, int(image_size * h / w)))

        ## write to temporary folder
        cv2.imwrite(os.path.join(temp_dir, img_name), img)

        # test a single image
        result = inference_detector(model, os.path.join(temp_dir, img_name))
        
        ## only use person category
        bbox_result, segm_result = result
        person_bbox_result = bbox_result[0]
        person_segm_result = segm_result[0]

        bbox_result = [person_bbox_result] + [] * (len(bbox_result) - 1)
        segm_result = [person_segm_result] + [] * (len(segm_result) - 1)

        result = bbox_result, segm_result

        # show the results
        show_result_pyplot(
            model,
            os.path.join(temp_dir, img_name),
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=os.path.join(args.out_folder, img_name))

    ## delete the temporary folder
    shutil.rmtree(temp_dir)
    print('Done, results are saved in {}'.format(args.out_folder))
    return

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
