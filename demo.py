import argparse
import os
import random

import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from model import model_static

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    dlib = None
    DLIB_AVAILABLE = False

HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
HAAR_AVAILABLE = os.path.exists(HAAR_CASCADE_PATH)


parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('--max_frames', type=int, help='process only the first N frames (for debugging)', default=None)
parser.add_argument('--quiet', help='suppress per-frame logging', action='store_true')

args = parser.parse_args()

CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2


def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def run(video_path, face_path, model_weight, jitter, quiet, max_frames=None):
    device = select_device()

    print(f'Running inference on {device}')

    # set up video source
    if video_path is None:
        cap = cv2.VideoCapture(0)
        source_name = 'webcam'
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = video_path

    # set up face detection mode
    if face_path is None:
        if DLIB_AVAILABLE:
            facemode = 'DLIB'
        elif HAAR_AVAILABLE:
            facemode = 'HAAR'
        else:
            raise RuntimeError('Face detections are required. Install dlib or ensure OpenCV haar cascades are available.')
    else:
        facemode = 'GIVEN'
        column_names = ['frame', 'left', 'top', 'right', 'bottom']
        df = pd.read_csv(face_path, names=column_names, index_col=0)
        df['left'] -= (df['right']-df['left'])*0.2
        df['right'] += (df['right']-df['left'])*0.2
        df['top'] -= (df['bottom']-df['top'])*0.1
        df['bottom'] += (df['bottom']-df['top'])*0.1
        df['left'] = df['left'].astype('int')
        df['top'] = df['top'].astype('int')
        df['right'] = df['right'].astype('int')
        df['bottom'] = df['bottom'].astype('int')

    if not cap.isOpened():
        raise RuntimeError("Error opening video stream or file. For live mode, ensure camera permissions are granted or provide --video.")

    haar_detector = None
    if facemode == 'DLIB':
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    elif facemode == 'HAAR':
        haar_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if haar_detector.empty():
            raise RuntimeError(f'Failed to load Haar cascade from {HAAR_CASCADE_PATH}.')
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static()
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight, map_location=device)
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    # video reading loop
    while cap.isOpened():
        if max_frames is not None and frame_cnt >= max_frames:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        height, width, channels = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_cnt += 1
        bbox = []
        if facemode == 'DLIB':
            dets = cnn_face_detector(frame_rgb, 1)
            for d in dets:
                l = d.rect.left()
                r = d.rect.right()
                t = d.rect.top()
                b = d.rect.bottom()
                # expand a bit
                l -= (r-l)*0.2
                r += (r-l)*0.2
                t -= (b-t)*0.2
                b += (b-t)*0.2
                bbox.append([l, t, r, b])
        elif facemode == 'HAAR':
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                l = x
                r = x + w
                t = y
                b = y + h
                # expand similarly to dlib branch
                expand_w = (r - l) * 0.2
                expand_h = (b - t) * 0.2
                l -= expand_w
                r += expand_w
                t -= expand_h
                b += expand_h
                bbox.append([l, t, r, b])
        elif facemode == 'GIVEN':
            if frame_cnt in df.index:
                bbox.append([df.loc[frame_cnt, 'left'], df.loc[frame_cnt, 'top'], df.loc[frame_cnt, 'right'], df.loc[frame_cnt, 'bottom']])

        # sanitize bbox coordinates to stay within frame bounds
        sanitized_bbox = []
        for l, t, r, b in bbox:
            l = max(0, min(int(round(l)), width - 1))
            t = max(0, min(int(round(t)), height - 1))
            r = max(0, min(int(round(r)), width - 1))
            b = max(0, min(int(round(b)), height - 1))
            if r <= l or b <= t:
                continue
            sanitized_bbox.append([l, t, r, b])

        frame_pil = Image.fromarray(frame_rgb)
        frame_scores = []
        for b_left, b_top, b_right, b_bottom in sanitized_bbox:
            face = frame_pil.crop((b_left, b_top, b_right, b_bottom))
            img = test_transforms(face).unsqueeze(0)
            samples = [img]
            if jitter > 0:
                for _ in range(jitter):
                    bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b_left, b_top, b_right, b_bottom)
                    bj_left = max(0, min(int(round(bj_left)), width - 1))
                    bj_top = max(0, min(int(round(bj_top)), height - 1))
                    bj_right = max(0, min(int(round(bj_right)), width - 1))
                    bj_bottom = max(0, min(int(round(bj_bottom)), height - 1))
                    if bj_right <= bj_left or bj_bottom <= bj_top:
                        continue
                    facej = frame_pil.crop((bj_left, bj_top, bj_right, bj_bottom))
                    img_jittered = test_transforms(facej).unsqueeze(0)
                    samples.append(img_jittered)

            imgs = torch.cat(samples, dim=0).to(device)

            # forward pass
            with torch.no_grad():
                output = model(imgs)
            if jitter > 0:
                output = torch.mean(output, dim=0)
            score = torch.sigmoid(output).item()
            frame_scores.append(score)

        if not quiet:
            if frame_scores:
                scores_fmt = ', '.join(f'{s:.3f}' for s in frame_scores)
                print(f'[{source_name}] frame {frame_cnt}: scores={scores_fmt}')
            else:
                print(f'[{source_name}] frame {frame_cnt}: no face detected')

    cap.release()
    print('DONE!')


if __name__ == "__main__":
    run(args.video, args.face, args.model_weight, args.jitter, args.quiet, args.max_frames)
