import argparse
import os
from fractions import Fraction
from pathlib import Path
import av
import cv2
import torch
import numpy as np
from tqdm import tqdm

from models.unet import UNet
from utils import get_checkpoint_path, lab_tensor_to_rgb, rgb_to_lab_tensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path: str | None = None, device: torch.device = DEVICE) -> UNet:
    if checkpoint_path is None:
        resolved_path = get_checkpoint_path()
        if resolved_path is None:
            expected = Path(__file__).parent / "checkpoints"
            raise FileNotFoundError(
                f"No checkpoint found. Expected {expected / 'best.pth'} "
                f"(fallback: {expected / 'latest.pth'})."
            )
        checkpoint_path = str(resolved_path)
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def colorize_image(model: UNet, image_path: str, output_path: str) -> None:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))
    # Convert to LAB tensor (L + ab)
    lab = rgb_to_lab_tensor(img_resized / 255.0)
    L = lab[0].unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,256,256)
    with torch.no_grad():
        pred_ab = model(L)
    # Combine L and predicted ab
    pred_lab = torch.cat([L.squeeze(0), pred_ab.squeeze(0)], dim=0)
    colorized = lab_tensor_to_rgb(pred_lab)
    # Resize back to original size
    colorized = cv2.resize(colorized, (img.shape[1], img.shape[0]))
    cv2.imwrite(output_path, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
    print(f"Saved colorized image to {output_path}")

def colorize_video(model: UNet, video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_rate = Fraction(fps).limit_denominator(1001)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = width + width % 2
    output_height = height + height % 2
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing video')
    try:
        with av.open(output_path, mode='w') as output:
            output_stream = output.add_stream('libx264', rate=frame_rate)
            output_stream.width = output_width
            output_stream.height = output_height
            output_stream.pix_fmt = 'yuv420p'
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (256, 256))
                lab = rgb_to_lab_tensor(rgb_resized / 255.0)
                L = lab[0].unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred_ab = model(L)
                pred_lab = torch.cat([L.squeeze(0), pred_ab.squeeze(0)], dim=0)
                colorized = lab_tensor_to_rgb(pred_lab)
                colorized = cv2.resize(colorized, (width, height))
                if output_width != width or output_height != height:
                    colorized = cv2.copyMakeBorder(
                        colorized,
                        0,
                        output_height - height,
                        0,
                        output_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                output_frame = av.VideoFrame.from_ndarray(
                    cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR),
                    format='bgr24',
                )
                for packet in output_stream.encode(output_frame):
                    output.mux(packet)
                pbar.update(1)
            for packet in output_stream.encode():
                output.mux(packet)
    finally:
        pbar.close()
        cap.release()
    print(f"Saved colorized video to {output_path}")

def colorize_webcam(model: UNet) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (256, 256))
            lab = rgb_to_lab_tensor(rgb_resized / 255.0)
            L = lab[0].unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_ab = model(L)
            pred_lab = torch.cat([L.squeeze(0), pred_ab.squeeze(0)], dim=0)
            colorized = lab_tensor_to_rgb(pred_lab)
            colorized = cv2.resize(colorized, (frame.shape[1], frame.shape[0]))
            # Overlay FPS
            cv2.putText(colorized, 'Press Q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('Real‑Time Colorization', cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def export_onnx(model: UNet, onnx_path: str) -> None:
    dummy_input = torch.randn(1, 1, 256, 256, device=DEVICE)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['L'], output_names=['ab'], opset_version=14)
    print(f"Exported ONNX model to {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference utilities for Lab colorization')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (defaults to best, then latest)')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    img_parser = subparsers.add_parser('image')
    img_parser.add_argument('--input', type=str, required=True)
    img_parser.add_argument('--output', type=str, required=True)
    vid_parser = subparsers.add_parser('video')
    vid_parser.add_argument('--input', type=str, required=True)
    vid_parser.add_argument('--output', type=str, required=True)
    subparsers.add_parser('webcam')
    onnx_parser = subparsers.add_parser('export')
    onnx_parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    if args.mode == 'image':
        colorize_image(model, args.input, args.output)
    elif args.mode == 'video':
        colorize_video(model, args.input, args.output)
    elif args.mode == 'webcam':
        colorize_webcam(model)
    elif args.mode == 'export':
        export_onnx(model, args.output)
