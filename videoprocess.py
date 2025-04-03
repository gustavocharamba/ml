import shutil
import numpy as np
import cv2
import os

def getFrames(video_path="video.mp4", frame_skip=60, patch_size=128, patches_per_frame=15):
    # Remover e recriar as pastas
    if os.path.exists("frames"):
        shutil.rmtree("frames")
    os.makedirs("frames")

    if os.path.exists("128"):
        shutil.rmtree("128")
    os.makedirs("128")

    capture = cv2.VideoCapture(video_path)
    aux = 0
    frame_count = 0

    while True:
        flag, frame = capture.read()
        if not flag:
            break

        if frame_count % frame_skip == 0:
            # Salvando frame completo em RGB na pasta "frames"
            frame_name = os.path.join("frames", f'frame_{aux:04d}.png')
            cv2.imwrite(frame_name, frame)

            # Gerando patches aleatórios de 128x128
            h, w, _ = frame.shape
            for i in range(patches_per_frame):
                # Escolhendo uma posição aleatória dentro do frame
                x = np.random.randint(0, max(1, w - patch_size))
                y = np.random.randint(0, max(1, h - patch_size))

                # Recortando a região 128x128
                patch = frame[y:y + patch_size, x:x + patch_size].copy()

                # Salvando o patch original na pasta "128"
                patch_name = os.path.join("128", f'frame_{aux:04d}_original_{i}.png')
                cv2.imwrite(patch_name, patch)

                # Aplicando espelhamento horizontal
                patch_flipped = cv2.flip(patch, 1)

                # Salvando o patch espelhado na pasta "256"
                flipped_name = os.path.join("128", f'frame_{aux:04d}_flipped_{i}.png')
                cv2.imwrite(flipped_name, patch_flipped)

            aux += 1

        frame_count += 1

    capture.release()
    print("Salvo.")
getFrames()
